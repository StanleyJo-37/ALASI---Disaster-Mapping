import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent   # .../ai/training
AI_DIR = SCRIPT_DIR.parent                     # .../ai
PROJECT_DIR = AI_DIR.parent                    # .../vision-ai-development
LIB_DIR = AI_DIR / 'lib'

os.chdir(AI_DIR)

for p in (AI_DIR, PROJECT_DIR, LIB_DIR):
  if str(p) not in sys.path:
    sys.path.append(str(p))

print('Importing Dependencies..')

from types import SimpleNamespace
import gc
import csv
from dotenv import load_dotenv

import torch
from peft import LoraConfig, get_peft_model
from ultralytics.utils.loss import SemanticSegmentationLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset
from huggingface_hub import snapshot_download
import numpy as np

from datasets.rescuenet_dataset import RescueNetDataset, collate_fn
from utils.augmentations import get_augmentation_pipeline
from model.TriheadSegmentationModel import TriheadSegmentationModel
from model.UncertaintyLossWeighting import UncertaintyLossWeighting
from utils.training import SSILoss, compute_normal_loss, EarlyStoppingAndCheckpointing
from custom_types.training import AblationStudyType
from utils.runpod import end_session
from utils.storage import upload_folder_to_huggingface

print('Loading variables..')

load_dotenv()

MODEL_WEIGHT_DIR = 'model/weights'
YOLO_WEIGHTS = f'{MODEL_WEIGHT_DIR}/yolo26m-sem.pt'
BACKGROUND_CLASS = 0
NUM_CLASSES = 11

TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
TRAIN_SUBSET_FRACTION = 4
VAL_SUBSET_SIZE = 128

TOTAL_EPOCHS = 5
TOTAL_WARMUP_STEPS = 2
TOTAL_STATIC_STEPS = 3

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
print(f'Device used: {device}')

CLASS_WEIGHTS = torch.tensor([
  0.14733338,
  0.39315039,
  0.66941495,
  0.65486371,
  0.82150387,
  0.88164306,
  1.84860004,
  0.40923846,
  0.85621029,
  0.23214045,
  4.0859014
], device=device)
assert CLASS_WEIGHTS.numel() == NUM_CLASSES

rng = np.random.default_rng(13)

print('Downloading dataset..')

# os.makedirs('./data', exist_ok=True)
# snapshot_download(
#   repo_id=os.environ.get('HF_DATASET_REPO_ID'),
#   repo_type="dataset",
#   local_dir="./data",
#   token=os.environ.get('HF_TOKEN')
# )

print('Defining functions..')

TASK_CONFIG: dict[str, tuple[bool, bool]] = {
  'vanilla':           (False, False),
  'additional-depth':  (True,  False),
  'additional-normal': (False, True),
  'additional-both':   (True,  True),
}

def set_training_mode(model: torch.nn.Module, mode: bool = True):
    """Sets mode, but ensures backbone BatchNorms stay frozen."""
    model.train(mode)
        
    if hasattr(model, 'yolo_backbone'):
      for m in model.yolo_backbone.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
          m.eval()
                
    return model

def create_peft_model(model: TriheadSegmentationModel):
  valid_target_modules = []
  
  backbone_layers = list(model.yolo_backbone.model.children())[:11]
    
  for layer_idx, layer in enumerate(backbone_layers):
    for name, module in layer.named_modules():
      if isinstance(module, torch.nn.Conv2d) and module.groups == 1:
        full_name = f"model.{layer_idx}.{name}" if name else f"model.{layer_idx}"
        valid_target_modules.append(full_name)
  
  lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=valid_target_modules,
    bias="none",
  )

  peft_backbone = get_peft_model(model.yolo_backbone, peft_config=lora_config).to(device)

  for name, param in peft_backbone.named_parameters():
    if 'model.17' in name:
      param.requires_grad = True
  
  peft_backbone.print_trainable_parameters()
  peft_backbone.to(device)

  model.yolo_backbone = peft_backbone
  return model.to(device)

def infuse_args(model: TriheadSegmentationModel):
  current_args = model.yolo_backbone.args if isinstance(model.yolo_backbone.args, dict) else {}
  current_args['nc'] = NUM_CLASSES
  model.yolo_backbone.args = SimpleNamespace(**current_args)

def get_model(model_type: AblationStudyType):
  include_depth, include_normals = TASK_CONFIG[model_type]

  model = TriheadSegmentationModel(
    yolo_pt_path=YOLO_WEIGHTS,
    include_depth=include_depth,
    include_normals=include_normals,
    device=device
  )

  infuse_args(model)
  raw_yolo_architecture = model.yolo_backbone
  final_model = create_peft_model(model)
  loss_balancer = UncertaintyLossWeighting().to(device=device)

  return raw_yolo_architecture, final_model, loss_balancer

spatial_aug, photometric_aug = get_augmentation_pipeline()

def get_dataset_and_loader(model_type: AblationStudyType):
  include_depth, include_normals = TASK_CONFIG[model_type]

  full_train_dataset = RescueNetDataset(
    data_dir='data/RescueNet/train',
    include_depth=include_depth,
    include_normals=include_normals,
    spatial_transform=spatial_aug,
    photometric_transform=photometric_aug,
    training=True
  )
  subset_size = len(full_train_dataset) // TRAIN_SUBSET_FRACTION
  indices = rng.choice(len(full_train_dataset), size=subset_size, replace=False)
  train_dataset = Subset(full_train_dataset, indices)

  # Class pixel distribution of the sampled subset (sanity check vs CLASS_WEIGHTS)
  class_pixel_counts = np.zeros(NUM_CLASSES)
  for idx in indices:
    label = np.load(full_train_dataset.label_paths[idx]).squeeze()
    class_pixel_counts += np.bincount(label.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
  print(class_pixel_counts)

  val_dataset = RescueNetDataset(
    data_dir='data/RescueNet/val',
    include_depth=include_depth,
    include_normals=include_normals
  )
  val_dataset = Subset(val_dataset, list(range(VAL_SUBSET_SIZE)))

  train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    # persistent_workers=True,
    # num_workers=8,
    # pin_memory=True,
    # prefetch_factor=2
  )

  val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    # persistent_workers=True,
    # num_workers=4,
    # pin_memory=True,
    # prefetch_factor=2
  )

  return {
    'train': (train_dataset, train_loader),
    'val': (val_dataset, val_loader),
  }

print('Start training - ablation study')

for model_type in [
  # 'vanilla',
  'additional-both',
  # 'additional-normal',
  # 'additional-depth'
]:
  dataset_and_loader = get_dataset_and_loader(model_type)
  raw_yolo_architecture, final_model, loss_balancer = get_model(model_type)  
  include_depth, include_normals = TASK_CONFIG[model_type]

  trainable_params = [p for p in final_model.parameters() if p.requires_grad]
  trainable_params.extend(loss_balancer.parameters())

  model_params = [p for p in final_model.parameters() if p.requires_grad]
  optimizer = AdamW(
    [
      {'params': model_params, 'weight_decay': 5e-2},
      {'params': list(loss_balancer.parameters()), 'weight_decay': 0.0},
    ],
    lr=2e-4,
  )
  
  warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min((step + 1) / TOTAL_WARMUP_STEPS, 1.0)
  )
  static_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda _: 1.0
  )
  cosine_annealing_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=max(1, TOTAL_EPOCHS - TOTAL_WARMUP_STEPS - TOTAL_STATIC_STEPS),
    eta_min=1e-6
  )
  scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, static_scheduler, cosine_annealing_scheduler],
    milestones=[TOTAL_WARMUP_STEPS, TOTAL_WARMUP_STEPS+TOTAL_STATIC_STEPS]
  )

  seg_loss_criterion = SemanticSegmentationLoss(raw_yolo_architecture)
  seg_loss_criterion.ce = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
  depth_loss_criterion = SSILoss()

  early_stopping = EarlyStoppingAndCheckpointing(patience=5, delta=0.01, save_per_epoch=1)

  train_loader = dataset_and_loader['train'][1]
  val_loader = dataset_and_loader['val'][1]

  epoch_history = []
  train_loss_history = []
  val_loss_history = []

  for epoch in range(1, TOTAL_EPOCHS + 1):
    epoch_train_loss = torch.zeros((), device=device) 
    epoch_train_seg_loss = torch.zeros((), device=device)
    epoch_weighted_train_seg_loss = torch.zeros((), device=device)
    epoch_train_depth_loss = torch.zeros((), device=device)
    epoch_weighted_train_depth_loss = torch.zeros((), device=device)
    epoch_train_normal_loss = torch.zeros((), device=device)
    epoch_weighted_train_normal_loss = torch.zeros((), device=device)

    set_training_mode(final_model, True)
    
    for batch_idx, (batch_images, batch_targets) in enumerate(train_loader, 1):
      optimizer.zero_grad(set_to_none=True)

      with autocast(device_type=device_name, dtype=torch.bfloat16):
        segmentation_out, depth_out, normal_out = final_model(batch_images.to(device=device))

        true_segmentation_map = batch_targets[0]
        
        true_depth_map = None
        if include_depth:
          true_depth_map = batch_targets[1]['depth'].to(device=device)
        
        true_surface_normals = batch_targets[2]['normals'].to(device=device) if include_normals else None

        seg_loss, seg_loss_items = seg_loss_criterion(segmentation_out, true_segmentation_map)
        seg_loss /= batch_images.shape[0]
        depth_loss = depth_loss_criterion(depth_out, true_depth_map) if include_depth else torch.tensor(0.0, device=device)
        normal_loss = compute_normal_loss(normal_out, true_surface_normals) if include_normals else torch.tensor(0.0, device=device)

        weighted_seg_loss = seg_loss
        weighted_depth_loss = depth_loss
        weighted_normal_loss = normal_loss

        if model_type == 'vanilla':
          loss_total = seg_loss
        else:
          loss_total, weighted_seg_loss, weighted_depth_loss, weighted_normal_loss = loss_balancer(
            seg_loss,
            depth_loss if include_depth else None,
            normal_loss if include_normals else None
          )

      loss_total.backward()
      optimizer.step()

      epoch_train_loss += loss_total.mean().detach()
      epoch_train_seg_loss += seg_loss.mean().detach()
      epoch_weighted_train_seg_loss += weighted_seg_loss.mean().detach()
      epoch_train_depth_loss += depth_loss.mean().detach()
      epoch_weighted_train_depth_loss += weighted_depth_loss.mean().detach()
      epoch_train_normal_loss += normal_loss.mean().detach()
      epoch_weighted_train_normal_loss += weighted_normal_loss.mean().detach()

      print(
        f"Epoch [{epoch:03d}/{TOTAL_EPOCHS:03d}] Batch [{batch_idx:04d}/{len(train_loader):04d}] | "
        f"Batch Train Loss: {loss_total.item():.4f} │ "
        f"LR: {optimizer.param_groups[0]['lr']:.2e}", end='\r'
      )

    scheduler.step()
    torch.cuda.empty_cache()

    avg_train_loss = (epoch_train_loss / len(train_loader)).item()
    avg_train_seg_loss = (epoch_train_seg_loss / len(train_loader)).item()
    avg_weighted_train_seg_loss = (epoch_weighted_train_seg_loss / len(train_loader)).item()
    avg_train_depth_loss = (epoch_train_depth_loss / len(train_loader)).item()
    avg_weighted_train_depth_loss = (epoch_weighted_train_depth_loss / len(train_loader)).item()
    avg_train_normal_loss = (epoch_train_normal_loss / len(train_loader)).item()
    avg_weighted_train_normal_loss = (epoch_weighted_train_normal_loss / len(train_loader)).item()

    epoch_val_loss = torch.zeros((), device=device)
    epoch_val_seg_loss = torch.zeros((), device=device)
    epoch_weighted_val_seg_loss = torch.zeros((), device=device)
    epoch_val_depth_loss = torch.zeros((), device=device)
    epoch_weighted_val_depth_loss = torch.zeros((), device=device)
    epoch_val_normal_loss = torch.zeros((), device=device)
    epoch_weighted_val_normal_loss = torch.zeros((), device=device)

    set_training_mode(final_model, False)
    final_model.yolo_backbone.eval()
    with torch.no_grad():
      for batch_images_val, batch_targets_val in val_loader:
        with autocast(device_type=device_name, dtype=torch.bfloat16):
          segmentation_out, depth_out, normal_out = final_model(batch_images_val.to(device=device))

          true_segmentation_map = batch_targets_val[0]
          
          true_depth_map = None
          if include_depth:
            true_depth_map = batch_targets_val[1]['depth'].to(device=device)
          
          true_surface_normals = batch_targets_val[2]['normals'].to(device=device) if include_normals else None

          seg_loss, seg_loss_items = seg_loss_criterion(segmentation_out, true_segmentation_map)
          seg_loss /= batch_images_val.shape[0]
          depth_loss = depth_loss_criterion(depth_out, true_depth_map) if include_depth else torch.tensor(0.0, device=device)
          normal_loss = compute_normal_loss(normal_out, true_surface_normals) if include_normals else torch.tensor(0.0, device=device)

          weighted_seg_loss = seg_loss
          weighted_depth_loss = depth_loss
          weighted_normal_loss = normal_loss

          if model_type == 'vanilla':
            val_loss_total = seg_loss
          else:
            val_loss_total, weighted_seg_loss, weighted_depth_loss, weighted_normal_loss = loss_balancer(
              seg_loss,
              depth_loss if include_depth else None,
              normal_loss if include_normals else None
            )

          epoch_val_loss += val_loss_total.mean().detach()
          epoch_val_seg_loss += seg_loss.mean().detach()
          epoch_weighted_val_seg_loss += weighted_seg_loss.mean().detach()
          epoch_val_depth_loss += depth_loss.mean().detach()
          epoch_weighted_val_depth_loss += weighted_depth_loss.mean().detach()
          epoch_val_normal_loss += normal_loss.mean().detach()
          epoch_weighted_val_normal_loss += weighted_normal_loss.mean().detach()

    avg_val_loss = (epoch_val_loss / len(val_loader)).item()
    avg_val_seg_loss = (epoch_val_seg_loss / len(val_loader)).item()
    avg_weighted_val_seg_loss = (epoch_weighted_val_seg_loss / len(val_loader)).item()
    avg_val_depth_loss = (epoch_val_depth_loss / len(val_loader)).item()
    avg_weighted_val_depth_loss = (epoch_weighted_val_depth_loss / len(val_loader)).item()
    avg_val_normal_loss = (epoch_val_normal_loss / len(val_loader)).item()
    avg_weighted_val_normal_loss = (epoch_weighted_val_normal_loss / len(val_loader)).item()

    # Record the metrics
    epoch_history.append(epoch)
    train_loss_history.append({
      'loss': avg_train_loss,
      'seg_loss': avg_train_seg_loss,
      'weighted_seg_loss': avg_weighted_train_seg_loss,
      'depth_loss': avg_train_depth_loss,
      'weighted_depth_loss': avg_weighted_train_depth_loss,
      'normal_loss': avg_train_normal_loss,
      'weighted_normal_loss': avg_weighted_train_normal_loss
    })
    val_loss_history.append({
      'loss': avg_val_loss,
      'seg_loss': avg_val_seg_loss,
      'weighted_seg_loss': avg_weighted_val_seg_loss,
      'depth_loss': avg_val_depth_loss,
      'weighted_depth_loss': avg_weighted_val_depth_loss,
      'normal_loss': avg_val_normal_loss,
      'weighted_normal_loss': avg_weighted_val_normal_loss
    })

    # Early Stopping evaluated ONCE per epoch using the average validation loss
    halt = early_stopping.record_and_check_if_halt(
      avg_val_seg_loss,
      final_model.state_dict(),
      loss_balancer.state_dict()
    )
    print(
      f"Epoch [{epoch:03d}/{TOTAL_EPOCHS:03d}] Batch [{batch_idx:04d}/{len(train_loader):04d}] | "
      f"LR: {optimizer.param_groups[0]['lr']:.6f}\n",
      f"============================================\n"
      f"---TRAINING---\n"
      f"- Total Loss: {avg_train_loss:.4f}\n"
      f"- Seg | Weighted: {avg_train_seg_loss:.4f} | {avg_weighted_train_seg_loss:.4f}\n"
      f"- Depth | Weighted: {avg_train_depth_loss:.4f} | {avg_weighted_train_depth_loss:.4f}\n"
      f"- Norm | Weighted: {avg_train_normal_loss:.4f} | {avg_weighted_train_normal_loss:.4f}\n"
      f"============================================\n"
      f"---VALIDATION---\n"
      f"- Total Loss: {avg_val_loss:.4f}\n"
      f"- Seg | Weighted: {avg_val_seg_loss:.4f} | {avg_weighted_val_seg_loss:.4f}\n"
      f"- Depth | Weighted: {avg_val_depth_loss:.4f} | {avg_weighted_val_depth_loss:.4f}\n"
      f"- Norm | Weighted: {avg_val_normal_loss:.4f} | {avg_weighted_val_normal_loss:.4f}\n"
      f"============================================\n"
      f"---PENALTY TERMS---\n"
      f"Segmentation Penalty Term: {loss_balancer.alpha.item():.4f}\n"
      f"Depth Penalty Term: {loss_balancer.beta.item():.4f}\n"
      f"Surface Normal Penalty Term: {loss_balancer.gamma.item():.4f}\n"
    )

    if halt:
      break

  early_stopping.save_weights('./', model_type, final_model.state_dict(), loss_balancer.state_dict())

  os.makedirs('eval_results', exist_ok=True)
  csv_filename = f"eval_results/{model_type}_loss_history.csv"
  with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
      'epoch',
      'train_loss',
      'train_seg_loss',
      'train_weighted_seg_loss',
      'train_depth_loss',
      'train_weighted_depth_loss',
      'train_normal_loss',
      'train_weighted_normal_loss',
      'val_loss',
      'val_seg_loss',
      'val_weighted_seg_loss',
      'val_depth_loss',
      'val_weighted_depth_loss',
      'val_normal_loss',
      'val_weighted_normal_loss',
    ])
    for i in range(len(epoch_history)):
      writer.writerow([
        epoch_history[i], 
        train_loss_history[i]['loss'],
        train_loss_history[i]['seg_loss'],
        train_loss_history[i]['weighted_seg_loss'],
        train_loss_history[i]['depth_loss'],
        train_loss_history[i]['weighted_depth_loss'],
        train_loss_history[i]['normal_loss'],
        train_loss_history[i]['weighted_normal_loss'],
        val_loss_history[i]['loss'],
        val_loss_history[i]['seg_loss'],
        val_loss_history[i]['weighted_seg_loss'],
        val_loss_history[i]['depth_loss'],
        val_loss_history[i]['weighted_depth_loss'],
        val_loss_history[i]['normal_loss'],
        val_loss_history[i]['weighted_normal_loss'],
      ])
  print(f"Saved training history to {csv_filename}")

  # upload_folder_to_huggingface(
  #   'weights',
  #   'weights'
  # )
  # upload_folder_to_huggingface(
  #   'eval_results',
  #   'eval_results'
  # )
  
  # Cleanup
  if torch.cuda.is_available():
    alloc_before = torch.cuda.memory_allocated() / (1024 ** 3)
    res_before = torch.cuda.memory_reserved() / (1024 ** 3)

  del (
    final_model, raw_yolo_architecture, loss_balancer,
    seg_loss_criterion, depth_loss_criterion, early_stopping,
    optimizer, scheduler, train_loader, val_loader, dataset_and_loader,
  )
  gc.collect()

  if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    alloc_after = torch.cuda.memory_allocated() / (1024 ** 3)
    res_after = torch.cuda.memory_reserved() / (1024 ** 3)

    print(f"VRAM Allocated: {alloc_before:.2f} GB  ->  {alloc_after:.2f} GB")
    print(f"VRAM Reserved:  {res_before:.2f} GB  ->  {res_after:.2f} GB")
    print("✅ PyTorch CUDA cache successfully flushed!")
  else:
    print("⚠️ CUDA not detected. Only system RAM was flushed.")


# end_session()
