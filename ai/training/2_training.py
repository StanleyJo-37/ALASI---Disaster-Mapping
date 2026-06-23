import os
import sys

ROOT_PATH = '../'
os.chdir(ROOT_PATH)

sys.path.append(os.path.abspath(ROOT_PATH))

AI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(AI_DIR)

if AI_DIR not in sys.path:
  sys.path.append(AI_DIR)

lib_path = os.path.abspath(os.path.join(ROOT_PATH, 'lib'))
if lib_path not in sys.path:
  sys.path.append(lib_path)

print('Importing Dependencies..')
from types import SimpleNamespace
import gc
import csv
from dotenv import load_dotenv

import torch
from peft import LoraConfig, get_peft_model
from ultralytics.utils.loss import v8SegmentationLoss, E2ELoss
from torch.optim import AdamW
from torch.nn import L1Loss
from torch.amp import autocast, GradScaler
from huggingface_hub import snapshot_download

from datasets.rescuenet_dataset import RescueNetDataset, collate_fn
from utils.augmentations import get_augmentation_pipeline
from model.TriheadSegmentationModel import TriheadSegmentationModel
from model.UncertaintyLossWeighting import UncertaintyLossWeighting
from utils.training import compute_normal_loss, EarlyStoppingAndCheckpointing
from custom_types.training import AblationStudyType
from utils.runpod import terminate_session
from utils.storage import upload_folder_to_huggingface

print('Loading variables..')
load_dotenv()
MODEL_WEIGHT_DIR = 'model/weights'
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
print(device)

print('Downloading dataset..')
os.makedirs('./data', exist_ok=True)
snapshot_download(
  repo_id=os.environ.get('HF_DATASET_REPO_ID'),
  repo_type="dataset",
  local_dir="./data",
  token=os.environ.get('HF_TOKEN')
)

print('Defining functions..')
def create_peft_model(model: TriheadSegmentationModel):
  valid_target_modules = []
  for name, module in model.yolo.model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
      if module.groups == 1:
        valid_target_modules.append(name)

  lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=valid_target_modules,
    modules_to_save=["model.23.*"],
    bias="none",
  )
  peft_backbone = get_peft_model(model.yolo.model, peft_config=lora_config).to(device)
  peft_backbone.print_trainable_parameters()
  peft_backbone.to(device)

  model.yolo.model = peft_backbone
  final_model = model.to(device)

  return final_model

def infuse_args(model: TriheadSegmentationModel):
  current_args = model.yolo.model.args if isinstance(model.yolo.model.args, dict) else {}

  if 'overlap_mask' not in current_args:
    current_args['overlap_mask'] = True

  current_args['box'] = 7.5
  current_args['cls'] = 0.5
  current_args['dfl'] = 1.5

  model.yolo.model.args = SimpleNamespace(**current_args)

def get_model(model_type: AblationStudyType):
  if model_type == 'vanilla':
    model = TriheadSegmentationModel(
      yolo_pt_path=f'{MODEL_WEIGHT_DIR}/yolo26m-seg.pt',
      include_depth=False,
      include_normals=False,
      device=device
    )
  elif model_type == 'additional-depth':
    model = TriheadSegmentationModel(
      yolo_pt_path=f'{MODEL_WEIGHT_DIR}/yolo26m-seg.pt',
      include_depth=True,
      include_normals=False,
      device=device
    )
  elif model_type == 'additional-normal':
    model = TriheadSegmentationModel(
      yolo_pt_path=f'{MODEL_WEIGHT_DIR}/yolo26m-seg.pt',
      include_depth=False,
      include_normals=True,
      device=device
    )
  elif model_type == 'additional-both':
    model = TriheadSegmentationModel(
      yolo_pt_path=f'{MODEL_WEIGHT_DIR}/yolo26m-seg.pt',
      include_depth=True,
      include_normals=True,
      device=device
    )

  infuse_args(model)
  final_model = create_peft_model(model)
  loss_balancer = UncertaintyLossWeighting()

  return final_model, loss_balancer

spatial_aug, photometric_aug = get_augmentation_pipeline()

def get_dataset_and_loader(model_type: AblationStudyType):
  if model_type == 'vanilla':
    include_depth = False
    include_normals = False
  elif model_type == 'additional-depth':
    include_depth = True
    include_normals = False
  elif model_type == 'additional-normal':
    include_depth = False
    include_normals = True
  elif model_type == 'additional-both':
    include_depth = True
    include_normals = True
  
  train_dataset = RescueNetDataset(
    data_dir='./data/RescueNet/train',
    include_depth=include_depth,
    spatial_transform=spatial_aug,
    photometric_transform=photometric_aug,
    training=True
  )
  val_dataset = RescueNetDataset(
    data_dir='./data/RescueNet/val',
    include_depth=include_depth,
    include_normals=include_normals
  )

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
  
  return {
    'train': (train_dataset, train_loader),
    'val': (val_dataset, val_loader),
  }

def get_depth_and_normals_inclusion(model_type: AblationStudyType) -> tuple[bool, bool]:
  if model_type == 'vanilla':
    return False, False
  elif model_type == 'additional-depth':
    return True, False
  elif model_type == 'additional-normal':
    return False, True
  elif model_type == 'additional-both':
    return True, True

TOTAL_EPOCHS = 300

print('Start training - ablation study')
for model_type in [
    'vanilla',
    'additional-depth',
    'additional-normal',
    'additional-both'
  ]:
    final_model, loss_balancer = get_model(model_type)
    dataset_and_loader = get_dataset_and_loader(model_type)
    include_depth, include_normals = get_depth_and_normals_inclusion(model_type)

    trainable_params = [p for p in final_model.parameters() if p.requires_grad]
    trainable_params.extend(loss_balancer.parameters())

    optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler('cuda')

    raw_yolo_architecture = final_model.yolo.model.base_model.model
    seg_loss_criterion = E2ELoss(raw_yolo_architecture, loss_fn=v8SegmentationLoss)
    depth_loss_criterion = L1Loss()

    early_stopping = EarlyStoppingAndCheckpointing()

    train_loader = dataset_and_loader['train'][1]
    val_loader = dataset_and_loader['val'][1]

    epoch_history = []
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, TOTAL_EPOCHS + 1, ):
      epoch_train_loss = 0.0 
      epoch_train_seg_loss = 0.0
      epoch_weighted_train_seg_loss = 0.0
      epoch_train_depth_loss = 0.0
      epoch_weighted_train_depth_loss = 0.0
      epoch_train_normal_loss = 0.0
      epoch_weighted_train_normal_loss = 0.0

      final_model.train()
      with autocast(device_type=device_name, dtype=torch.bfloat16):
        for batch_idx, (batch_images, batch_targets) in enumerate(train_loader, 1):
          optimizer.zero_grad()

          segmentation_out, depth_out, normal_out = final_model(batch_images.to(device=device))

          true_segmentation_map = batch_targets[0]
          true_depth_map = batch_targets[1]['depth'].to(device=device) if include_depth else None
          true_surface_normals = batch_targets[2]['normals'].to(device=device) if include_normals else None

          seg_loss, seg_loss_items = seg_loss_criterion(segmentation_out, true_segmentation_map)
          depth_loss = depth_loss_criterion(depth_out, true_depth_map) if include_depth else torch.tensor(0.0, device=device)
          normal_loss = compute_normal_loss(normal_out, true_surface_normals) if include_normals else torch.tensor(0.0, device=device)

          weighted_seg_loss = torch.abs(loss_balancer.alpha).to(device) * seg_loss
          weighted_depth_loss = torch.abs(loss_balancer.beta).to(device) * depth_loss
          weighted_normal_loss = torch.abs(loss_balancer.gamma).to(device) * normal_loss

          loss_total = weighted_seg_loss + weighted_depth_loss + weighted_normal_loss
          loss_total = loss_total.mean()

          scaler.scale(loss_total).backward()
          scaler.step(optimizer)
          scaler.update()

          epoch_train_loss += loss_total.item()
          epoch_train_seg_loss += seg_loss.mean().item()
          epoch_weighted_train_seg_loss = weighted_seg_loss.mean().item()
          epoch_train_depth_loss += depth_loss.mean().item()
          epoch_weighted_train_depth_loss = weighted_depth_loss.mean().item()
          epoch_train_normal_loss += normal_loss.mean().item()
          epoch_weighted_train_normal_loss += weighted_normal_loss.mean().item()

          print(
            f"Epoch [{epoch:03d}/{TOTAL_EPOCHS:03d}] Batch [{batch_idx:04d}/{len(train_loader):04d}] | "
            f"Batch Train Loss: {loss_total.item():.4f} │ "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}", end='\r'
          )

      avg_train_loss = epoch_train_loss / len(train_loader)
      avg_train_seg_loss = epoch_train_seg_loss / len(train_loader)
      avg_weighted_train_seg_loss = epoch_weighted_train_seg_loss / len(train_loader)
      avg_train_depth_loss = epoch_train_depth_loss / len(train_loader)
      avg_weighted_train_depth_loss = epoch_weighted_train_depth_loss / len(train_loader)
      avg_train_normal_loss = epoch_train_normal_loss / len(train_loader)
      avg_weighted_train_normal_loss = epoch_weighted_train_normal_loss / len(train_loader)

      epoch_val_loss = 0.0
      epoch_val_seg_loss = 0.0
      epoch_weighted_val_seg_loss = 0.0
      epoch_val_depth_loss = 0.0
      epoch_weighted_val_depth_loss = 0.0
      epoch_val_normal_loss = 0.0
      epoch_weighted_val_normal_loss = 0.0

      final_model.eval()
      with torch.no_grad():
        for batch_images_val, batch_targets_val in val_loader:
          segmentation_out, depth_out, normal_out = final_model(batch_images_val.to(device=device))

          true_segmentation_map = batch_targets_val[0]
          true_depth_map = batch_targets_val[1]['depth'].to(device=device) if include_depth else None
          true_surface_normals = batch_targets_val[2]['normals'].to(device=device) if include_normals else None

          seg_loss, seg_loss_items = seg_loss_criterion(segmentation_out, true_segmentation_map)
          depth_loss = depth_loss_criterion(depth_out, true_depth_map) if include_depth else torch.tensor(0.0, device=device)
          normal_loss = compute_normal_loss(normal_out, true_surface_normals) if include_normals else torch.tensor(0.0, device=device)

          weighted_seg_loss = torch.abs(loss_balancer.alpha).to(device) * seg_loss
          weighted_depth_loss = torch.abs(loss_balancer.beta).to(device) * depth_loss
          weighted_normal_loss = torch.abs(loss_balancer.gamma).to(device) * normal_loss

          val_loss_total = weighted_seg_loss + weighted_depth_loss + weighted_normal_loss

          epoch_val_loss += val_loss_total.mean().item()
          epoch_val_seg_loss += seg_loss.mean().item()
          epoch_weighted_val_seg_loss = weighted_seg_loss.mean().item()
          epoch_val_depth_loss += depth_loss.mean().item()
          epoch_weighted_val_depth_loss = weighted_depth_loss.mean().item()
          epoch_val_normal_loss += normal_loss.mean().item()
          epoch_weighted_val_normal_loss += weighted_normal_loss.mean().item()

      avg_val_loss = epoch_val_loss / len(val_loader)
      avg_val_seg_loss = epoch_val_seg_loss / len(val_loader)
      avg_weighted_val_seg_loss = epoch_weighted_val_seg_loss / len(val_loader)
      avg_val_depth_loss = epoch_val_depth_loss / len(val_loader)
      avg_weighted_val_depth_loss = epoch_weighted_val_depth_loss / len(val_loader)
      avg_val_normal_loss = epoch_val_normal_loss / len(val_loader)
      avg_weighted_val_normal_loss = epoch_weighted_val_normal_loss / len(val_loader)

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
      halt = early_stopping.record_and_check_if_halt(avg_val_loss, final_model.state_dict(), loss_balancer.state_dict())

      print(
        f"Epoch [{epoch:03d}/{TOTAL_EPOCHS:03d}] Batch [{batch_idx:04d}/{len(train_loader):04d}] | "
        f"LR: {optimizer.param_groups[0]['lr']:.2e}\n",
        f"============================================\n"
        f"---TRAINING---\n"
        f"- Total Loss: {avg_train_loss:.4f}\n"
        f"- Seg: {avg_train_seg_loss:.4f}\n"
        f"- Depth: {avg_train_depth_loss:.4f}\n"
        f"- Norm: {avg_train_normal_loss:.4f}\n"
        f"============================================\n"
        f"---VALIDATION---\n"
        f"- Total Loss: {avg_val_loss:.4f}\n"
        f"- Seg: {avg_val_seg_loss:.4f}\n"
        f"- Depth: {avg_val_depth_loss:.4f}\n"
        f"- Norm: {avg_val_normal_loss:.4f}\n"
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

    upload_folder_to_huggingface(
      'weights',
      'weights'
    )
    upload_folder_to_huggingface(
      'eval_results',
      'eval_results'
    )
    
    # Cleanup
    del final_model, loss_balancer, optimizer, scaler, train_loader, val_loader, dataset_and_loader

    unreachable_object = gc.collect()

    if torch.cuda.is_available():
      alloc_before = torch.cuda.memory_allocated() / (1024 ** 3)
      res_before = torch.cuda.memory_reserved() / (1024 ** 3)

      torch.cuda.empty_cache()
      torch.cuda.ipc_collect()

      alloc_after = torch.cuda.memory_allocated() / (1024 ** 3)
      res_after = torch.cuda.memory_reserved() / (1024 ** 3)

      print(f"VRAM Allocated: {alloc_before:.2f} GB  ->  {alloc_after:.2f} GB")
      print(f"VRAM Reserved:  {res_before:.2f} GB  ->  {res_after:.2f} GB")
      print("✅ PyTorch CUDA cache successfully flushed!")
    else:
      print("⚠️ CUDA not detected. Only system RAM was flushed.")

# terminate_session()

