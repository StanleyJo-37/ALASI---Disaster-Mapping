import os
from huggingface_hub import upload_file
from pathlib import Path
 
 
def upload_folder_to_huggingface(local_folder_path: str, path_in_repo: str = ""):
    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_MODEL_REPO_ID")
 
    for file_path in Path(local_folder_path).rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_folder_path)
            hf_path = f"{path_in_repo}/{relative_path}".replace("\\", "/") if path_in_repo else str(relative_path).replace("\\", "/")
 
            upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=hf_path,
                repo_id=repo_id,
                token=token,
            )
 
upload_folder_to_huggingface(
    local_folder_path="./output/models",  
    path_in_repo="",
)
