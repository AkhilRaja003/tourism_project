import os
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "AkhilRaja/visitwithus-wellness-customer-dataset"

api = HfApi(token=HF_TOKEN)

files_to_upload = [
    "model/random_forest_model.pkl",
    "model/cat_cols.json"
]

for file_path in files_to_upload:
    filename = os.path.basename(file_path)
    api.upload_file(path_or_fileobj=file_path, path_in_repo=filename, repo_id=REPO_ID, repo_type="model")
    print(f"Uploaded {filename} to Hugging Face Hub.")

print("All model artifacts uploaded successfully!")
