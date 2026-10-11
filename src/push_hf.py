import os

from huggingface_hub import HfApi

REPO_ID = "Elisha622/ml-pipeline"
FOLDER = "models"


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set. Add it with: gh secret set HF_TOKEN")

    if not os.path.isdir(FOLDER) or not os.listdir(FOLDER):
        raise SystemExit(f"Nothing to upload: {FOLDER}/ is missing or empty.")

    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=FOLDER, repo_id=REPO_ID, repo_type="model")

    print(f"Uploaded {FOLDER}/ to https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
