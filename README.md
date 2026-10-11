# ml-pipeline

GitHub-centred ML pipeline. Every push to `main` runs the full chain:

```
Kaggle (data)  ->  train  ->  Hugging Face (model)  ->  Docker Hub (image)
```

## What runs where

| Stage | Where it happens |
|---|---|
| Dataset download | `kaggle` CLI, authenticated with repo secrets |
| Training | `src/train.py` |
| Model publish | `src/push_hf.py` -> Hugging Face |
| Image build | `Dockerfile` -> Docker Hub |

## Required repository secrets

| Secret | Purpose |
|---|---|
| `KAGGLE_USERNAME` | Kaggle API auth |
| `KAGGLE_KEY` | Kaggle API auth |
| `HF_TOKEN` | Hugging Face write token |
| `DOCKERHUB_USERNAME` | Docker Hub auth |
| `DOCKERHUB_TOKEN` | Docker Hub access token (needs **Read & Write**) |

## Running it

Push to `main`, or trigger manually from the **Actions** tab.

```bash
gh workflow run "ML Pipeline"
gh run watch
```

## Changing the dataset

Edit the slug in `.github/workflows/ml-pipeline.yml`:

```yaml
kaggle datasets download -d OWNER/DATASET -p data/ --unzip
```

`src/train.py` picks up whichever CSV lands in `data/`, so no other change is needed.
