from huggingface_hub import HfApi, create_repo

REPO = "ObeservabilityTranformerPruning/distilbert-pruning-baselines"

create_repo(REPO, repo_type="model", private=True, exist_ok=True)

HfApi().upload_folder(
    folder_path="results/checkpoints",   # contains imdb/, ag_news/, banking77/
    repo_id=REPO,
    repo_type="model",
    commit_message="IMDB, AG News, BANKING77 baselines (best + rewind_epoch1)",
)

