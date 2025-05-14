from huggingface_hub import snapshot_download

# 1️⃣ download weights + configs into one folder
local_path = snapshot_download(
    repo_id="tiiuae/falcon-7b-instruct",
    revision="main",
    local_dir="./models/falcon-7b-instruct",
    local_dir_use_symlinks=True,            # saves disk by symlinking cache
    allow_patterns=["*.safetensors","*.json","*.txt", "*.py"],
)

