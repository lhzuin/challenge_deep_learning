from huggingface_hub import login, snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# 1️⃣ mirror the HF repo (this will fetch the actual safetensors via LFS)
local_repo = "models/falcon-7b-instruct"
snapshot_download(
    repo_id="tiiuae/falcon-7b-instruct",
    local_dir=local_repo,
    local_dir_use_symlinks=False,
    use_auth_token=True,
)

# 2️⃣ now load from disk instead of the Hub
model = AutoModelForCausalLM.from_pretrained(
    local_repo,
    trust_remote_code=True,
    device_map="auto",
    # quantization_config=...,    # <-- if you want 4-bit/8-bit
)
tokenizer = AutoTokenizer.from_pretrained(
    local_repo,
    trust_remote_code=True,
)

# 3️⃣ (optional) re-save into that folder so you never have to touch HF again
model.save_pretrained(local_repo)
tokenizer.save_pretrained(local_repo)