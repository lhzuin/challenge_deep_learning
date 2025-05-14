import os, hashlib, tqdm, torch, hydra, pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers import BitsAndBytesConfig



def prompt_sum(row):
    return (
        "Summarize the content of this YouTube video in one concise sentence "
        "(max 150 characters). "
        f"Here's the description:\n\n{row['description']}\n\n"
        "Summary:"
    )

def batch_generate(model, tokenizer, prompts, args: dict):
    inputs = tokenizer(prompts, max_length=1024, padding="max_length", truncation=True, 
                       return_tensors="pt").to(model.device)
    with torch.no_grad():
        outs = model.generate(**inputs,**args)

    # cut the echo
    gens = outs[:, inputs["input_ids"].size(1):]
    del inputs, outs
    torch.cuda.empty_cache()
    return tokenizer.batch_decode(gens, skip_special_tokens=True)


@hydra.main(config_path="../configs", config_name="preprocess", version_base="1.1")
def main(cfg):
    print("Loading model...", flush=True)

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            device_map="auto",
            torch_dtype=torch.float16,   # or bfloat16
            quantization_config=bnb_config,
            local_files_only=True,       # ⬅️  **offline**
            trust_remote_code=True       # Qwen needs this
    )
    tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_path,
            trust_remote_code=True,
            local_files_only=True
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    csv_in  = f"{cfg.dataset_path}/{cfg.split}.csv"
    csv_out = f"{cfg.dataset_path}/{cfg.split}_gpt.csv"
    df = pd.read_csv(csv_in)
    df["description"] = df["description"].fillna("")

    # ------- 1. add a flag in the hydra yaml if you like ------------
    # you can tune this independently for the two passes
    BATCH = getattr(cfg, "batch_size", 64)


    # ––––– Summarization (sampling for variety) –––––
    sum_kwargs = dict(
        max_new_tokens     = 50,       # allow up to ~40 tokens
        min_new_tokens     = 5,
        do_sample          = True,
        temperature        = 0.7,      # more creativity for summaries
        top_p              = 0.9,      # nucleus: top 90% of probability mass
        top_k              = 50,       # sample from top-50 tokens
        repetition_penalty = 1.1,      # avoid verbatim copying
        pad_token_id       = tokenizer.eos_token_id,
        #use_cache=False,
    )

    # ---------- PASS B : SUMMARY ----------------------------------
    sums = [None]*len(df)
    todo = []
    os.makedirs(cfg.cache_dir, exist_ok=True)
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Sums"):
        key = hashlib.md5((row["channel"]+row["title"]+row["description"]).encode()).hexdigest()
        path = os.path.join(cfg.cache_dir, f"{key}.sum")
        if os.path.exists(path):
            sums[i] = open(path).read().strip()
            continue
        todo.append((i, row, path))
        if len(todo)==BATCH or i==len(df)-1:
            txts = batch_generate(model, tokenizer,
                                  [prompt_sum(r) for _, r, _ in todo],
                                  args=sum_kwargs)
            for (idx, _, pth), sent in zip(todo, txts):
                # keep first line, trim length
                lines = sent.splitlines()
                summary = next((line.strip() for line in lines if line.strip()), "")
                sent = summary[:300]
                # capitalise fallback
                if sent and sent[0].islower():
                    sent = sent[0].upper()+sent[1:]
                sums[idx] = sent
                open(pth, "w").write(sent)
            todo.clear()


    df["summary"] = sums
    df.drop(columns=["description"], inplace=True)
    df.to_csv(csv_out, index=False)
    print("Wrote", csv_out)

if __name__ == "__main__":
    main()