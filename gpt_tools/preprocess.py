import os, json, hashlib, time, tqdm, torch, hydra, pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import re
bucket_lines = [
    "1:Feature & full-length films",
    "2:Film scenes & clips",
    "3:Short narrative ≤15 min",
    "4:Music & animated videos",
    "5:Promos, trailers & reels",
    "6:Making-of/BTS/tutorials",
    "7:Compilations & montages",
    "8:Interactive & micro ≤60 s",
    "9:Podcasts & talk shows",
    "10:Vlogs & personal journals",
    "11:Gaming & esports",
    "12:Product reviews & unboxings",
    "13:Live streams & premieres",
    "14:Reaction & commentary",
    "15:News & current events",
    "16:Educational explainers",
    "17:ASMR & wellness",
    "18:How-to & DIY",
    "19:Challenges & pranks",
    "20:Other",
]
bucket_text = "\n".join(bucket_lines)

few_shot = """
Example 1:
Channel: Foo  
Title: "My Trailer"  
Desc: "A 30s teaser for the upcoming movie."  
Number: 5

Example 2:
Channel: Bar  
Title: "Epic Feature Film"  
Desc: "A two-hour Bengali drama about ... (2h runtime)"  
Number: 1

Example 3:
Channel: CGMeetup  
Title: "VFX Breakdown: Alien"  
Desc: "Behind-the-scenes of the alien compositing."  
Number: 6
"""

CAT_PROMPT = f"""
You must reply with exactly one number (1–20) and nothing else.
Pick the best category from:
{bucket_text}

{few_shot}

""".strip()

# ────────── SUMMARY prompt (one sentence) ──────────
SUM_PROMPT = """
Write one concise sentence (≤150 characters) that captures the video's topic.
Start with a capital letter, no line breaks, no hashtags, no markdown.
""".strip()

def prompt_cat(row):
    return (f"{CAT_PROMPT}\n Now classify: \n"
            f"Channel: {row['channel']}\n"
            f"Title: {row['title']}\n"
            f"Desc: {row['description'][:300]}\n"
            "Number: ")

def prompt_sum(row):
    return (f"<s>[INST]{SUM_PROMPT}\n"
            f"Channel: {row['channel']}\n"
            f"Title: {row['title']}\n"
            f"Description: {row['description']}\n"
            "Sentence:[/INST]")

def batch_generate(model, tokenizer, prompts, max_new=32):
    inputs = tokenizer(prompts, padding=True,
                       return_tensors="pt").to(model.device)
    with torch.no_grad():
        outs = model.generate(**inputs,
                            max_new_tokens=max_new,
                            do_sample=False,
                            temperature=0.0,
                            top_p=None,
                            top_k=None,)
    # cut the echo
    gens = outs[:, inputs["input_ids"].size(1):]
    return tokenizer.batch_decode(gens, skip_special_tokens=True)


@hydra.main(config_path="../configs", config_name="preprocess", version_base="1.1")
def main(cfg):
    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            device_map="auto",
            torch_dtype=torch.float16,   # or bfloat16
            load_in_4bit=True,           # keep if you want 4-bit
            local_files_only=True,       # ⬅️  **offline**
            trust_remote_code=True       # Qwen needs this
    )
    tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_path,
            trust_remote_code=True,
            local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    csv_in  = f"{cfg.dataset_path}/{cfg.split}.csv"
    csv_out = f"{cfg.dataset_path}/{cfg.split}_gpt.csv"
    df = pd.read_csv(csv_in)
    df["description"] = df["description"].fillna("")

    # ------- 1. add a flag in the hydra yaml if you like ------------
    # you can tune this independently for the two passes
    BATCH = getattr(cfg, "batch_size", 32)

    # ---------- PASS A : CATEGORY ---------------------------------
    cats  = [None]*len(df)
    todo  = []
    os.makedirs(cfg.cache_dir, exist_ok=True)

    for i, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Cats"):
        key = hashlib.md5((row["channel"]+row["title"]+row["description"]).encode()).hexdigest()
        path = os.path.join(cfg.cache_dir, f"{key}.cat")
        if os.path.exists(path):
            cats[i] = int(open(path).read().strip())
            continue
        todo.append((i, row, path))
        if len(todo)==BATCH or i==len(df)-1:
            txts = batch_generate(model, tokenizer,
                                  [prompt_cat(r) for _, r, _ in todo],
                                  max_new=50)
            for (idx, _, pth), txt in zip(todo, txts):
                m = re.search(r"\b([1-9]|1\d|20)\b", txt)
                cid = int(m.group(1)) if m else 20
                cats[idx] = cid
                #open(pth, "w").write(str(cid))
                open(pth, "w").write(str(txt))
                
            todo.clear()

    df["category"] = cats
    df.to_csv(f"{csv_out.rsplit('.',1)[0]}_cat.csv", index=False)

    # ---------- PASS B : SUMMARY ----------------------------------
    sums = [None]*len(df)
    todo = []

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
                                  max_new=40)
            for (idx, _, pth), sent in zip(todo, txts):
                # keep first line, trim length
                sent = sent.splitlines()[0].strip()[:150]
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