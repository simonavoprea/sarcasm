# %%
import time
import torch
from transformers import DistilBertTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DistilBertForSequenceClassification
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# %% [markdown]
# in articol cu magenta

# %%
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# 1) Load dataset
ds = load_dataset("raquiba/Sarcasm_News_Headline")

# 2) Create a validation split if missing
if "validation" not in ds:
    split = ds["train"].train_test_split(test_size=0.1, seed=42)
    ds["train"], ds["validation"] = split["train"], split["test"]

# 3) Model & tokenizer
model_name = "roberta-base"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4) Tokenize
def tokenize(batch):
    return tok(batch["headline"], truncation=True, padding="max_length", max_length=192)

ds = ds.map(tokenize, batched=True)

# 5) Rename label column to 'labels' and keep only necessary cols
for split in ds.keys():
    if "is_sarcastic" in ds[split].column_names:
        ds[split] = ds[split].rename_column("is_sarcastic", "labels")

keep_cols = ["input_ids", "attention_mask", "labels"]
if "token_type_ids" in ds["train"].column_names:
    keep_cols.append("token_type_ids")

for split in ds.keys():
    drop = [c for c in ds[split].column_names if c not in keep_cols]
    ds[split] = ds[split].remove_columns(drop)

# 6) Training args — strategies must match when load_best_model_at_end=True
args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=2,
    eval_strategy="epoch",   # <- correct key
    save_strategy="epoch",         # <- match evaluation strategy
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=50,
)
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": f1, "precision_macro": p, "recall_macro": r}
# 7) Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics,
)


trainer.train()


# %%
import torch
import torch.nn.functional as F

trainer.model.eval()
id2label = {0: "non-sarcastic", 1: "sarcastic"}

texts = [

"Wow, two hours of my life I’ll never get back — 10/10 would recommend!",

"Oscar-worthy performances… if the award is for ‘Best Way to Fall Asleep in 5 Minutes’.",

"I laughed, I cried… mostly because it was painfully bad.",

"Finally, a movie that makes watching paint dry feel exciting.",

"The CGI was so realistic I almost believed those were actual cardboard cutouts.",

"If you love plot holes, bad acting, and awkward pauses, this is your masterpiece.",

"I was on the edge of my seat… trying to find the remote to turn it off.",

"A cinematic tour de force in making the audience question their life choices.",

"The soundtrack really stood out — probably because it was better than everything else.",

"A gripping reminder that trailers can be better than the movie."
]

enc = tok(texts, padding=True, truncation=True, max_length=192, return_tensors="pt")
with torch.no_grad():
    logits = trainer.model(**{k: v.to(trainer.model.device) for k, v in enc.items()}).logits
probs = F.softmax(logits, dim=-1)
pred_ids = probs.argmax(-1).tolist()

for t, pid, p in zip(texts, pred_ids, probs.tolist()):
    print(f"{t}\n  -> pred: {id2label[pid]} | probs: {{non-sarcastic: {p[0]:.3f}, sarcastic: {p[1]:.3f}}}")


# %% [markdown]
# fara sarcasm, oricum la arulare a gasit 8 din 10

# %%
# pip install datasets transformers requests scikit-learn pandas
import os, re, shutil, subprocess, requests
from pathlib import Path
import pandas as pd
import numpy as np

# --- 0) Config & URLs ---
BASE_URL = "https://raw.githubusercontent.com/ef2020/SarcasmAmazonReviewsCorpus/master/"
FILES = ["Ironic.rar", "Regular.rar", "file_pairing.txt"]  # (sarcasm_lines.txt optional)
DATA_DIR = Path("sarcasm_amazon_data")
EXTRACT_DIR = DATA_DIR / "extracted"
IRONIC_DIR = EXTRACT_DIR / "Ironic"
REGULAR_DIR = EXTRACT_DIR / "Regular"
DATA_DIR.mkdir(parents=True, exist_ok=True); EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

# --- 1) Download archives & pairing file ---
def download(url: str, out_path: Path):
    if out_path.exists(): return
    r = requests.get(url, timeout=120); r.raise_for_status()
    out_path.write_bytes(r.content)

for fname in FILES:
    download(BASE_URL + fname, DATA_DIR / fname)

# --- 2) Extract .rar archives via 7-Zip (robust) ---
def _find_7z() -> str:
    p = shutil.which("7z")
    if p: return p
    for c in [r"C:\Program Files\7-Zip\7z.exe",
              r"C:\Program Files (x86)\7-Zip\7z.exe",
              "/usr/bin/7z", "/usr/local/bin/7z", "/opt/homebrew/bin/7z"]:
        if Path(c).exists(): return c
    p = os.environ.get("SEVENZ_PATH")
    if p and Path(p).exists(): return p
    raise FileNotFoundError(
        "7z not found. Install 7-Zip and add to PATH, or set SEVENZ_PATH.\n"
        "Windows: winget install 7zip.7zip"
    )

def extract_with_7z(rar_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # skip if there are already .txt files somewhere under out_dir
    if list(out_dir.rglob("*.txt")): return
    sevenz = _find_7z()
    cmd = [sevenz, "x", str(rar_path), f"-o{out_dir}", "-y"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"7z failed for {rar_path}\nSTDERR:\n{res.stderr[:400]}")

extract_with_7z(DATA_DIR / "Ironic.rar", IRONIC_DIR)
extract_with_7z(DATA_DIR / "Regular.rar", REGULAR_DIR)

# --- 3) Read reviews (recursive) ---
def read_txt_dir_recursive(d: Path):
    out = {}
    for fp in d.rglob("*.txt"):
        try:
            txt = fp.read_text(encoding="utf-8", errors="ignore").strip()
        except UnicodeDecodeError:
            txt = fp.read_text(errors="ignore").strip()
        out[fp.name] = txt
    return out

ironic_raw  = read_txt_dir_recursive(IRONIC_DIR)
regular_raw = read_txt_dir_recursive(REGULAR_DIR)
print(f"Ironic files found: {len(ironic_raw)} | Regular files found: {len(regular_raw)}")

# Build normalized lookup: key = lowercase basename without extension
def norm_key(name: str) -> str:
    base = Path(name).name
    return re.sub(r"\.txt$", "", base, flags=re.I).lower()

ironic = {norm_key(k): v for k, v in ironic_raw.items()}
regular = {norm_key(k): v for k, v in regular_raw.items()}

# --- 4) Parse file_pairing.txt using normalized names ---
pairing_txt = (DATA_DIR / "file_pairing.txt").read_text(encoding="utf-8", errors="ignore")
pairs, singles = [], []
for line in pairing_txt.splitlines():
    line = line.strip()
    if not line: continue
    if line.upper().startswith("PAIRS:"):
        fns = re.findall(r"([A-Za-z0-9_\-]+\.txt)", line)
        if len(fns) >= 2:
            pairs.append((norm_key(fns[0]), norm_key(fns[1])))
    elif line.upper().startswith("IRONIC:"):
        fns = re.findall(r"([A-Za-z0-9_\-]+\.txt)", line)
        if fns:
            singles.append((norm_key(fns[0]), 1))
    elif line.upper().startswith("REGULAR:"):
        fns = re.findall(r"([A-Za-z0-9_\-]+\.txt)", line)
        if fns:
            singles.append((norm_key(fns[0]), 0))

# --- 5) Build rows; track mismatches for debugging ---
rows = []
missing_pairs, missing_singles = 0, 0
group_id = 0

for fi, fr in pairs:
    ti, tr = ironic.get(fi), regular.get(fr)
    if ti is not None and tr is not None:
        rows.append({"fname": fi, "text": ti, "label": 1, "group": f"pair_{group_id}"})
        rows.append({"fname": fr, "text": tr, "label": 0, "group": f"pair_{group_id}"})
        group_id += 1
    else:
        missing_pairs += 1

for fn, lab in singles:
    if lab == 1:
        t = ironic.get(fn)
        if t is not None:
            rows.append({"fname": fn, "text": t, "label": 1, "group": f"single_{fn}"})
        else:
            missing_singles += 1
    else:
        t = regular.get(fn)
        if t is not None:
            rows.append({"fname": fn, "text": t, "label": 0, "group": f"single_{fn}"})
        else:
            missing_singles += 1

print(f"Built {len(rows)} examples from pairing.txt | missing_pairs={missing_pairs}, missing_singles={missing_singles}")

# Fallback: if pairing produced nothing (or very few), just take all files
if len(rows) < 50:
    print("Pairing produced few/zero matches. Falling back to 'all files in Ironic/Regular'.")
    rows = []
    for k, v in ironic.items():
        rows.append({"fname": k, "text": v, "label": 1, "group": f"ir_{k}"})
    for k, v in regular.items():
        rows.append({"fname": k, "text": v, "label": 0, "group": f"reg_{k}"})
    print(f"Total rows after fallback: {len(rows)}")

# --- 6) DataFrame & splits ---
df = pd.DataFrame(rows)
if "text" not in df.columns or df.empty:
    raise RuntimeError(
        "No review texts assembled. Check that extraction created .txt files and that file_pairing names "
        "match extracted filenames. Print a few keys:\n"
        f"  Ironic keys sample: {list(list(ironic.keys())[:5])}\n"
        f"  Regular keys sample: {list(list(regular.keys())[:5])}"
    )

print(df.head(), "\nCounts:\n", df["label"].value_counts())

from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(test_size=0.1, n_splits=1, random_state=42)
train_idx, val_idx = next(gss.split(df, groups=df["group"]))
df_train = df.iloc[train_idx].reset_index(drop=True)
df_val   = df.iloc[val_idx].reset_index(drop=True)

# Optional test split from train
gss2 = GroupShuffleSplit(test_size=0.1, n_splits=1, random_state=123)
tr_idx, te_idx = next(gss2.split(df_train, groups=df_train["group"]))
df_test  = df_train.iloc[te_idx].reset_index(drop=True)
df_train = df_train.iloc[tr_idx].reset_index(drop=True)
print("Split sizes:", len(df_train), len(df_val), len(df_test))

# --- 7) HF DatasetDict ---
from datasets import Dataset, DatasetDict
def to_hfds(dframe: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(dframe[["text", "label"]], preserve_index=False)

hf_ds = DatasetDict({
    "train": to_hfds(df_train),
    "validation": to_hfds(df_val),
    "test": to_hfds(df_test)
})

# --- 8) Tokenize & fine-tune RoBERTa ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
#model_name = "roberta-base"
#model_name = "microsoft/deberta-v3-base"
#model_name = "microsoft/deberta-v3-large"
#model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model_name = "roberta-large"

tok = AutoTokenizer.from_pretrained(model_name)

id2label = {0: "non-sarcastic", 1: "sarcastic"}
label2id = {"non-sarcastic": 0, "sarcastic": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, id2label=id2label, label2id=label2id
)

def tokenize(batch):
    return tok(batch["text"], truncation=True, padding="max_length", max_length=192)

hf_ds = hf_ds.map(tokenize, batched=True)

cols = ["input_ids", "attention_mask", "label"]
if "token_type_ids" in hf_ds["train"].column_names:
    cols.append("token_type_ids")
for split in hf_ds.keys():
    drops = [c for c in hf_ds[split].column_names if c not in cols]
    hf_ds[split] = hf_ds[split].remove_columns(drops)
    hf_ds[split] = hf_ds[split].rename_column("label", "labels")

from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": f1, "precision_macro": p, "recall_macro": r}

args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=hf_ds["train"],
    eval_dataset=hf_ds["validation"],
    tokenizer=tok,
    compute_metrics=compute_metrics,
)


trainer.train()
print(trainer.evaluate(hf_ds["test"]))


# %% [markdown]
# cu FILES = ["Ironic.rar", "Regular.rar", "file_pairing.txt", "sarcasm_lines.txt"]

# %%
# pip install datasets transformers requests scikit-learn pandas
import os, re, shutil, subprocess, requests, random
from pathlib import Path
import pandas as pd
import numpy as np

# --- 0) Config & URLs ---
BASE_URL = "https://raw.githubusercontent.com/ef2020/SarcasmAmazonReviewsCorpus/master/"
FILES = ["Ironic.rar", "Regular.rar", "file_pairing.txt", "sarcasm_lines.txt"]
DATA_DIR = Path("sarcasm_amazon_data")
EXTRACT_DIR = DATA_DIR / "extracted"
IRONIC_DIR = EXTRACT_DIR / "Ironic"
REGULAR_DIR = EXTRACT_DIR / "Regular"
DATA_DIR.mkdir(parents=True, exist_ok=True); EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

# --- 1) Download archives & pairing file ---
def download(url: str, out_path: Path):
    if out_path.exists(): return
    r = requests.get(url, timeout=120); r.raise_for_status()
    out_path.write_bytes(r.content)

for fname in FILES:
    download(BASE_URL + fname, DATA_DIR / fname)

# --- 2) Extract .rar archives via 7-Zip (robust) ---
def _find_7z() -> str:
    p = shutil.which("7z")
    if p: return p
    for c in [r"C:\Program Files\7-Zip\7z.exe",
              r"C:\Program Files (x86)\7-Zip\7z.exe",
              "/usr/bin/7z", "/usr/local/bin/7z", "/opt/homebrew/bin/7z"]:
        if Path(c).exists(): return c
    p = os.environ.get("SEVENZ_PATH")
    if p and Path(p).exists(): return p
    raise FileNotFoundError(
        "7z not found. Install 7-Zip and add to PATH, or set SEVENZ_PATH.\n"
        "Windows: winget install 7zip.7zip"
    )

def extract_with_7z(rar_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if list(out_dir.rglob("*.txt")): return
    sevenz = _find_7z()
    cmd = [sevenz, "x", str(rar_path), f"-o{out_dir}", "-y"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"7z failed for {rar_path}\nSTDERR:\n{res.stderr[:400]}")

extract_with_7z(DATA_DIR / "Ironic.rar", IRONIC_DIR)
extract_with_7z(DATA_DIR / "Regular.rar", REGULAR_DIR)

# --- 3) Read reviews (recursive) ---
def read_txt_dir_recursive(d: Path):
    out = {}
    for fp in d.rglob("*.txt"):
        try:
            txt = fp.read_text(encoding="utf-8", errors="ignore").strip()
        except UnicodeDecodeError:
            txt = fp.read_text(errors="ignore").strip()
        out[fp.name] = txt
    return out

ironic_raw  = read_txt_dir_recursive(IRONIC_DIR)
regular_raw = read_txt_dir_recursive(REGULAR_DIR)
print(f"Ironic files found: {len(ironic_raw)} | Regular files found: {len(regular_raw)}")

# Build normalized lookup: key = lowercase basename without extension
def norm_key(name: str) -> str:
    base = Path(name).name
    return re.sub(r"\.txt$", "", base, flags=re.I).lower()

# Simple text normalizer for deduplication
def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

ironic = {norm_key(k): v for k, v in ironic_raw.items()}
regular = {norm_key(k): v for k, v in regular_raw.items()}

# --- 4) Parse file_pairing.txt using normalized names ---
pairing_txt = (DATA_DIR / "file_pairing.txt").read_text(encoding="utf-8", errors="ignore")
pairs, singles = [], []
for line in pairing_txt.splitlines():
    line = line.strip()
    if not line: continue
    if line.upper().startswith("PAIRS:"):
        fns = re.findall(r"([A-Za-z0-9_\-]+\.txt)", line)
        if len(fns) >= 2:
            pairs.append((norm_key(fns[0]), norm_key(fns[1])))
    elif line.upper().startswith("IRONIC:"):
        fns = re.findall(r"([A-Za-z0-9_\-]+\.txt)", line)
        if fns:
            singles.append((norm_key(fns[0]), 1))
    elif line.upper().startswith("REGULAR:"):
        fns = re.findall(r"([A-Za-z0-9_\-]+\.txt)", line)
        if fns:
            singles.append((norm_key(fns[0]), 0))

# --- 5) Build rows; track mismatches for debugging ---
rows = []
missing_pairs, missing_singles = 0, 0
group_id = 0

for fi, fr in pairs:
    ti, tr = ironic.get(fi), regular.get(fr)
    if ti is not None and tr is not None:
        rows.append({"fname": fi, "text": ti, "label": 1, "group": f"pair_{group_id}"})
        rows.append({"fname": fr, "text": tr, "label": 0, "group": f"pair_{group_id}"})
        group_id += 1
    else:
        missing_pairs += 1

for fn, lab in singles:
    if lab == 1:
        t = ironic.get(fn)
        if t is not None:
            rows.append({"fname": fn, "text": t, "label": 1, "group": f"single_{fn}"})
        else:
            missing_singles += 1
    else:
        t = regular.get(fn)
        if t is not None:
            rows.append({"fname": fn, "text": t, "label": 0, "group": f"single_{fn}"})
        else:
            missing_singles += 1

print(f"Built {len(rows)} examples from pairing.txt | missing_pairs={missing_pairs}, missing_singles={missing_singles}")

# Fallback: if pairing produced nothing (or very few), just take all files
if len(rows) < 50:
    print("Pairing produced few/zero matches. Falling back to 'all files in Ironic/Regular'.")
    rows = []
    for k, v in ironic.items():
        rows.append({"fname": k, "text": v, "label": 1, "group": f"ir_{k}"})
    for k, v in regular.items():
        rows.append({"fname": k, "text": v, "label": 0, "group": f"reg_{k}"})
    print(f"Total rows after fallback: {len(rows)}")

# --- 5b) ADD sarcasm_lines.txt (label=1), dedupe, then shuffle ---
sarcasm_path = DATA_DIR / "sarcasm_lines.txt"
added = 0
if sarcasm_path.exists():
    s_txt = sarcasm_path.read_text(encoding="utf-8", errors="ignore")
    s_lines = [ln.strip() for ln in s_txt.splitlines() if ln.strip()]
    # Build a set of existing normalized texts to avoid duplicates
    existing_norms = set(norm_text(r["text"]) for r in rows)
    for i, line in enumerate(s_lines):
        nline = norm_text(line)
        if nline and nline not in existing_norms:
            rows.append({
                "fname": f"sarcasm_line_{i:06d}.txt",
                "text": line,
                "label": 1,
                "group": f"sline_{i:06d}"     # unique group per line to avoid leakage
            })
            existing_norms.add(nline)
            added += 1
    print(f"Appended {added} sarcastic lines from sarcasm_lines.txt (deduped).")
else:
    print("sarcasm_lines.txt not found; skipping.")

# Shuffle all rows before making the DataFrame (for good measure; split is group-aware anyway)
random.seed(42)
random.shuffle(rows)

# --- 6) DataFrame & splits ---
df = pd.DataFrame(rows)
if "text" not in df.columns or df.empty:
    raise RuntimeError(
        "No review texts assembled. Check that extraction created .txt files and that file_pairing names "
        "match extracted filenames."
    )

print(df.head(), "\nCounts:\n", df["label"].value_counts())

from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(test_size=0.1, n_splits=1, random_state=42)
train_idx, val_idx = next(gss.split(df, groups=df["group"]))
df_train = df.iloc[train_idx].reset_index(drop=True)
df_val   = df.iloc[val_idx].reset_index(drop=True)

# Optional test split from train
gss2 = GroupShuffleSplit(test_size=0.1, n_splits=1, random_state=123)
tr_idx, te_idx = next(gss2.split(df_train, groups=df_train["group"]))
df_test  = df_train.iloc[te_idx].reset_index(drop=True)
df_train = df_train.iloc[tr_idx].reset_index(drop=True)
print("Split sizes:", len(df_train), len(df_val), len(df_test))

# --- 7) HF DatasetDict ---
from datasets import Dataset, DatasetDict
def to_hfds(dframe: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(dframe[["text", "label"]], preserve_index=False)

hf_ds = DatasetDict({
    "train": to_hfds(df_train),
    "validation": to_hfds(df_val),
    "test": to_hfds(df_test)
})

# --- 8) Tokenize & fine-tune RoBERTa ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
#model_name = "roberta-base"
#model_name = "microsoft/deberta-v3-base"
#model_name = "microsoft/deberta-v3-large"
#model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model_name = "distilbert-base-uncased"
#model_name = "roberta-large"

tok = AutoTokenizer.from_pretrained(model_name)

id2label = {0: "non-sarcastic", 1: "sarcastic"}
label2id = {"non-sarcastic": 0, "sarcastic": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, id2label=id2label, label2id=label2id
)

def tokenize(batch):
    return tok(batch["text"], truncation=True, padding="max_length", max_length=192)

hf_ds = hf_ds.map(tokenize, batched=True)

cols = ["input_ids", "attention_mask", "label"]
if "token_type_ids" in hf_ds["train"].column_names:
    cols.append("token_type_ids")
for split in hf_ds.keys():
    drops = [c for c in hf_ds[split].column_names if c not in cols]
    hf_ds[split] = hf_ds[split].remove_columns(drops)
    hf_ds[split] = hf_ds[split].rename_column("label", "labels")

from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": f1, "precision_macro": p, "recall_macro": r}

args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=50,
)
"""from transformers import TrainingArguments, EarlyStoppingCallback
callbacks = [EarlyStoppingCallback(
    early_stopping_patience=1,
    early_stopping_threshold=1e-3  # ignoră “îmbunătățiri” minuscule ca +0.00009
)]"""

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=hf_ds["train"],
    eval_dataset=hf_ds["validation"],
    tokenizer=tok,
    compute_metrics=compute_metrics,
)


trainer.train()
print(trainer.evaluate(hf_ds["test"]))


# %%
import torch
import torch.nn.functional as F

trainer.model.eval()
id2label = {0: "non-sarcastic", 1: "sarcastic"}

texts = [

"Wow, two hours of my life I’ll never get back — 10/10 would recommend!",

"Oscar-worthy performances… if the award is for ‘Best Way to Fall Asleep in 5 Minutes’.",

"I laughed, I cried… mostly because it was painfully bad.",

"Finally, a movie that makes watching paint dry feel exciting.",

"The CGI was so realistic I almost believed those were actual cardboard cutouts.",

"If you love plot holes, bad acting, and awkward pauses, this is your masterpiece.",

"I was on the edge of my seat… trying to find the remote to turn it off.",

"A cinematic tour de force in making the audience question their life choices.",

"The soundtrack really stood out — probably because it was better than everything else.",

"A gripping reminder that trailers can be better than the movie."
]

enc = tok(texts, padding=True, truncation=True, max_length=192, return_tensors="pt")
with torch.no_grad():
    logits = trainer.model(**{k: v.to(trainer.model.device) for k, v in enc.items()}).logits
probs = F.softmax(logits, dim=-1)
pred_ids = probs.argmax(-1).tolist()

for t, pid, p in zip(texts, pred_ids, probs.tolist()):
    print(f"{t}\n  -> pred: {id2label[pid]} | probs: {{non-sarcastic: {p[0]:.3f}, sarcastic: {p[1]:.3f}}}")


# %%
import torch
import torch.nn.functional as F

trainer.model.eval()
id2label = {0: "non-sarcastic", 1: "sarcastic"}

texts = [
"Five stars for teaching me patience—only took three hours to load the homepage.",
"Battery life is incredible—dies just from looking at 20%.",
"The ‘noise-cancelling’ headphones work great at cancelling music, not the noise.",
"Our room had a ‘city view’—if you count a brick wall as urban scenery.",
"Customer support was lightning-fast; I blinked and only waited two weeks.",
"The camera’s low-light performance is amazing—you can’t see a thing.",
"Truly intuitive UI; I only needed a PhD and a treasure map.",
"Love the premium build—feels expensive to replace.",
"Great value: you get two features for the price of five.",
"The chef really nailed it—my steak had the same personality as the shoe it resembled."
]

enc = tok(texts, padding=True, truncation=True, max_length=192, return_tensors="pt")
with torch.no_grad():
    logits = trainer.model(**{k: v.to(trainer.model.device) for k, v in enc.items()}).logits
probs = F.softmax(logits, dim=-1)
pred_ids = probs.argmax(-1).tolist()

for t, pid, p in zip(texts, pred_ids, probs.tolist()):
    print(f"{t}\n  -> pred: {id2label[pid]} | probs: {{non-sarcastic: {p[0]:.3f}, sarcastic: {p[1]:.3f}}}")


# %% [markdown]
# "Perfect for minimalists—comes without the features I actually needed.",
# "Seamless connectivity—disconnects right before you hit connect.",
# "Smells like premium—probably the plastic melting.",
# "Auto-save works great—saves everything except my progress.",
# "Compact design—especially the battery life.",
# "Truly hands-free—because it never responds to touch.",
# "Ergonomic keyboard—if your hands are shaped like question marks.",
# "Water-resistant—tears only, not rain.",
# "Delivers overnight—just not the same night as the order.",
# "Top-tier security—no one, including me, can log in.",
# "Voice assistant understands me—when I say nothing.",
# "The update improved everything—by making sure nothing opens.",
# "Room with natural light—after the sun bounces off the parking lot.",
# "Noise isolation is superb—I couldn’t hear the music at all.",
# "Lag-free gaming—because the game never starts.",
# "Travel pillow so comfy—I stayed awake the entire flight.",
# "Self-cleaning oven—burnt everything to a crisp, problem solved.",
# "Eco mode is powerful—saves energy by not working.",
# "Generous portions—of disappointment.",
# "Shockproof case—phone fainted anyway.",
# "Works out of the box—once you buy the missing parts.",
# "Crystal-clear display—if you enjoy fog.",
# "Fast charging—percentages move at the speed of drama.",
# "Kid-friendly app—my inner child cried.",
# "Five-star dining—the bill, not the food.",
# "The map is so accurate—it led me straight to nowhere.",
# "Bluetooth range is impressive—if you stand exactly next to it.",
# "Build quality feels solid—like a brick with buttons.",
# "The tutorial is intuitive—after the third reread.",
# "Sleek interface—hides the settings you actually need."
# 

# %%
import torch
import torch.nn.functional as F

trainer.model.eval()
id2label = {0: "non-sarcastic", 1: "sarcastic"}

texts = [
"Perfect for minimalists—comes without the features I actually needed.",
"Seamless connectivity—disconnects right before you hit connect.",
"Smells like premium—probably the plastic melting.",
"Auto-save works great—saves everything except my progress.",
"Compact design—especially the battery life.",
"Truly hands-free—because it never responds to touch.",
"Ergonomic keyboard—if your hands are shaped like question marks.",
"Water-resistant—tears only, not rain.",
"Delivers overnight—just not the same night as the order.",
"Top-tier security—no one, including me, can log in.",
"Voice assistant understands me—when I say nothing.",
"The update improved everything—by making sure nothing opens.",
"Room with natural light—after the sun bounces off the parking lot.",
"Noise isolation is superb—I couldn’t hear the music at all.",
"Lag-free gaming—because the game never starts.",
"Travel pillow so comfy—I stayed awake the entire flight.",
"Self-cleaning oven—burnt everything to a crisp, problem solved.",
"Eco mode is powerful—saves energy by not working.",
"Generous portions—of disappointment.",
"Shockproof case—phone fainted anyway.",
"Works out of the box—once you buy the missing parts.",
"Crystal-clear display—if you enjoy fog.",
"Fast charging—percentages move at the speed of drama.",
"Kid-friendly app—my inner child cried.",
"Five-star dining—the bill, not the food.",
"The map is so accurate—it led me straight to nowhere.",
"Bluetooth range is impressive—if you stand exactly next to it.",
"Build quality feels solid—like a brick with buttons.",
"The tutorial is intuitive—after the third reread.",
"Sleek interface—hides the settings you actually need."
]

enc = tok(texts, padding=True, truncation=True, max_length=192, return_tensors="pt")
with torch.no_grad():
    logits = trainer.model(**{k: v.to(trainer.model.device) for k, v in enc.items()}).logits
probs = F.softmax(logits, dim=-1)
pred_ids = probs.argmax(-1).tolist()

for t, pid, p in zip(texts, pred_ids, probs.tolist()):
    print(f"{t}\n  -> pred: {id2label[pid]} | probs: {{non-sarcastic: {p[0]:.3f}, sarcastic: {p[1]:.3f}}}")


# %%
import pandas as pd

pd.set_option('display.max_colwidth', None)  # no truncation
display(df_train[['text']])                   # or: print(df_test[['text']])


# %%
df_train


