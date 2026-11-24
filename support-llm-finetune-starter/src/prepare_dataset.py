import argparse
import json
import re
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"),
    "ORDER_OR_ID": re.compile(r"\b(?:order|ticket|inv|acct|ord)[-_:#]?\s*[A-Z0-9]{5,}\b", re.IGNORECASE),
}

NOISE_LINES = re.compile(
    r"^\s*(please hold|one moment while i check|transferring you to|can you hear me)\b.*",
    re.IGNORECASE,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a customer support assistant aligned with our brand voice. "
    "Acknowledge the customer's concern empathetically, provide accurate product guidance, "
    "and insert [AGENT_NOTE: ...] markers whenever human review or account action is required."
)


def scrub_pii(text: str) -> str:
    cleaned = text or ""
    for label, pattern in PII_PATTERNS.items():
        cleaned = pattern.sub(f"[{label}_REDACTED]", cleaned)
    lines = [ln.strip() for ln in cleaned.splitlines() if not NOISE_LINES.match(ln.strip())]
    return "\n".join(lines).strip()


def format_chatml(row: Dict) -> Dict:
    return {
        "messages": [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["user_message"]},
            {"role": "assistant", "content": row["assistant_response"]},
        ],
        "metadata": {
            "ticket_id": row.get("ticket_id"),
            "topic": row.get("topic", "general"),
            "csat": row.get("csat"),
        },
    }


def prepare_dataset(
    input_path: str,
    output_dir: str,
    min_date: str = "2024-01-01",
    min_csat: float = 4.0,
    test_size: float = 0.2,
    seed: int = 42,
) -> Dict[str, int]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_json(input_path, lines=True)

    if "created_at" in df.columns and min_date:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df = df[df["created_at"] >= pd.to_datetime(min_date)]

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().isin(["resolved", "closed", "solved"])]

    if "csat" in df.columns and min_csat is not None:
        df = df[pd.to_numeric(df["csat"], errors="coerce").fillna(min_csat) >= min_csat]

    df = df.copy()
    df["user_message"] = df["customer_inquiry"].map(scrub_pii)
    df["assistant_response"] = df["agent_reply"].map(scrub_pii)
    df["system_prompt"] = DEFAULT_SYSTEM_PROMPT

    df = df[(df["user_message"].str.len() > 10) & (df["assistant_response"].str.len() > 15)].drop_duplicates(
        subset=["user_message", "assistant_response"]
    )

    can_stratify = (
        "topic" in df.columns
        and df["topic"].nunique() > 1
        and df["topic"].value_counts().min() >= 2
        and int(len(df) * test_size) >= df["topic"].nunique()
    )
    stratify_col = df["topic"] if can_stratify else None
    train_df, holdout_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify_col)

    counts = {}
    for split_name, split_df in [("train", train_df), ("holdout_eval", holdout_df)]:
        records = [format_chatml(r) for r in split_df.to_dict(orient="records")]
        target_file = out / f"{split_name}.jsonl"
        with target_file.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[split_name] = len(records)
        print(f"Wrote {len(records)} records -> {target_file}")

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare customer support transcripts for QLoRA fine-tuning")
    parser.add_argument("--input", default="data/raw/sample_transcripts.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--min-date", default="2024-01-01")
    parser.add_argument("--min-csat", type=float, default=4.0)
    args = parser.parse_args()
    prepare_dataset(args.input, args.output_dir, min_date=args.min_date, min_csat=args.min_csat)
