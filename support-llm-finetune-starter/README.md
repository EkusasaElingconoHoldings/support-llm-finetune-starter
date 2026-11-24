# Customer Support Open-Source LLM Fine-Tuning & Firebase Sidecar Pipeline

End-to-end reference repository demonstrating **Dataset Preparation**, **QLoRA Fine-Tuning (PyTorch + Hugging Face Transformers + PEFT + TRL)**, **Automated Evaluation Harnesses (90% Product Accuracy / 95% Brand Alignment)**, and **Zero-Disruption Firebase Sidecar Serving**.

## Repository Structure

```text
support-llm-finetune-starter/
├── data/
│   └── raw/
│       └── sample_transcripts.csv      # Synthetic 5-year style support dataset for immediate smoke testing
├── src/
│   ├── prepare_dataset.py              # PII redaction, temporal/CSAT filtering, stratified ChatML JSONL split
│   ├── train_qlora.py                  # 4-bit NF4 QLoRA training + adapter merge for vLLM/TGI serving
│   ├── eval_harness.py                 # Automated accuracy (90%) & brand alignment (95%) evaluation suite
│   └── serving_client.py               # OpenAI-compatible inference client with escalation tag detection
├── firebase/
│   └── main.py                         # Non-blocking Firestore sidecar trigger (zero legacy workflow changes)
├── tests/
│   └── test_pipeline.py                # Unit & smoke tests for preprocessing, eval harness, and client
├── Dockerfile.serving                  # Containerized vLLM OpenAI-compatible model server
├── requirements.txt
└── .github/workflows/ci.yml            # GitHub Actions CI verifying data prep & eval harness on push
```

## Quickstart (Local Smoke Test)

```bash
pip install -r requirements.txt

# 1. Clean transcripts, scrub PII, and generate train/holdout JSONL splits
python src/prepare_dataset.py --input data/raw/sample_transcripts.csv --output-dir data/processed --min-date 2024-01-01

# 2. Run automated evaluation harness in mock/dry-run mode (or point at your vLLM endpoint)
python src/eval_harness.py --eval-file data/processed/holdout_eval.jsonl --report-file artifacts/eval_report.json --mock

# 3. Run QLoRA fine-tuning on GPU and merge weights for serving
python src/train_qlora.py --model-name Qwen/Qwen2.5-7B-Instruct --train-file data/processed/train.jsonl --eval-file data/processed/holdout_eval.jsonl --merge-and-save
```
