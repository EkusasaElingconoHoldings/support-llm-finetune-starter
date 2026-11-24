import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, List

import requests

JUDGE_SYSTEM_PROMPT = """You are an objective QA evaluator for customer support AI.
Score the candidate response from 0.0 to 1.0 on:
1. accuracy: Factual alignment with the reference answer/product facts and proper escalation when needed.
2. brand_alignment: Empathetic, clear, non-defensive tone and clean structure matching brand voice.

Respond ONLY with JSON:
{"accuracy_score": float, "brand_alignment_score": float, "notes": str}
"""


def load_jsonl(path: str) -> List[Dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_evaluation(
    eval_samples: List[Dict],
    generate_fn: Callable[[List[Dict]], str],
    judge_fn: Callable[[str, str], Dict],
    accuracy_threshold: float = 0.80,
    brand_threshold: float = 0.80,
) -> Dict:
    results = []
    latencies_ms = []

    for sample in eval_samples:
        messages = sample["messages"]
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        expected_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]

        t0 = time.perf_counter()
        candidate_msg = generate_fn(prompt_msgs)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        judge_payload = (
            f"Customer Inquiry:\n{user_msg}\n\n"
            f"Reference Response:\n{expected_msg}\n\n"
            f"Candidate AI Response:\n{candidate_msg}"
        )
        verdict = judge_fn(JUDGE_SYSTEM_PROMPT, judge_payload)

        acc_pass = float(verdict.get("accuracy_score", 0.0)) >= accuracy_threshold
        brand_pass = float(verdict.get("brand_alignment_score", 0.0)) >= brand_threshold

        results.append(
            {
                "ticket_id": sample.get("metadata", {}).get("ticket_id"),
                "accuracy_score": verdict.get("accuracy_score", 0.0),
                "brand_alignment_score": verdict.get("brand_alignment_score", 0.0),
                "acc_pass": acc_pass,
                "brand_pass": brand_pass,
                "notes": verdict.get("notes", ""),
            }
        )

    n = max(len(results), 1)
    accuracy_rate = sum(1 for r in results if r["acc_pass"]) / n
    brand_rate = sum(1 for r in results if r["brand_pass"]) / n

    return {
        "total_samples": len(results),
        "product_accuracy_pass_rate": round(accuracy_rate * 100, 2),
        "target_product_accuracy_pct": 90.0,
        "meets_accuracy_target": (accuracy_rate * 100) >= 90.0,
        "brand_alignment_pass_rate": round(brand_rate * 100, 2),
        "target_brand_alignment_pct": 95.0,
        "meets_brand_target": (brand_rate * 100) >= 95.0,
        "avg_latency_ms": round(sum(latencies_ms) / n, 1),
        "details": results,
    }


def make_openai_compatible_generator(base_url: str, model_name: str, api_key: str = "EMPTY") -> Callable[[List[Dict]], str]:
    def _gen(messages: List[Dict]) -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model_name, "messages": messages, "temperature": 0.2, "max_tokens": 400}
        resp = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    return _gen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run customer support LLM evaluation harness")
    parser.add_argument("--eval-file", default="data/processed/holdout_eval.jsonl")
    parser.add_argument("--report-file", default="artifacts/eval_report.json")
    parser.add_argument("--mock", action="store_true", help="Run deterministic local mock verification")
    args = parser.parse_args()

    samples = load_jsonl(args.eval_file)
    if args.mock:
        gen_fn = lambda msgs: "I'd be happy to help with that! Here are the verified steps from our support guide."
        judge_fn = lambda sys_p, usr_p: {"accuracy_score": 0.94, "brand_alignment_score": 0.97, "notes": "Mock pass"}
    else:
        gen_fn = make_openai_compatible_generator(
            os.getenv("LLM_SERVING_URL", "http://localhost:8000/v1"),
            os.getenv("FINE_TUNED_MODEL_ID", "support-llm-finetuned"),
        )
        judge_gen = make_openai_compatible_generator(
            os.getenv("JUDGE_SERVING_URL", "http://localhost:8000/v1"),
            os.getenv("JUDGE_MODEL_ID", "support-llm-finetuned"),
        )
        judge_fn = lambda sys_p, usr_p: json.loads(
            judge_gen([{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}])
        )

    report = run_evaluation(samples, gen_fn, judge_fn)
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
