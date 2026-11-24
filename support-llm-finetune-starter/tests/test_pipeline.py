import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from eval_harness import load_jsonl, run_evaluation
from prepare_dataset import prepare_dataset, scrub_pii
from serving_client import FineTunedSupportClient


def test_scrub_pii_and_noise():
    raw = "One moment while I check\nContact me at jane.doe@company.com or 555-123-4567 for ORD-881234."
    cleaned = scrub_pii(raw)
    assert "jane.doe@company.com" not in cleaned
    assert "555-123-4567" not in cleaned
    assert "ORD-881234" not in cleaned
    assert "One moment while I check" not in cleaned
    assert "[EMAIL_REDACTED]" in cleaned


def test_prepare_and_eval_end_to_end(tmp_path):
    raw_csv = Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_transcripts.csv"
    counts = prepare_dataset(str(raw_csv), str(tmp_path), min_date="2024-01-01", test_size=0.3)
    assert counts["train"] >= 4
    assert counts["holdout_eval"] >= 2

    eval_samples = load_jsonl(str(tmp_path / "holdout_eval.jsonl"))
    report = run_evaluation(
        eval_samples,
        generate_fn=lambda msgs: "Happy to help! Here is the accurate resolution.",
        judge_fn=lambda sys_p, usr_p: {"accuracy_score": 0.95, "brand_alignment_score": 0.98, "notes": "Pass"},
    )
    assert report["meets_accuracy_target"] is True
    assert report["meets_brand_target"] is True


def test_serving_client_escalation_flag():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "I have flagged this for our billing specialist. [AGENT_NOTE: Verify account owner]"
                }
            }
        ]
    }
    with patch("serving_client.requests.post", return_value=mock_resp):
        client = FineTunedSupportClient(base_url="http://mock-vllm:8000/v1", model_name="support-llm")
        out = client.draft_reply("Please reset my API key")
        assert out["requires_human_review"] is True
