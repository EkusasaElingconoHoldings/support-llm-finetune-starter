import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from firebase_admin import initialize_app
from firebase_functions import firestore_fn
from serving_client import FineTunedSupportClient

initialize_app()
llm_client = FineTunedSupportClient()


@firestore_fn.on_document_created(document="support_tickets/{ticket_id}")
def generate_sidecar_draft(event: firestore_fn.Event[firestore_fn.DocumentSnapshot]) -> None:
    """Non-blocking sidecar: writes draft reply to 'ai_sidecar' without altering existing ticket fields."""
    if not event.data:
        return

    ticket = event.data.to_dict() or {}
    if ticket.get("ai_sidecar", {}).get("status") == "completed":
        return

    inquiry = ticket.get("message") or ticket.get("description") or ""
    if not inquiry.strip():
        return

    try:
        result = llm_client.draft_reply(customer_inquiry=inquiry)
        event.data.reference.set(
            {
                "ai_sidecar": {
                    "draft_reply": result["draft_reply"],
                    "requires_human_review": result["requires_human_review"],
                    "model": result["model"],
                    "status": "completed",
                }
            },
            merge=True,
        )
    except Exception as exc:
        event.data.reference.set(
            {
                "ai_sidecar": {
                    "status": "failed",
                    "error": str(exc)[:300],
                }
            },
            merge=True,
        )
