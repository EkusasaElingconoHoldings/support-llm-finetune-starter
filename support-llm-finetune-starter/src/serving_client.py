import os
from typing import Dict
import requests


class FineTunedSupportClient:
    def __init__(
        self,
        base_url: str = os.getenv("LLM_SERVING_URL", "http://localhost:8000/v1"),
        model_name: str = os.getenv("FINE_TUNED_MODEL_ID", "support-llm-finetuned"),
        api_key: str = os.getenv("LLM_API_KEY", "EMPTY"),
        timeout_sec: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def draft_reply(self, customer_inquiry: str, context_notes: str = "") -> Dict:
        system_content = (
            "You are a customer support assistant aligned with our brand voice. "
            "Provide accurate, concise, empathetic product help and mark [AGENT_NOTE: ...] "
            "whenever human review or account action is required."
        )
        user_content = customer_inquiry if not context_notes else f"{customer_inquiry}\n\nContext:\n{context_notes}"

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "temperature": 0.2,
            "max_tokens": 400,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()
        reply_text = data["choices"][0]["message"]["content"].strip()
        return {
            "draft_reply": reply_text,
            "requires_human_review": "[AGENT_NOTE:" in reply_text,
            "model": self.model_name,
        }
