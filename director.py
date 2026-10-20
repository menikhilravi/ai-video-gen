"""
LLM Story/Scene Enhancer (director.py)
Expands high-level user prompts into rich video diffusion prompts with explicit
camera movement, lighting, atmosphere, and temporal motion cues.
"""

import logging
import os
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

DIRECTOR_SYSTEM_PROMPT = """You are an expert AI Video Director and Cinematographer specializing in latent video diffusion models (Wan2.1 and LTX-Video).
Your task is to take a brief user concept and expand it into a single, cohesive, highly descriptive video generation prompt (60-110 words).

Follow these strict rules:
1. Explicitly describe subject motion and temporal progression (what moves and how it moves from start to finish).
2. Specify camera movement (e.g., slow dolly in, smooth tracking shot, low-angle pan, handheld cinematic movement, static tripod shot).
3. Detail lighting, atmosphere, and visual style (e.g., volumetric golden hour lighting, anamorphic lens flare, shallow depth of field, 35mm film grain, photorealistic texture).
4. Output ONLY the expanded prompt text. Do not include titles, quotes, markdown formatting, or explanations."""


class SceneDirector:
    """Lightweight multi-provider LLM client via httpx with automatic fallback."""

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout

    def enhance_prompt(
        self,
        prompt: str,
        use_enhancer: bool = False,
        api_key: Optional[str] = None,
        provider: str = "auto",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> str:
        """
        Expands a user prompt into a detailed video generation prompt.
        Falls back cleanly to the original prompt if disabled, missing an API key,
        or if the upstream API request fails.
        """
        cleaned_prompt = (prompt or "").strip()
        if not cleaned_prompt or not use_enhancer:
            return cleaned_prompt

        resolved_key = (api_key or "").strip() or self._get_env_api_key(provider)
        if not resolved_key:
            logger.info("No API key provided for SceneDirector; using raw prompt passthrough.")
            return cleaned_prompt

        resolved_provider = self._detect_provider(resolved_key, provider, base_url)

        try:
            if resolved_provider == "anthropic":
                expanded = self._call_anthropic(cleaned_prompt, resolved_key, model)
            elif resolved_provider == "gemini":
                expanded = self._call_gemini(cleaned_prompt, resolved_key, model)
            else:
                expanded = self._call_openai_compatible(
                    cleaned_prompt, resolved_key, model, base_url
                )

            if expanded and expanded.strip():
                return expanded.strip()
        except Exception as exc:
            logger.warning("SceneDirector expansion failed (%s). Falling back to raw prompt.", exc)

        return cleaned_prompt

    @staticmethod
    def _get_env_api_key(provider: str) -> str:
        provider = (provider or "auto").lower()
        if provider == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY", "")
        if provider == "gemini":
            return os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        if provider == "openai":
            return os.environ.get("OPENAI_API_KEY", "")
        return (
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or ""
        )

    @staticmethod
    def _detect_provider(api_key: str, provider: str, base_url: Optional[str] = None) -> str:
        if base_url and base_url.strip():
            return "openai"
        prov = (provider or "auto").lower()
        if prov in {"openai", "anthropic", "gemini"}:
            return prov
        if api_key.startswith("sk-ant-"):
            return "anthropic"
        if api_key.startswith("AIza"):
            return "gemini"
        return "openai"

    def _call_openai_compatible(
        self,
        prompt: str,
        api_key: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> str:
        endpoint = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        chosen_model = (model or "").strip() or "gpt-4o-mini"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": chosen_model,
            "messages": [
                {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
                {"role": "user", "content": f"Expand this scene for video generation: {prompt}"},
            ],
            "temperature": 0.7,
            "max_tokens": 250,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def _call_anthropic(
        self,
        prompt: str,
        api_key: str,
        model: Optional[str] = None,
    ) -> str:
        endpoint = "https://api.anthropic.com/v1/messages"
        chosen_model = (model or "").strip() or "claude-3-5-haiku-latest"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": chosen_model,
            "max_tokens": 250,
            "system": DIRECTOR_SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": f"Expand this scene for video generation: {prompt}"}
            ],
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]

    def _call_gemini(
        self,
        prompt: str,
        api_key: str,
        model: Optional[str] = None,
    ) -> str:
        chosen_model = (model or "").strip() or "gemini-2.0-flash"
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{chosen_model}:generateContent"
        )
        params = {"key": api_key}
        payload = {
            "systemInstruction": {"parts": [{"text": DIRECTOR_SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"Expand this scene for video generation: {prompt}"}],
                }
            ],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 250},
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(endpoint, params=params, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
