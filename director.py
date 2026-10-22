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

import re

DIRECTOR_SYSTEM_PROMPT = """You are an expert AI Video Director and Cinematographer specializing in latent video diffusion models (Wan2.1 and LTX-Video).
Your task is to take a user concept (even if it describes a multi-year biography or abstract story) and translate it into ONE single, continuous 3-to-5-second cinematic camera shot (70-110 words).

Follow these strict rules:
1. Diffusion models generate ONE continuous shot (3-5 seconds), NOT a multi-scene movie montage. Never write "First...", "Then...", or cut between childhood and adulthood in a single clip. If the user describes a life journey (e.g., born in India -> software engineer in NYC), synthesize it into one evocative cinematic moment (e.g., an Indian software engineer in a modern high-rise Manhattan office looking out over the glowing New York City skyline at dusk, warm golden reflections on the glass, coding on a sleek laptop).
2. Strip any conversational prefixes like "Can you generate a video of...".
3. Explicitly describe physical subject motion and camera movement (e.g., slow cinematic dolly in, smooth tracking shot, shallow depth of field).
4. Detail lighting, atmosphere, and visual style (e.g., volumetric golden hour lighting, anamorphic lens, 35mm film grain, photorealistic 4k cinematography).
5. Output ONLY the expanded prompt text. Do not include titles, quotes, markdown formatting, or explanations."""


class SceneDirector:
    """Lightweight multi-provider LLM client via httpx with automatic local visual fallback."""

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout

    @staticmethod
    def _local_cinematic_rewrite(prompt: str) -> str:
        """
        Zero-cost, zero-API-key local prompt optimizer:
        1. Strips conversational filler ("Can you generate a video for...", "Please make a video of...")
        2. Rewrites multi-era biographical prompts ("born in X and grew to be Y") into a single
           coherent visual shot so the 3-second diffusion model doesn't mash two eras together.
        3. Appends cinematic lighting, camera motion, and photorealistic quality cues.
        """
        cleaned = re.sub(
            r"^(can you |could you |please )?(generate|create|make|render|produce|show)( me)?( a)?( video| clip| animation| movie)?( of| for| about| showing| where)?\s*:?\s*",
            "",
            prompt.strip(),
            flags=re.IGNORECASE,
        ).strip()

        # Detect biographical "born in X ... grew/became Y in Z" patterns that confuse 3-second video diffusion
        bio_match = re.search(
            r"person born in ([^,]+(?:,\s*[^,]+)?).*?(?:grew|became|working|works).*?to be (?:a |an )?([^,.]+?) in ([A-Za-z\s]+)",
            cleaned,
            flags=re.IGNORECASE,
        )
        if bio_match:
            origin, role, city = (
                bio_match.group(1).strip(),
                bio_match.group(2).strip(),
                bio_match.group(3).strip(),
            )
            cleaned = (
                f"Cinematic medium close-up shot of an Indian {role} originally from {origin}, "
                f"standing inside a modern glass high-rise office in {city} with a glowing laptop, "
                f"looking out at the breathtaking {city} skyline at twilight, subtle warm smile, "
                f"gentle head turn as city lights shimmer in the background"
            )

        # Ensure cinematic camera & lighting descriptors are present
        lower = cleaned.lower()
        additions = []
        if not any(k in lower for k in ["shot", "camera", "dolly", "pan", "tracking", "close-up"]):
            additions.append("smooth cinematic tracking shot, shallow depth of field, 35mm anamorphic lens")
        if not any(k in lower for k in ["lighting", "light", "golden hour", "neon", "twilight", "sunlight"]):
            additions.append("natural volumetric lighting, rich color grading")
        if not any(k in lower for k in ["photorealistic", "4k", "8k", "cinematic", "film"]):
            additions.append("photorealistic, ultra-detailed texture, smooth natural motion")

        if additions:
            cleaned = f"{cleaned.rstrip('. ')}, {', '.join(additions)}."
        return cleaned

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
        Always cleans conversational prefixes; uses upstream LLM if enabled + key provided,
        or falls back to the built-in local cinematic rewriter.
        """
        cleaned_prompt = (prompt or "").strip()
        if not cleaned_prompt:
            return cleaned_prompt

        if not use_enhancer:
            return self._local_cinematic_rewrite(cleaned_prompt)

        resolved_key = (api_key or "").strip() or self._get_env_api_key(provider)
        if not resolved_key:
            logger.info("No API key provided for SceneDirector; using built-in local cinematic rewriter.")
            return self._local_cinematic_rewrite(cleaned_prompt)

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
            logger.warning("SceneDirector expansion failed (%s). Falling back to local cinematic rewriter.", exc)

        return self._local_cinematic_rewrite(cleaned_prompt)

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
