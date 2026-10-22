"""
LLM Story/Scene & Foley Enhancer (director.py)
Expands high-level user prompts into:
1. Rich visual prompts for Stage 1 (Keyframe) & Stage 2 (I2V / T2V Animation).
2. Acoustic Foley & ambient sound effect prompts for Stage 3 (AudioLDM 2 / AudioGen).
"""

import logging
import os
import re
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

DIRECTOR_SYSTEM_PROMPT = """You are an expert AI Video Director and Cinematographer specializing in latent video diffusion models (FLUX, SDXL, Wan2.1, and LTX-Video).
Your task is to take a user concept and translate it into ONE single, continuous 3-to-5-second cinematic camera shot (70-110 words).

Follow these strict rules:
1. Diffusion models generate ONE continuous shot (3-5 seconds), NOT a multi-scene movie montage. Never write "First...", "Then...", or cut between childhood and adulthood in a single clip.
2. Strip any conversational prefixes like "Can you generate a video of...".
3. Explicitly describe physical subject motion and camera movement (e.g., slow cinematic dolly in, smooth tracking shot, shallow depth of field).
4. Detail lighting, atmosphere, and visual style (e.g., volumetric golden hour lighting, anamorphic lens, 35mm film grain, photorealistic 4k cinematography).
5. Output ONLY the expanded prompt text. Do not include titles, quotes, markdown formatting, or explanations."""


class SceneDirector:
    """Lightweight multi-provider LLM client via httpx with automatic local visual & audio fallback."""

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout

    @staticmethod
    def _local_cinematic_rewrite(prompt: str) -> str:
        """
        Zero-cost, zero-API-key local prompt optimizer:
        1. Strips conversational filler ("Can you generate a video for...", "Please make a video of...")
        2. Supports multi-shot sequences separated by "||" (or automatically splits biographical
           "born in X ... grew to be Y in Z" prompts into a 2-shot story).
        3. Appends cinematic lighting, camera motion, and photorealistic quality cues.
        """
        if "||" in prompt:
            parts = [p.strip() for p in prompt.split("||") if p.strip()]
            return " || ".join(SceneDirector._rewrite_single_shot(p) for p in parts)

        cleaned = re.sub(
            r"^(can you |could you |please )?(generate|create|make|render|produce|show)( me)?( a)?( video| clip| animation| movie)?( of| for| about| showing| where)?\s*:?\s*",
            "",
            prompt.strip(),
            flags=re.IGNORECASE,
        ).strip()

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
            shot1 = (
                f"Cinematic golden hour tracking shot of a young Indian student in {origin}, "
                f"walking along a sunlit scenic street lined with lush green trees and traditional architecture, "
                f"carrying a notebook with an inspired, determined expression, warm sunlight, 35mm film lens, photorealistic 4k"
            )
            shot2 = (
                f"Cinematic medium shot of an accomplished Indian {role} inside a sleek modern glass skyscraper office in {city}, "
                f"typing on a laptop and looking out at the towering {city} skyline at twilight, "
                f"glowing city lights reflecting on glass, smooth dolly in, shallow depth of field, photorealistic 4k"
            )
            return f"{shot1} || {shot2}"

        return SceneDirector._rewrite_single_shot(cleaned)

    @staticmethod
    def _rewrite_single_shot(cleaned: str) -> str:
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

    @staticmethod
    def derive_audio_prompt(visual_prompt: str, custom_audio_prompt: Optional[str] = None) -> str:
        """
        Derives an optimized Foley / ambient soundscape prompt for AudioLDM 2 from the visual prompt
        whenever the user leaves the secondary audio prompt blank.
        """
        if custom_audio_prompt and custom_audio_prompt.strip():
            return custom_audio_prompt.strip()

        lower = (visual_prompt or "").lower()
        cues = []

        if any(w in lower for w in ["beach", "ocean", "sea", "wave", "coast", "shore"]):
            cues.append("gentle ocean waves washing onto shore, coastal breeze")
        if any(w in lower for w in ["rain", "storm", "thunder", "wet", "downpour"]):
            cues.append("steady rainfall pattering on surfaces, distant soft thunder")
        if any(w in lower for w in ["city", "nyc", "manhattan", "skyscraper", "street", "traffic", "urban"]):
            cues.append("subtle urban city ambience, distant traffic hum, soft indoor room tone")
        if any(w in lower for w in ["office", "laptop", "typing", "coding", "engineer", "desk", "computer"]):
            cues.append("soft mechanical keyboard typing clicks, quiet modern office ambience")
        if any(w in lower for w in ["forest", "tree", "jungle", "nature", "park", "orchard", "india", "village"]):
            cues.append("gentle wind rustling through leaves, warm outdoor nature ambience, distant birds chirping")
        if any(w in lower for w in ["walk", "walking", "footsteps", "running", "stepping"]):
            cues.append("soft natural footsteps in rhythm")
        if any(w in lower for w in ["fire", "campfire", "flame", "torch"]):
            cues.append("warm crackling fire and glowing embers")
        if any(w in lower for w in ["space", "astronaut", "alien", "sci-fi", "neon", "cyberpunk"]):
            cues.append("deep atmospheric sci-fi synth drone, subtle futuristic hum")

        if cues:
            return f"High quality realistic Foley sound effects and ambient audio: {', '.join(cues)}, clear stereo recording."

        # Clean visual camera jargon from fallback audio description
        stripped = re.sub(
            r"\b(cinematic|tracking shot|dolly in|close-up|medium shot|35mm|anamorphic|lens|photorealistic|4k|8k|shallow depth of field|volumetric lighting|color grading)\b",
            "",
            visual_prompt.split("||")[0],
            flags=re.IGNORECASE,
        )
        stripped = re.sub(r"\s+", " ", stripped).strip(" ,.")
        return f"Realistic ambient soundscape and natural Foley sound effects for: {stripped[:140]}, high fidelity audio."

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
