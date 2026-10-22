"""
Video Generation Engine (engine.py)
Wraps Hugging Face Diffusers text-to-video pipelines (Wan2.1-1.3B-T2V and Lightricks/LTX-Video)
with low-VRAM optimizations (<12-14 GB VRAM for T4 / consumer GPUs).
"""

import gc
import logging
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

# Guard against Colab's pre-installed torchao version mismatch with diffusers
try:
    import torchao.quantization as _tao_quant
    if not hasattr(_tao_quant, "FqnToConfig"):
        _tao_quant.FqnToConfig = dict
except Exception:
    pass

from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    repo_id: str
    spatial_multiple: int
    frame_divisor: int = 4  # temporal compression stride (4k + 1 or 8k + 1)
    default_negative_prompt: str = (
        "low quality, worst quality, blurry, flickering, distorted, deformed, "
        "watermark, text, static, overexposed, underexposed, compression artifacts"
    )


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "Wan2.1-1.3B-T2V": ModelSpec(
        display_name="Wan2.1-1.3B-T2V",
        repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        spatial_multiple=16,
        frame_divisor=4,
    ),
    "Lightricks/LTX-Video": ModelSpec(
        display_name="Lightricks/LTX-Video",
        repo_id="Lightricks/LTX-Video",
        spatial_multiple=32,
        frame_divisor=8,
    ),
}


def detect_compute_backend() -> Tuple[str, torch.dtype]:
    """
    Automatically detects CUDA, MPS, or CPU and picks the optimal floating-point dtype.
    """
    if torch.cuda.is_available():
        # Prefer float16 on T4/Turing/Volta for strict VRAM budget, or bfloat16 if Ampere+
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return "cuda", dtype
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def align_spatial_dim(value: int, multiple: int) -> int:
    """Rounds a spatial dimension (height/width) to the nearest valid multiple."""
    return max(multiple, int(round(value / multiple) * multiple))


def align_frame_count(num_frames: int, divisor: int) -> int:
    """
    Wan2.1 requires (num_frames - 1) % 4 == 0 (e.g., 17, 21, 25, 33, 49, 81).
    LTX-Video requires (num_frames - 1) % 8 == 0 (e.g., 9, 17, 25, 33, 41, 49).
    If the user passes 16 or 25, this safely snaps to a valid latent frame count (e.g., 17 or 25)
    so the 3D VAE never raises a shape mismatch error.
    """
    if num_frames <= 1:
        return divisor + 1
    rem = (num_frames - 1) % divisor
    if rem == 0:
        return num_frames
    # Snap to nearest valid k * divisor + 1
    lower = num_frames - rem
    upper = lower + divisor
    return upper if (num_frames - lower) >= (upper - num_frames) else max(divisor + 1, lower)


class VideoGenerationEngine:
    """Manages pipeline lifecycle, VRAM offloading, and video export."""

    def __init__(self):
        self.device, self.dtype = detect_compute_backend()
        self.current_model_key: Optional[str] = None
        self.pipe: Optional[DiffusionPipeline] = None

    def _unload_current_pipeline(self) -> None:
        if self.pipe is not None:
            logger.info("Unloading active pipeline (%s) to free memory...", self.current_model_key)
            del self.pipe
            self.pipe = None
            self.current_model_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_pipeline(self, model_key: str) -> DiffusionPipeline:
        """Loads and configures the requested video diffusion pipeline with VRAM optimizations."""
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported model '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}"
            )

        if self.pipe is not None and self.current_model_key == model_key:
            return self.pipe

        self._unload_current_pipeline()
        spec = MODEL_REGISTRY[model_key]
        logger.info(
            "Loading %s (%s) on %s with dtype=%s...",
            spec.display_name,
            spec.repo_id,
            self.device,
            self.dtype,
        )

        # Check available system RAM (Colab free tier only provides ~12.7 GB system RAM,
        # whereas Wan2.1's UMT5-XXL text encoder (9.5 GB) + DiT (2.8 GB) + VAE (0.3 GB) = 12.6 GB in fp16,
        # which triggers the Linux OOM killer if all components reside in CPU RAM simultaneously).
        total_ram_gb = 32.0
        try:
            import os as _os
            total_ram_gb = (
                _os.sysconf("SC_PAGE_SIZE") * _os.sysconf("SC_PHYS_PAGES")
            ) / (1024**3)
        except Exception:
            pass

        logger.info("Detected System RAM: %.1f GB", total_ram_gb)

        if self.device == "cuda" and total_ram_gb < 16.0:
            # Colab Free Tier only has 12.7 GB System RAM, whereas Wan2.1's UMT5-XXL (9.5 GB)
            # + DiT (2.8 GB) + VAE (0.3 GB) = 12.6 GB total. Loading all three into System RAM
            # at once triggers Linux's OOM killer ("Connection to the server was lost").
            # Solution: Load pipeline WITHOUT text_encoder first, move DiT + VAE to T4 VRAM (3.1 GB),
            # then load the 9.5 GB text_encoder into CPU RAM so System RAM never exceeds ~10 GB!
            logger.info(
                "Colab Low-RAM Mode (%.1f GB System RAM): Staged loading across T4 VRAM + System RAM "
                "to prevent host OOM kills...",
                total_ram_gb,
            )
            from transformers import AutoModel, UMT5EncoderModel, T5EncoderModel

            pipe = DiffusionPipeline.from_pretrained(
                spec.repo_id,
                text_encoder=None,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            if hasattr(pipe, "transformer") and pipe.transformer is not None:
                pipe.transformer.to("cuda")
            if hasattr(pipe, "vae") and pipe.vae is not None:
                pipe.vae.to("cuda")
            gc.collect()
            torch.cuda.empty_cache()

            # Now System RAM is completely free (~1.5 GB used); safely load the 9.5 GB text encoder on CPU
            encoder_cls = UMT5EncoderModel if "Wan" in spec.repo_id else T5EncoderModel
            try:
                text_encoder = encoder_cls.from_pretrained(
                    spec.repo_id,
                    subfolder="text_encoder",
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                )
            except Exception:
                text_encoder = AutoModel.from_pretrained(
                    spec.repo_id,
                    subfolder="text_encoder",
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                )
            pipe.register_modules(text_encoder=text_encoder)
        else:
            # Standard load when System RAM >= 16 GB
            pipe = DiffusionPipeline.from_pretrained(
                spec.repo_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            if self.device == "cuda":
                pipe.enable_model_cpu_offload()
            elif self.device == "mps":
                pipe.to("mps")
            else:
                pipe.to("cpu")

        # Enable VAE slicing & tiling when supported by the pipeline/VAE
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()

        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.pipe = pipe
        self.current_model_key = model_key
        return pipe

    def generate_video(
        self,
        prompt: str,
        model_key: str = "Wan2.1-1.3B-T2V",
        num_frames: int = 25,
        width: int = 480,
        height: int = 272,
        num_inference_steps: int = 20,
        fps: int = 8,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[str] = None,
        seed: int = -1,
        output_path: Optional[str] = None,
        progress_callback=None,
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generates a video clip from text and exports frames directly to an .mp4 file.
        Returns the path to the generated .mp4 file and a metadata dictionary of effective dimensions.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Please provide a non-empty scene description prompt.")

        spec = MODEL_REGISTRY[model_key]
        if progress_callback is not None:
            progress_callback(0.05, desc=f"Loading {spec.display_name} weights...")
        pipe = self.load_pipeline(model_key)

        effective_width = align_spatial_dim(width, spec.spatial_multiple)
        effective_height = align_spatial_dim(height, spec.spatial_multiple)
        effective_frames = align_frame_count(num_frames, spec.frame_divisor)
        effective_neg_prompt = (
            negative_prompt.strip()
            if negative_prompt and negative_prompt.strip()
            else spec.default_negative_prompt
        )

        generator = None
        if seed is not None and int(seed) >= 0:
            gen_device = "cuda" if self.device == "cuda" else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(int(seed))

        logger.info(
            "Generating video | model=%s | res=%dx%d | frames=%d | steps=%d | fps=%d",
            model_key,
            effective_width,
            effective_height,
            effective_frames,
            num_inference_steps,
            fps,
        )

        total_steps = int(num_inference_steps)

        def _on_step_end(_pipe, step_index: int, _timestep, callback_kwargs):
            if progress_callback is not None:
                frac = 0.1 + 0.8 * ((step_index + 1) / max(1, total_steps))
                progress_callback(
                    min(0.92, frac),
                    desc=f"Denoising step {step_index + 1}/{total_steps}...",
                )
            return callback_kwargs

        if progress_callback is not None:
            progress_callback(0.1, desc="Encoding prompt & starting diffusion...")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # If text_encoder is still on CPU while transformer is on CUDA (Colab Low-RAM split mode),
        # encode the prompt first on GPU/CPU or temporarily move text_encoder to GPU if VRAM allows.
        pipe_kwargs = dict(
            height=effective_height,
            width=effective_width,
            num_frames=effective_frames,
            num_inference_steps=total_steps,
            guidance_scale=float(guidance_scale),
            generator=generator,
            callback_on_step_end=_on_step_end,
        )

        with torch.inference_mode():
            if (
                self.device == "cuda"
                and hasattr(pipe, "text_encoder")
                and pipe.text_encoder is not None
                and pipe.text_encoder.device.type == "cpu"
                and not hasattr(pipe, "_all_hooks")
            ):
                # Temporarily move text_encoder to GPU (9.5 GB + 3.1 GB = 12.6 GB < 14.6 GB T4 VRAM),
                # encode prompt embeddings at GPU speed, then immediately return text_encoder to CPU!
                try:
                    pipe.text_encoder.to("cuda")
                    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                        prompt=prompt.strip(),
                        negative_prompt=effective_neg_prompt,
                        do_classifier_free_guidance=float(guidance_scale) > 1.0,
                        device=torch.device("cuda"),
                        dtype=self.dtype,
                    )
                    pipe.text_encoder.to("cpu")
                    gc.collect()
                    torch.cuda.empty_cache()
                    pipe_kwargs["prompt_embeds"] = prompt_embeds
                    pipe_kwargs["negative_prompt_embeds"] = negative_prompt_embeds
                except Exception as exc:
                    logger.warning("Fast GPU prompt encode fallback (%s); encoding via standard call.", exc)
                    pipe.text_encoder.to("cpu")
                    gc.collect()
                    torch.cuda.empty_cache()
                    pipe_kwargs["prompt"] = prompt.strip()
                    pipe_kwargs["negative_prompt"] = effective_neg_prompt
            else:
                pipe_kwargs["prompt"] = prompt.strip()
                pipe_kwargs["negative_prompt"] = effective_neg_prompt

            result = pipe(**pipe_kwargs)

        frames = result.frames[0]

        if progress_callback is not None:
            progress_callback(0.95, desc="Encoding .mp4 video...")

        if output_path is None:
            tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            output_path = tmp_file.name
            tmp_file.close()

        export_to_video(frames, output_video_path=output_path, fps=int(fps))

        del result, frames
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if progress_callback is not None:
            progress_callback(1.0, desc="Complete!")

        metadata = {
            "width": effective_width,
            "height": effective_height,
            "num_frames": effective_frames,
            "fps": int(fps),
            "steps": total_steps,
        }
        return output_path, metadata
