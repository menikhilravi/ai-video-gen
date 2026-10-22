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
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
        "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
        "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
        "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
        "three legs, many people in the background, walking backwards"
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
        self.low_system_ram: bool = self._detect_low_system_ram()

    @staticmethod
    def _detect_low_system_ram() -> bool:
        try:
            import os as _os
            total_ram_gb = (
                _os.sysconf("SC_PAGE_SIZE") * _os.sysconf("SC_PHYS_PAGES")
            ) / (1024**3)
            logger.info("Detected Host System RAM: %.1f GB", total_ram_gb)
            return total_ram_gb < 16.0
        except Exception:
            return False

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
        """Loads and configures the requested video diffusion pipeline with VRAM/RAM optimizations."""
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported model '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}"
            )

        if self.pipe is not None and self.current_model_key == model_key:
            return self.pipe

        self._unload_current_pipeline()
        spec = MODEL_REGISTRY[model_key]
        logger.info(
            "Loading %s (%s) on %s with dtype=%s (low_system_ram=%s)...",
            spec.display_name,
            spec.repo_id,
            self.device,
            self.dtype,
            self.low_system_ram,
        )

        if self.device == "cuda" and self.low_system_ram:
            # Colab Free Tier only provides ~12.7 GB System RAM, and Wan2.1's UMT5-XXL
            # text encoder alone is ~10-11 GB in fp16, which triggers the Linux OOM killer
            # if loaded into CPU RAM alongside Python/Gradio (~2 GB).
            # Solution: Keep ONLY the lightweight DiT (2.8 GB) + VAE (0.5 GB) in the persistent
            # pipeline on T4 VRAM, and stream the text encoder directly onto T4 VRAM ONLY during
            # the 1-second prompt encoding step! Peak CPU RAM stays under ~2.5 GB!
            pipe = DiffusionPipeline.from_pretrained(
                spec.repo_id,
                text_encoder=None,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            if hasattr(pipe, "transformer") and pipe.transformer is not None:
                pipe.transformer.to("cuda")
            if hasattr(pipe, "vae") and pipe.vae is not None:
                # CRITICAL for Wan2.1: The 3D Causal VAE MUST run in float32!
                # Running Wan's VAE in float16 on T4 causes severe numerical overflow & mushy/corrupted video.
                vae_dtype = torch.float32 if "Wan" in spec.repo_id else self.dtype
                pipe.vae.to(device="cuda", dtype=vae_dtype)
        else:
            pipe = DiffusionPipeline.from_pretrained(
                spec.repo_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            if hasattr(pipe, "vae") and pipe.vae is not None and "Wan" in spec.repo_id:
                pipe.vae.to(dtype=torch.float32)
            if self.device == "cuda":
                pipe.enable_model_cpu_offload()
            elif self.device == "mps":
                pipe.to("mps")
            else:
                pipe.to("cpu")

        # Configure UniPCMultistepScheduler with flow_shift=3.0 for 480p Wan2.1 generation
        # (Default flow_shift=5.0 is tuned for 720p and causes over-sharpening/weird motion at 480p)
        if "Wan" in spec.repo_id:
            try:
                from diffusers import UniPCMultistepScheduler
                pipe.scheduler = UniPCMultistepScheduler.from_config(
                    pipe.scheduler.config, flow_shift=3.0
                )
            except Exception as exc:
                logger.warning("Could not set flow_shift=3.0 on scheduler: %s", exc)

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

    def _encode_prompt_low_ram(
        self,
        pipe: DiffusionPipeline,
        spec: ModelSpec,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
    ):
        """
        Loads the text_encoder directly into T4 GPU VRAM (device_map={'': 'cuda'}),
        computes prompt embeddings in ~1 second, and immediately purges the text_encoder
        from VRAM so CPU RAM never spikes and GPU VRAM is freed for denoising.
        """
        from transformers import AutoModel, T5EncoderModel, UMT5EncoderModel

        # Temporarily move transformer to CPU (only 2.8 GB, easily fits in 12.7 GB CPU RAM)
        # while the 9.5 GB text encoder uses the 14.6 GB T4 GPU!
        if hasattr(pipe, "transformer") and pipe.transformer is not None:
            pipe.transformer.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Loading text_encoder directly onto GPU VRAM to avoid CPU RAM OOM...")
        encoder_cls = UMT5EncoderModel if "Wan" in spec.repo_id else T5EncoderModel
        try:
            text_encoder = encoder_cls.from_pretrained(
                spec.repo_id,
                subfolder="text_encoder",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map={"": "cuda"},
            )
        except Exception:
            text_encoder = AutoModel.from_pretrained(
                spec.repo_id,
                subfolder="text_encoder",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map={"": "cuda"},
            )

        pipe.text_encoder = text_encoder
        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=float(guidance_scale) > 1.0,
                device=torch.device("cuda"),
                dtype=self.dtype,
            )

        # Immediately purge text_encoder from GPU VRAM and move DiT back to GPU
        pipe.text_encoder = None
        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()

        if hasattr(pipe, "transformer") and pipe.transformer is not None:
            pipe.transformer.to("cuda")
        gc.collect()
        torch.cuda.empty_cache()

        return prompt_embeds, negative_prompt_embeds

    @staticmethod
    def _export_high_quality_mp4(frames, output_path: str, fps: int) -> None:
        """
        Exports frames using libx264 with CRF 18 (visually lossless, high bitrate)
        instead of diffusers' default export_to_video (which uses low default bitrate
        and creates heavy macroblocking artifacts).
        """
        try:
            import imageio
            import numpy as np

            writer = imageio.get_writer(
                output_path,
                fps=int(fps),
                codec="libx264",
                quality=None,
                pixelformat="yuv420p",
                ffmpeg_params=["-crf", "18", "-preset", "medium"],
            )
            for frame in frames:
                if hasattr(frame, "convert"):
                    arr = np.array(frame.convert("RGB"), dtype=np.uint8)
                else:
                    arr = np.asarray(frame)
                    if arr.dtype != np.uint8:
                        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
                writer.append_data(arr)
            writer.close()
        except Exception as exc:
            logger.warning("High-bitrate imageio writer fallback (%s); using export_to_video.", exc)
            export_to_video(frames, output_video_path=output_path, fps=int(fps))

    def generate_video(
        self,
        prompt: str,
        model_key: str = "Wan2.1-1.3B-T2V",
        num_frames: int = 81,
        width: int = 832,
        height: int = 480,
        num_inference_steps: int = 30,
        fps: int = 16,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[str] = None,
        seed: int = -1,
        output_path: Optional[str] = None,
        progress_callback=None,
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generates a video clip (or multi-shot stitched sequence if prompt contains multiple
        scenes separated by '|') and exports frames to a high-bitrate .mp4 file.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Please provide a non-empty scene description prompt.")

        spec = MODEL_REGISTRY[model_key]
        if progress_callback is not None:
            progress_callback(0.05, desc=f"Loading {spec.display_name} pipeline...")
        pipe = self.load_pipeline(model_key)

        effective_width = align_spatial_dim(width, spec.spatial_multiple)
        effective_height = align_spatial_dim(height, spec.spatial_multiple)
        effective_frames = align_frame_count(num_frames, spec.frame_divisor)
        effective_neg_prompt = (
            negative_prompt.strip()
            if negative_prompt and negative_prompt.strip()
            else spec.default_negative_prompt
        )

        # Support multi-shot prompts separated by "||" for 10-15 second stories
        shot_prompts = [p.strip() for p in prompt.split("||") if p.strip()]
        num_shots = len(shot_prompts)
        total_steps = int(num_inference_steps)

        all_frames = []

        for shot_idx, shot_prompt in enumerate(shot_prompts):
            shot_prefix = f"[Shot {shot_idx + 1}/{num_shots}] " if num_shots > 1 else ""

            generator = None
            if seed is not None and int(seed) >= 0:
                gen_device = "cuda" if self.device == "cuda" else "cpu"
                generator = torch.Generator(device=gen_device).manual_seed(int(seed) + shot_idx)

            logger.info(
                "%sGenerating video | model=%s | res=%dx%d | frames=%d | steps=%d | fps=%d",
                shot_prefix,
                model_key,
                effective_width,
                effective_height,
                effective_frames,
                total_steps,
                fps,
            )

            def _on_step_end(_pipe, step_index: int, _timestep, callback_kwargs):
                if progress_callback is not None:
                    shot_base = shot_idx / num_shots
                    shot_span = 0.85 / num_shots
                    frac = 0.10 + shot_base * 0.85 + shot_span * ((step_index + 1) / max(1, total_steps))
                    progress_callback(
                        min(0.94, frac),
                        desc=f"{shot_prefix}Denoising step {step_index + 1}/{total_steps}...",
                    )
                return callback_kwargs

            if progress_callback is not None:
                progress_callback(
                    0.08 + (shot_idx / num_shots) * 0.85,
                    desc=f"{shot_prefix}Encoding prompt on GPU...",
                )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
                if self.device == "cuda" and self.low_system_ram:
                    prompt_embeds, negative_prompt_embeds = self._encode_prompt_low_ram(
                        pipe=pipe,
                        spec=spec,
                        prompt=shot_prompt,
                        negative_prompt=effective_neg_prompt,
                        guidance_scale=float(guidance_scale),
                    )
                    pipe_kwargs["prompt_embeds"] = prompt_embeds
                    pipe_kwargs["negative_prompt_embeds"] = negative_prompt_embeds
                else:
                    pipe_kwargs["prompt"] = shot_prompt
                    pipe_kwargs["negative_prompt"] = effective_neg_prompt

                if progress_callback is not None:
                    progress_callback(
                        0.12 + (shot_idx / num_shots) * 0.85,
                        desc=f"{shot_prefix}Starting {total_steps} denoising steps...",
                    )

                result = pipe(**pipe_kwargs)

            shot_frames = list(result.frames[0])
            all_frames.extend(shot_frames)
            del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if progress_callback is not None:
            progress_callback(0.96, desc="Encoding high-bitrate H.264 .mp4 video...")

        if output_path is None:
            tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            output_path = tmp_file.name
            tmp_file.close()

        self._export_high_quality_mp4(all_frames, output_path=output_path, fps=int(fps))
        total_rendered_frames = len(all_frames)

        del all_frames
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if progress_callback is not None:
            progress_callback(1.0, desc="Complete!")

        metadata = {
            "width": effective_width,
            "height": effective_height,
            "num_frames": total_rendered_frames,
            "fps": int(fps),
            "steps": total_steps,
            "shots": num_shots,
        }
        return output_path, metadata
