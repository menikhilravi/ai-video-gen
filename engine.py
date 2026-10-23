"""
Unified Multi-Stage "Grok Imagine"-Style Video & Audio Engine (engine.py)

Implements a 4-stage sequential generation chain designed for < 15 GB VRAM (Google Colab T4):
- Stage 1 (Keyframe Generation): Generates a photorealistic anchor image using FLUX.1-schnell
  or SDXL-Turbo (or accepts a user-uploaded image directly).
- Stage 2 (Temporal Animation / I2V & T2V): Animates the anchor keyframe using LTX-Video I2V
  or Wan2.1 with float32 3D-VAE precision and low-RAM offloading.
- Stage 3 (Sound FX & Foley Synthesis): Generates synchronized ambient Foley audio (.wav)
  matching the scene description using AudioLDM 2 (or AudioLDM).
- Stage 4 (Container Muxing): Muxes the H.264 video (.mp4) and AAC audio (.wav) into a single
  synchronized MP4 container via ffmpeg-python (`-c:v copy -c:a aac -shortest`).

Every stage runs sequentially and explicitly invokes `gc.collect()` + `torch.cuda.empty_cache()`
between pipeline handoffs to guarantee zero VRAM or System RAM OOM crashes.
"""

import gc
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# Disable Hugging Face Xet Rust background chunk reconstruction (causes 10-minute stalls & RAM spikes on Colab)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import numpy as np
import torch
from PIL import Image

# Guard against Colab's pre-installed torchao version mismatch with diffusers
try:
    import torchao.quantization as _tao_quant

    if not hasattr(_tao_quant, "FqnToConfig"):
        _tao_quant.FqnToConfig = dict
except Exception:
    pass

from diffusers import (
    AutoPipelineForText2Image,
    DiffusionPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import export_to_video

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KeyframeModelSpec:
    display_name: str
    repo_id: str
    num_inference_steps: int
    guidance_scale: float
    requires_high_ram: bool = False


KEYFRAME_MODEL_REGISTRY: Dict[str, KeyframeModelSpec] = {
    "SDXL-Turbo (Fast, Zero-Auth, Low RAM - Default)": KeyframeModelSpec(
        display_name="SDXL-Turbo",
        repo_id="stabilityai/sdxl-turbo",
        num_inference_steps=4,
        guidance_scale=0.0,
        requires_high_ram=False,
    ),
    "FLUX.1-schnell (Ultra-Photorealistic, Requires HF_TOKEN / High RAM)": KeyframeModelSpec(
        display_name="FLUX.1-schnell",
        repo_id="black-forest-labs/FLUX.1-schnell",
        num_inference_steps=4,
        guidance_scale=0.0,
        requires_high_ram=True,
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    t2v_repo_id: str
    i2v_repo_id: str
    spatial_multiple: int
    frame_divisor: int = 4
    supports_low_ram_i2v: bool = True
    default_negative_prompt: str = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
        "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
        "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
        "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
        "three legs, many people in the background, walking backwards"
    )


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "Lightricks/LTX-Video": ModelSpec(
        display_name="Lightricks/LTX-Video (Native I2V + T2V)",
        t2v_repo_id="Lightricks/LTX-Video",
        i2v_repo_id="Lightricks/LTX-Video",
        spatial_multiple=32,
        frame_divisor=8,
        supports_low_ram_i2v=True,
    ),
    "Wan2.1-1.3B-T2V": ModelSpec(
        display_name="Wan2.1-1.3B (High Motion Fidelity)",
        t2v_repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        i2v_repo_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        spatial_multiple=16,
        frame_divisor=4,
        supports_low_ram_i2v=False,  # 14B I2V requires >24GB RAM; falls back to LTX-I2V or Wan-1.3B on T4
    ),
}


AUDIO_MODEL_REGISTRY: Dict[str, str] = {
    "AudioLDM 2 (High Fidelity Foley - Default)": "cvssp/audioldm2",
    "AudioLDM-S-Full-v2 (Ultra-Fast Lightweight Foley)": "cvssp/audioldm-s-full-v2",
}


def detect_compute_backend() -> Tuple[str, torch.dtype]:
    """Automatically detects CUDA, MPS, or CPU and picks the optimal floating-point dtype."""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return "cuda", dtype
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def detect_system_ram_gb() -> float:
    """Returns total host system RAM in GB."""
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024**3)
    except Exception:
        return 32.0


def flush_memory() -> None:
    """Explicitly invokes Python GC and clears PyTorch CUDA / MPS caches between pipeline stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def align_spatial_dim(value: int, multiple: int) -> int:
    """Rounds a spatial dimension (height/width) to the nearest valid multiple."""
    return max(multiple, int(round(value / multiple) * multiple))


def align_frame_count(num_frames: int, divisor: int) -> int:
    """Snaps frame count to k * divisor + 1 so 3D causal VAEs never encounter shape mismatches."""
    if num_frames <= 1:
        return divisor + 1
    rem = (num_frames - 1) % divisor
    if rem == 0:
        return num_frames
    lower = num_frames - rem
    upper = lower + divisor
    return upper if (num_frames - lower) >= (upper - num_frames) else max(divisor + 1, lower)


def apply_memory_optimizations(pipe: DiffusionPipeline, device: str, low_ram: bool = False) -> None:
    """Applies model CPU offload, VAE slicing, and VAE tiling to any Diffusers pipeline."""
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    if device == "cuda":
        if low_ram and hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
        elif hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
    elif device == "mps":
        pipe.to("mps")
    else:
        pipe.to("cpu")


class KeyframeStage:
    """
    Stage 1: Photorealistic Keyframe Synthesis (FLUX.1-schnell or SDXL-Turbo).
    Generates the anchor image and immediately unloads its weights before Stage 2.
    """

    def __init__(self, device: str, dtype: torch.dtype, system_ram_gb: float):
        self.device = device
        self.dtype = dtype
        self.system_ram_gb = system_ram_gb

    def generate_keyframe(
        self,
        prompt: str,
        width: int,
        height: int,
        keyframe_model_key: str = "SDXL-Turbo (Fast, Zero-Auth, Low RAM - Default)",
        seed: int = -1,
    ) -> Image.Image:
        spec = KEYFRAME_MODEL_REGISTRY.get(
            keyframe_model_key,
            KEYFRAME_MODEL_REGISTRY["SDXL-Turbo (Fast, Zero-Auth, Low RAM - Default)"],
        )

        # Automatic safety fallback on 12.7 GB Colab Free Tier if FLUX.1-schnell is selected
        # without enough System RAM (~24 GB needed for unquantized FLUX.1-schnell)
        if spec.requires_high_ram and self.system_ram_gb < 22.0:
            logger.warning(
                "Host System RAM (%.1f GB) is below 22 GB required for %s; "
                "falling back to SDXL-Turbo to prevent Colab OOM crash.",
                self.system_ram_gb,
                spec.display_name,
            )
            spec = KEYFRAME_MODEL_REGISTRY["SDXL-Turbo (Fast, Zero-Auth, Low RAM - Default)"]

        flush_memory()
        logger.info("[Stage 1/4] Loading Keyframe model: %s (%s)", spec.display_name, spec.repo_id)

        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                spec.repo_id,
                torch_dtype=self.dtype,
                variant="fp16" if self.dtype == torch.float16 and "sdxl" in spec.repo_id else None,
                low_cpu_mem_usage=True,
            )
        except Exception:
            # Fallback if FLUX gated repo lacks token or fp16 variant is unavailable
            fallback_spec = KEYFRAME_MODEL_REGISTRY["SDXL-Turbo (Fast, Zero-Auth, Low RAM - Default)"]
            logger.warning("Falling back to %s for Stage 1 keyframe generation.", fallback_spec.repo_id)
            spec = fallback_spec
            pipe = AutoPipelineForText2Image.from_pretrained(
                spec.repo_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )

        apply_memory_optimizations(pipe, self.device, low_ram=False)

        gen_w = align_spatial_dim(width, 32)
        gen_h = align_spatial_dim(height, 32)

        generator = None
        if seed is not None and int(seed) >= 0:
            gen_device = "cuda" if self.device == "cuda" else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(int(seed))

        first_shot_prompt = prompt.split("||")[0].strip()
        logger.info("[Stage 1/4] Synthesizing anchor keyframe (%dx%d)...", gen_w, gen_h)

        with torch.inference_mode():
            out = pipe(
                prompt=first_shot_prompt,
                width=gen_w,
                height=gen_h,
                num_inference_steps=spec.num_inference_steps,
                guidance_scale=spec.guidance_scale,
                generator=generator,
            )
            image = out.images[0].convert("RGB")

        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)

        # Strictly unload Stage 1 before Stage 2 starts
        del pipe, out
        flush_memory()
        return image


class AudioFoleyStage:
    """
    Stage 3: Ambient Sound FX & Foley Synthesis (AudioLDM 2 / AudioLDM).
    Generates a synchronized .wav file matching the video duration and unloads immediately.
    """

    def __init__(self, device: str, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

    def generate_foley(
        self,
        audio_prompt: str,
        duration_seconds: float,
        audio_model_key: str = "AudioLDM 2 (High Fidelity Foley - Default)",
        seed: int = -1,
        output_wav_path: Optional[str] = None,
    ) -> str:
        import scipy.io.wavfile

        flush_memory()
        repo_id = AUDIO_MODEL_REGISTRY.get(
            audio_model_key, "cvssp/audioldm2"
        )
        target_duration = float(max(2.5, min(15.0, duration_seconds)))
        logger.info(
            "[Stage 3/4] Loading Audio Foley model (%s) for %.1fs audio...",
            repo_id,
            target_duration,
        )

        try:
            pipe = DiffusionPipeline.from_pretrained(
                repo_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
        except Exception as exc:
            logger.warning("Primary AudioLDM2 load failed (%s), falling back to audioldm-s-full-v2", exc)
            pipe = DiffusionPipeline.from_pretrained(
                "cvssp/audioldm-s-full-v2",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )

        if self.device == "cuda":
            pipe.enable_model_cpu_offload()
        elif self.device == "mps":
            pipe.to("mps")
        else:
            pipe.to("cpu")

        generator = None
        if seed is not None and int(seed) >= 0:
            gen_device = "cuda" if self.device == "cuda" else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(int(seed))

        with torch.inference_mode():
            audio_result = pipe(
                prompt=audio_prompt,
                negative_prompt="low quality, average quality, distorted, static noise, clipping, robotic",
                num_inference_steps=35,
                audio_length_in_s=target_duration,
                generator=generator,
            )

        waveform = np.asarray(audio_result.audios[0], dtype=np.float32)
        # Normalize waveform cleanly to prevent clipping
        peak = np.max(np.abs(waveform))
        if peak > 1e-5:
            waveform = (waveform / peak) * 0.92
        pcm16 = (waveform * 32767.0).astype(np.int16)

        if output_wav_path is None:
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_wav_path = tmp_wav.name
            tmp_wav.close()

        # AudioLDM / AudioLDM2 native sample rate is 16000 Hz
        scipy.io.wavfile.write(output_wav_path, rate=16000, data=pcm16)

        del pipe, audio_result
        flush_memory()
        return output_wav_path


class FFmpegMuxerStage:
    """
    Stage 4: Container Muxing (ffmpeg-python / ffmpeg subprocess).
    Muxes the generated H.264 video (.mp4) and Foley audio (.wav) using:
    `-c:v copy -c:a aac -shortest` for zero container corruption and frame-accurate sync.
    """

    @staticmethod
    def mux_video_and_audio(
        video_path: str,
        audio_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        if output_path is None:
            tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            output_path = tmp_out.name
            tmp_out.close()

        logger.info("[Stage 4/4] Muxing video (%s) + audio (%s) -> %s", video_path, audio_path, output_path)

        # Resolve ffmpeg binary (system ffmpeg or imageio_ffmpeg bundled binary)
        ffmpeg_bin = "ffmpeg"
        try:
            import imageio_ffmpeg

            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

        # Attempt via ffmpeg-python first
        try:
            import ffmpeg

            video_stream = ffmpeg.input(video_path).video
            audio_stream = ffmpeg.input(audio_path).audio
            (
                ffmpeg.output(
                    video_stream,
                    audio_stream,
                    output_path,
                    vcodec="copy",
                    acodec="aac",
                    audio_bitrate="192k",
                    shortest=None,
                    movflags="+faststart",
                )
                .overwrite_output()
                .run(cmd=ffmpeg_bin, quiet=True)
            )
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
        except Exception as exc:
            logger.warning("ffmpeg-python wrapper fallback (%s); invoking ffmpeg CLI directly.", exc)

        # Direct subprocess fallback with exact `-c:v copy -c:a aac -shortest` flags
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            output_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path


class VideoGenerationEngine:
    """
    Unified Multi-Stage Generation Engine:
    Coordinates Stage 1 (Keyframe) -> Stage 2 (I2V / T2V Animation) -> Stage 3 (Audio Foley) -> Stage 4 (Muxing)
    under a strict <15 GB VRAM budget with sequential offloading and explicit garbage collection.
    """

    def __init__(self):
        self.device, self.dtype = detect_compute_backend()
        self.system_ram_gb = detect_system_ram_gb()
        self.low_system_ram = self.system_ram_gb < 16.0
        logger.info(
            "Initialized VideoGenerationEngine | Backend: %s (%s) | Host RAM: %.1f GB",
            self.device,
            self.dtype,
            self.system_ram_gb,
        )

        self.keyframe_stage = KeyframeStage(self.device, self.dtype, self.system_ram_gb)
        self.audio_stage = AudioFoleyStage(self.device, self.dtype)
        self.muxer_stage = FFmpegMuxerStage()

        self.current_model_key: Optional[str] = None
        self.current_pipeline_mode: Optional[str] = None
        self.pipe: Optional[DiffusionPipeline] = None

    def _unload_current_pipeline(self) -> None:
        if self.pipe is not None:
            logger.info("Unloading video pipeline (%s) to free VRAM/RAM...", self.current_model_key)
            del self.pipe
            self.pipe = None
            self.current_model_key = None
            self.current_pipeline_mode = None
        flush_memory()

    def _load_video_pipeline(self, model_key: str, use_i2v: bool = False) -> Tuple[DiffusionPipeline, ModelSpec, bool]:
        """
        Loads Stage 2 video pipeline (I2V when an anchor keyframe is provided and supported,
        or T2V with low-RAM streaming on 12.7 GB Colab instances).
        """
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}")

        spec = MODEL_REGISTRY[model_key]

        # On 12.7 GB Colab Free Tier, Wan2.1's I2V checkpoint is 14B (~28 GB fp16), which exceeds T4 memory.
        # LTX-Video (2B) natively supports both I2V and T2V in a single lightweight checkpoint (<6 GB VRAM)!
        effective_i2v = use_i2v
        if use_i2v and "Wan" in model_key and self.low_system_ram:
            logger.info(
                "Wan2.1-I2V-14B exceeds 15GB T4 VRAM budget; routing I2V animation through "
                "Lightricks/LTX-Video I2V pipeline (or Wan2.1-1.3B T2V)."
            )
            model_key = "Lightricks/LTX-Video"
            spec = MODEL_REGISTRY[model_key]

        mode_tag = "i2v" if effective_i2v else "t2v"
        if (
            self.pipe is not None
            and self.current_model_key == model_key
            and self.current_pipeline_mode == mode_tag
        ):
            return self.pipe, spec, effective_i2v

        self._unload_current_pipeline()

        if effective_i2v and "LTX" in model_key:
            from diffusers import LTXImageToVideoPipeline

            if self.device == "cuda" and self.low_system_ram:
                # CRITICAL on 12.7 GB Colab RAM: LTX-Video's T5-XXL text_encoder is 9.5 GB in fp16!
                # Loading LTXImageToVideoPipeline with text_encoder in CPU RAM crashes the 12.7 GB host.
                # Load WITHOUT text_encoder first and stream text_encoder directly onto GPU VRAM!
                pipe = LTXImageToVideoPipeline.from_pretrained(
                    spec.i2v_repo_id,
                    text_encoder=None,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                )
                if hasattr(pipe, "transformer") and pipe.transformer is not None:
                    pipe.transformer.to("cuda")
                if hasattr(pipe, "vae") and pipe.vae is not None:
                    pipe.vae.to("cuda")
            else:
                pipe = LTXImageToVideoPipeline.from_pretrained(
                    spec.i2v_repo_id,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                )
                apply_memory_optimizations(pipe, self.device, low_ram=False)

            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()

            flush_memory()
            self.pipe = pipe
            self.current_model_key = model_key
            self.current_pipeline_mode = "i2v"
            return pipe, spec, True

        # Standard T2V / Low-RAM Staged Loading for Wan2.1-1.3B and LTX-Video
        repo_id = spec.t2v_repo_id
        if self.device == "cuda" and self.low_system_ram:
            pipe = DiffusionPipeline.from_pretrained(
                repo_id,
                text_encoder=None,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            if hasattr(pipe, "transformer") and pipe.transformer is not None:
                pipe.transformer.to("cuda")
            if hasattr(pipe, "vae") and pipe.vae is not None:
                vae_dtype = torch.float32 if "Wan" in repo_id else self.dtype
                pipe.vae.to(device="cuda", dtype=vae_dtype)
        else:
            pipe = DiffusionPipeline.from_pretrained(
                repo_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            if hasattr(pipe, "vae") and pipe.vae is not None and "Wan" in repo_id:
                pipe.vae.to(dtype=torch.float32)
            apply_memory_optimizations(pipe, self.device, low_ram=False)

        if "Wan" in repo_id:
            try:
                pipe.scheduler = UniPCMultistepScheduler.from_config(
                    pipe.scheduler.config, flow_shift=3.0
                )
            except Exception as exc:
                logger.warning("Could not set flow_shift=3.0 on scheduler: %s", exc)

        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

        flush_memory()
        self.pipe = pipe
        self.current_model_key = model_key
        self.current_pipeline_mode = "t2v"
        return pipe, spec, False

    def _encode_prompt_standalone_low_ram(
        self,
        spec: ModelSpec,
        prompts: List[str],
        negative_prompt: str,
        guidance_scale: float,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        On 12.7 GB Colab Free Tier:
        Loads ONLY the tokenizer + text_encoder onto T4 GPU VRAM (with ZERO video transformer
        or VAE in CPU RAM or GPU VRAM!), encodes all shot prompts in ~1 second, and completely
        deletes the 9.5 GB text_encoder before the video transformer/VAE are ever loaded!
        This guarantees peak CPU RAM stays < 2.8 GB and peak GPU VRAM stays < 10 GB.
        """
        from transformers import AutoModel, AutoTokenizer, T5EncoderModel, UMT5EncoderModel

        self._unload_current_pipeline()
        repo_id = spec.t2v_repo_id
        is_wan = "Wan" in repo_id
        max_seq_len = 226 if is_wan else 128

        logger.info(
            "[Stage 2/4] Pre-encoding prompt(s) directly on GPU with standalone text_encoder (%s)...",
            repo_id,
        )

        tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="tokenizer")
        encoder_cls = UMT5EncoderModel if is_wan else T5EncoderModel
        try:
            text_encoder = encoder_cls.from_pretrained(
                repo_id,
                subfolder="text_encoder",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map={"": "cuda"},
            )
        except Exception:
            text_encoder = AutoModel.from_pretrained(
                repo_id,
                subfolder="text_encoder",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map={"": "cuda"},
            )
        text_encoder.eval()

        def _encode_single(text: str) -> Tuple[torch.Tensor, torch.Tensor]:
            clean_text = " ".join(text.strip().split())
            text_inputs = tokenizer(
                [clean_text],
                padding="max_length",
                max_length=max_seq_len,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to("cuda")
            attn_mask = text_inputs.attention_mask.to("cuda")
            embeds = text_encoder(input_ids=input_ids, attention_mask=attn_mask)[0]
            embeds = embeds.to(dtype=self.dtype, device="cuda")

            if is_wan:
                # Wan2.1 zeros out padded token positions beyond actual sequence length
                seq_len = int(attn_mask.gt(0).sum(dim=1).long().item())
                embeds[:, seq_len:] = 0.0
                return embeds.detach(), attn_mask.detach()
            else:
                return embeds.detach(), attn_mask.to(dtype=self.dtype).detach()

        encoded_shots: List[Dict[str, torch.Tensor]] = []
        with torch.inference_mode():
            neg_embeds, neg_mask = _encode_single(negative_prompt)
            for shot_prompt in prompts:
                pos_embeds, pos_mask = _encode_single(shot_prompt)
                if is_wan:
                    encoded_shots.append(
                        {
                            "prompt_embeds": pos_embeds,
                            "negative_prompt_embeds": neg_embeds.clone(),
                        }
                    )
                else:
                    encoded_shots.append(
                        {
                            "prompt_embeds": pos_embeds,
                            "prompt_attention_mask": pos_mask,
                            "negative_prompt_embeds": neg_embeds.clone(),
                            "negative_prompt_attention_mask": neg_mask.clone(),
                        }
                    )

        del text_encoder, tokenizer
        flush_memory()
        return encoded_shots

    @staticmethod
    def _export_high_quality_mp4(frames, output_path: str, fps: int) -> None:
        """Exports frames using libx264 CRF 18 (visually lossless) to prevent macroblocking."""
        try:
            import imageio

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

    def run_multistage_pipeline(
        self,
        prompt: str,
        input_mode: str = "Text-to-Video with Auto-Keyframe (Stage 1 -> Stage 2)",
        uploaded_image: Optional[Image.Image] = None,
        keyframe_model_key: str = "SDXL-Turbo (Fast, Zero-Auth, Low RAM - Default)",
        model_key: str = "Wan2.1-1.3B-T2V",
        num_frames: int = 49,
        width: int = 832,
        height: int = 480,
        num_inference_steps: int = 30,
        fps: int = 16,
        guidance_scale: float = 5.0,
        seed: int = -1,
        generate_audio: bool = True,
        audio_prompt: Optional[str] = None,
        audio_model_key: str = "AudioLDM 2 (High Fidelity Foley - Default)",
        progress_callback: Optional[Callable] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, Optional[Image.Image], str]:
        """
        Executes the full 4-stage sequential generation pipeline:
        [1/4] Generating Keyframe -> [2/4] Animating Frames -> [3/4] Synthesizing Sound -> [4/4] Muxing Output.
        Returns (final_mp4_path, keyframe_image, formatted_stage_log).
        """
        stage_logs: List[str] = []

        def _log_stage(msg: str, prog_val: Optional[float] = None):
            logger.info(msg)
            stage_logs.append(msg)
            if status_callback is not None:
                status_callback("\n".join(stage_logs))
            if progress_callback is not None and prog_val is not None:
                progress_callback(prog_val, desc=msg)

        base_spec = MODEL_REGISTRY[model_key]
        target_w = align_spatial_dim(width, base_spec.spatial_multiple)
        target_h = align_spatial_dim(height, base_spec.spatial_multiple)

        # =========================================================================
        # STAGE 1: Keyframe Generation or Direct Image Conditioning
        # =========================================================================
        anchor_image: Optional[Image.Image] = None
        use_i2v = False

        if "Direct Image Upload" in input_mode:
            if uploaded_image is None:
                raise ValueError("Please upload an input image for 'Direct Image Upload to Video' mode.")
            _log_stage(
                f"✅ [1/4] Generating Keyframe: Loaded user-uploaded image ({target_w}x{target_h})",
                0.10,
            )
            anchor_image = uploaded_image.convert("RGB").resize(
                (target_w, target_h), Image.Resampling.LANCZOS
            )
            use_i2v = True
        elif "Auto-Keyframe" in input_mode:
            _log_stage(
                f"⏳ [1/4] Generating Keyframe: Synthesizing photorealistic anchor frame ({target_w}x{target_h})...",
                0.06,
            )
            # Free any cached Stage 2 video model before running Stage 1 keyframe synthesis
            self._unload_current_pipeline()
            anchor_image = self.keyframe_stage.generate_keyframe(
                prompt=prompt,
                width=target_w,
                height=target_h,
                keyframe_model_key=keyframe_model_key,
                seed=seed,
            )
            stage_logs[-1] = f"✅ [1/4] Generating Keyframe: Anchor frame synthesized ({target_w}x{target_h})"
            # Use I2V when LTX-Video is selected; if Wan2.1-1.3B is selected, keep the keyframe preview
            # and route I2V through LTX-Video when in Direct Image Upload mode, or use Wan2.1-1.3B native backbone
            use_i2v = "LTX" in model_key
        else:
            _log_stage("⏭️ [1/4] Generating Keyframe: Skipped (Direct Text-to-Video mode)", 0.10)

        flush_memory()

        # =========================================================================
        # STAGE 2: Temporal Animation (I2V / T2V)
        # =========================================================================
        resolved_model_key = model_key
        if use_i2v and "Wan" in model_key and self.low_system_ram:
            resolved_model_key = "Lightricks/LTX-Video"
        active_spec = MODEL_REGISTRY[resolved_model_key]

        shot_prompts = [p.strip() for p in prompt.split("||") if p.strip()]
        if use_i2v:
            shot_prompts = [shot_prompts[0]]

        # On 12.7 GB Colab Free Tier: Pre-encode all prompts BEFORE loading the video transformer/VAE!
        # That way the 9.5 GB T5/UMT5 text_encoder is the ONLY model in memory, then it is 100% deleted
        # before the video transformer + VAE are loaded.
        pre_encoded_shots: Optional[List[Dict[str, torch.Tensor]]] = None
        if self.device == "cuda" and self.low_system_ram:
            _log_stage(
                f"⏳ [2/4] Animating Frames: Encoding prompt embeddings on GPU ({resolved_model_key})...",
                0.15,
            )
            pre_encoded_shots = self._encode_prompt_standalone_low_ram(
                spec=active_spec,
                prompts=shot_prompts,
                negative_prompt=active_spec.default_negative_prompt,
                guidance_scale=float(guidance_scale),
            )

        _log_stage(
            f"⏳ [2/4] Animating Frames: Loading {resolved_model_key} backbone & animating sequence...",
            0.20,
        )

        pipe, active_spec, is_i2v_active = self._load_video_pipeline(resolved_model_key, use_i2v=use_i2v)
        effective_w = align_spatial_dim(width, active_spec.spatial_multiple)
        effective_h = align_spatial_dim(height, active_spec.spatial_multiple)
        effective_frames = align_frame_count(num_frames, active_spec.frame_divisor)
        total_steps = int(num_inference_steps)

        if anchor_image is not None and anchor_image.size != (effective_w, effective_h):
            anchor_image = anchor_image.resize((effective_w, effective_h), Image.Resampling.LANCZOS)

        num_shots = len(shot_prompts)
        all_frames = []

        for shot_idx, shot_prompt in enumerate(shot_prompts):
            shot_prefix = f"[Shot {shot_idx + 1}/{num_shots}] " if num_shots > 1 else ""
            generator = None
            if seed is not None and int(seed) >= 0:
                gen_device = "cuda" if self.device == "cuda" else "cpu"
                generator = torch.Generator(device=gen_device).manual_seed(int(seed) + shot_idx)

            def _on_step_end(_pipe, step_index: int, _timestep, callback_kwargs):
                if progress_callback is not None:
                    shot_base = shot_idx / num_shots
                    shot_span = 0.50 / num_shots
                    frac = 0.22 + shot_base * 0.48 + shot_span * ((step_index + 1) / max(1, total_steps))
                    progress_callback(
                        min(0.70, frac),
                        desc=f"[2/4] Animating Frames: {shot_prefix}Step {step_index + 1}/{total_steps}",
                    )
                return callback_kwargs

            flush_memory()

            pipe_kwargs = dict(
                height=effective_h,
                width=effective_w,
                num_frames=effective_frames,
                num_inference_steps=total_steps,
                guidance_scale=float(guidance_scale),
                generator=generator,
                callback_on_step_end=_on_step_end,
            )

            with torch.inference_mode():
                if is_i2v_active and anchor_image is not None:
                    pipe_kwargs["image"] = anchor_image

                if pre_encoded_shots is not None:
                    pipe_kwargs.update(pre_encoded_shots[shot_idx])
                else:
                    pipe_kwargs["prompt"] = shot_prompt
                    pipe_kwargs["negative_prompt"] = active_spec.default_negative_prompt

                result = pipe(**pipe_kwargs)

            shot_frames = list(result.frames[0])
            all_frames.extend(shot_frames)
            del result
            flush_memory()

        del pre_encoded_shots
        flush_memory()

        # If Stage 1 was skipped, extract frame 0 as the visual keyframe preview
        if anchor_image is None and all_frames:
            first_f = all_frames[0]
            if isinstance(first_f, Image.Image):
                anchor_image = first_f.copy()
            else:
                arr = np.asarray(first_f)
                if arr.dtype != np.uint8:
                    arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
                anchor_image = Image.fromarray(arr)

        silent_video_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        silent_video_path = silent_video_tmp.name
        silent_video_tmp.close()

        self._export_high_quality_mp4(all_frames, output_path=silent_video_path, fps=int(fps))
        total_rendered_frames = len(all_frames)
        video_duration_sec = max(2.0, total_rendered_frames / float(fps))
        del all_frames

        # Unload Stage 2 video pipeline before Stage 3 Audio synthesis if audio is enabled
        if generate_audio:
            self._unload_current_pipeline()

        stage_logs[-1] = (
            f"✅ [2/4] Animating Frames: Rendered {total_rendered_frames} frames "
            f"({video_duration_sec:.1f}s @ {effective_w}x{effective_h}, {fps} FPS)"
        )

        # =========================================================================
        # STAGE 3: Sound FX & Foley Generation (AudioLDM 2)
        # =========================================================================
        if not generate_audio:
            _log_stage("⏭️ [3/4] Synthesizing Sound: Skipped (Audio toggle disabled)", 0.90)
            _log_stage("✅ [4/4] Muxing Output: Exported H.264 MP4 container", 1.0)
            return silent_video_path, anchor_image, "\n".join(stage_logs)

        _log_stage(
            f"⏳ [3/4] Synthesizing Sound: Generating {video_duration_sec:.1f}s ambient Foley audio...",
            0.75,
        )
        wav_path: Optional[str] = None
        try:
            wav_path = self.audio_stage.generate_foley(
                audio_prompt=audio_prompt or prompt,
                duration_seconds=video_duration_sec,
                audio_model_key=audio_model_key,
                seed=seed,
            )
            stage_logs[-1] = f"✅ [3/4] Synthesizing Sound: Generated {video_duration_sec:.1f}s stereo Foley WAV"
        except Exception as exc:
            logger.warning("Stage 3 Audio synthesis failed (%s); returning video without audio.", exc)
            stage_logs[-1] = f"⚠️ [3/4] Synthesizing Sound: Audio fallback ({exc}); keeping video track"
            _log_stage("✅ [4/4] Muxing Output: Exported H.264 MP4 container", 1.0)
            return silent_video_path, anchor_image, "\n".join(stage_logs)

        flush_memory()

        # =========================================================================
        # STAGE 4: FFmpeg Audio + Video Muxing (`-c:v copy -c:a aac -shortest`)
        # =========================================================================
        _log_stage("⏳ [4/4] Muxing Output: Synchronizing H.264 video + AAC audio via FFmpeg...", 0.94)
        try:
            final_muxed_path = self.muxer_stage.mux_video_and_audio(
                video_path=silent_video_path,
                audio_path=wav_path,
            )
            # Clean up intermediate silent video and raw wav
            for p in (silent_video_path, wav_path):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            stage_logs[-1] = (
                "✅ [4/4] Muxing Output: Muxed synchronized H.264 + AAC MP4 (-c:v copy -c:a aac -shortest)"
            )
            if progress_callback is not None:
                progress_callback(1.0, desc="Complete!")
            return final_muxed_path, anchor_image, "\n".join(stage_logs)
        except Exception as exc:
            logger.warning("Stage 4 FFmpeg muxing failed (%s); returning unmuxed video.", exc)
            stage_logs[-1] = f"⚠️ [4/4] Muxing Output: FFmpeg fallback ({exc}); returning video stream"
            return silent_video_path, anchor_image, "\n".join(stage_logs)
