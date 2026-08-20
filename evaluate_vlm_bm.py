#!/usr/bin/env python3
"""
Video Re-ID Benchmark Evaluation Script for VLMs (Memory-Optimized)

Evaluates open-source Vision-Language Models on the Video Person
Re-Identification benchmark with aggressive memory optimization.
Processes the entire benchmark in a single job - no batching needed.

Supported Models:
1. Qwen2-VL (Alibaba)
2. Qwen2.5-VL / Qwen3-VL (Alibaba)
3. Ovis (AIDC-AI)
4. Ovis2.5 (AIDC-AI)
5. LLaVA-NeXT-Video (Microsoft/LLaVA)
6. InternVL2 (OpenGVLab)
7. Video-LLaVA (PKU)

Usage:
    python evaluate_vlm_bm.py \
        --benchmark benchmark_questions.json \
        --video_dir /path/to/videos \
        --models qwen2-vl \
        --output_dir results \
        --num_frames 8 \
        --max_pixels 256 256
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Disable torch.compile for Python 3.14+ compatibility
if sys.version_info >= (3, 14):
    import torch
    _original_compile = torch.compile
    def _no_compile(model, *args, **kwargs):
        return model
    torch.compile = _no_compile

import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for SLURM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# Memory Utilities
# =============================================================================

def clear_gpu_memory():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_info():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")


def get_quantization_config(quantize_4bit: bool = False):
    """Return BitsAndBytes quantization config if requested."""
    if not quantize_4bit:
        return None
    try:
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    except ImportError:
        print("WARNING: bitsandbytes not installed. Skipping 4-bit quantization.")
        print("Install with: pip install bitsandbytes")
        return None


# =============================================================================
# Base VLM Interface
# =============================================================================

class BaseVLM(ABC):
    """Abstract base class for Video Language Models."""

    def __init__(self, model_name: str, device: str = "cuda",
                 num_frames: int = 8, max_pixels: Tuple[int, int] = (256, 256),
                 quantize_4bit: bool = False):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.num_frames = num_frames
        self.max_pixels = max_pixels[0] * max_pixels[1]
        self.max_pixels_h = max_pixels[0]
        self.max_pixels_w = max_pixels[1]
        self.quantize_4bit = quantize_4bit
        self.quant_config = get_quantization_config(quantize_4bit)

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def inference(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        pass

    def prepare_video(self, video_path: str) -> Any:
        return video_path

    def inference_with_cache(self, video_cache: Any, question: str, options) -> str:
        return self.inference(video_cache, question, options)

    def cleanup_inference(self):
        clear_gpu_memory()

    def inference_text_only(self, question: str, options) -> str:
        """Run inference with only the question text, no video/image input.
        Subclasses should override for model-specific text-only chat."""
        raise NotImplementedError(
            f"{self.model_name} does not implement text-only inference"
        )

    def batch_inference_text_only(self, questions, options_list):
        """Default implementation: loop over inference_text_only.
        Subclasses should override with a real batched path where safe
        (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Gemma3 — set padding_side='left').
        Models with single-sample-only APIs (Ovis, InternVL3) should keep the loop."""
        return [self.inference_text_only(q, o) for q, o in zip(questions, options_list)]

    def describe_images(self, pil_images, prompt: str, max_new_tokens: int = 256) -> str:
        """Free-text description of a list of images + a prompt (NOT an MCQ letter).
        Overridden by VLM classes used in the dossier pipeline (e.g. InternVL3)."""
        raise NotImplementedError(
            f"{self.model_name} does not implement describe_images")

    def format_mcq_prompt(self, question: str, options) -> str:
        if isinstance(options, list):
            letters = [chr(ord('A') + i) for i in range(len(options))]
            options = dict(zip(letters, options))

        prompt = f"{question}\n\nOptions:\n"
        sorted_keys = sorted(options.keys())
        for key in sorted_keys:
            prompt += f"{key}. {options[key]}\n"
        letter_list = ", ".join(sorted_keys[:-1]) + f", or {sorted_keys[-1]}"
        prompt += f"\nAnswer with only the letter ({letter_list}) of the correct option."
        return prompt

    def extract_answer(self, response: str, num_options: int = 5) -> str:
        response = response.strip().upper()

        max_letter = chr(ord('A') + num_options - 1)
        letter_range = f"A-{max_letter}"
        valid_letters = [chr(ord('A') + i) for i in range(num_options)]

        if response in valid_letters:
            return response

        patterns = [
            rf'^([{letter_range}])\.',
            rf'^([{letter_range}])\)',
            rf'^\(([{letter_range}])\)',
            rf'^answer[:\s]*([{letter_range}])',
            rf'^the answer is[:\s]*([{letter_range}])',
            rf'^([{letter_range}])\s*[-:]',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        match = re.search(rf'\b([{letter_range}])\b', response)
        if match:
            return match.group(1).upper()

        return "INVALID"


# =============================================================================
# VLM Implementations (Memory-Optimized)
# =============================================================================

class Qwen2VL(BaseVLM):
    """Qwen2-VL model implementation."""

    def __init__(self, model_size: str = "7B", device: str = "cuda",
                 num_frames: int = 8, max_pixels: Tuple[int, int] = (256, 256),
                 quantize_4bit: bool = False):
        super().__init__(f"Qwen2-VL-{model_size}", device, num_frames, max_pixels, quantize_4bit)
        self.model_size = model_size
        self.model_id = f"Qwen/Qwen2-VL-{model_size}-Instruct"

    def load_model(self):
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info

            load_kwargs = dict(
                dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

            try:
                import flash_attn
                load_kwargs["attn_implementation"] = "flash_attention_2"
                print(f"  Using flash_attention_2")
            except ImportError:
                print(f"  flash_attention_2 not available, using default")

            if self.quant_config:
                load_kwargs["quantization_config"] = self.quant_config

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id, **load_kwargs
            )
            self.model.eval()

            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                max_pixels=self.max_pixels,
                min_pixels=28 * 28
            )
            self.process_vision_info = process_vision_info
            print(f"Loaded {self.model_name} (dtype=bf16, frames={self.num_frames}, "
                  f"max_px={self.max_pixels_h}x{self.max_pixels_w}, "
                  f"4bit={self.quantize_4bit})")
            get_gpu_memory_info()
        except ImportError as e:
            print(f"Error loading {self.model_name}: {e}")
            print("Install with: pip install transformers qwen-vl-utils")
            raise

    def prepare_video(self, video_path: str) -> Any:
        from PIL import Image
        import decord
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        pil_frames = [Image.fromarray(f) for f in frames]
        del vr, frames

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": pil_frames,
                        "max_pixels": self.max_pixels,
                        "min_pixels": 28 * 28,
                        "nframes": self.num_frames,
                    },
                    {"type": "text", "text": "placeholder"}
                ]
            }
        ]
        image_inputs, video_inputs = self.process_vision_info(messages)
        return (image_inputs, video_inputs)

    def inference_with_cache(self, video_cache: Any, question: str, options) -> str:
        prompt = self.format_mcq_prompt(question, options)
        image_inputs, video_inputs = video_cache

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": "cached",
                        "max_pixels": self.max_pixels,
                        "nframes": self.num_frames,
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        del inputs, generated_ids, generated_ids_trimmed
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))

    def inference(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        video_cache = self.prepare_video(video_path)
        result = self.inference_with_cache(video_cache, question, options)
        del video_cache
        self.cleanup_inference()
        return result

    def inference_text_only(self, question: str, options) -> str:
        prompt = self.format_mcq_prompt(question, options)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=8, do_sample=False
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        del inputs, generated_ids, generated_ids_trimmed
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))

    def batch_inference_text_only(self, questions, options_list):
        # Qwen tokenizer must be left-padded for batched generate, or shorter
        # sequences get garbled outputs.
        self.processor.tokenizer.padding_side = "left"
        prompts = [self.format_mcq_prompt(q, o) for q, o in zip(questions, options_list)]
        texts = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": p}]}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in prompts
        ]
        inputs = self.processor(text=texts, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=8, do_sample=False
            )
        prompt_lens = inputs.input_ids.shape[1]
        trimmed = generated_ids[:, prompt_lens:]
        responses = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        out = [self.extract_answer(r, num_options=len(o)) for r, o in zip(responses, options_list)]
        del inputs, generated_ids, trimmed
        self.cleanup_inference()
        return out


class Qwen3VL(BaseVLM):
    """Qwen2.5-VL model implementation (also referred to as Qwen3-VL)."""

    def __init__(self, model_size: str = "7B", device: str = "cuda",
                 num_frames: int = 8, max_pixels: Tuple[int, int] = (256, 256),
                 quantize_4bit: bool = False):
        super().__init__(f"Qwen2.5-VL-{model_size}", device, num_frames, max_pixels, quantize_4bit)
        self.model_size = model_size
        self.model_id = f"Qwen/Qwen2.5-VL-{model_size}-Instruct"

    def load_model(self):
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info

            load_kwargs = dict(
                dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

            try:
                import flash_attn
                load_kwargs["attn_implementation"] = "flash_attention_2"
                print(f"  Using flash_attention_2")
            except ImportError:
                print(f"  flash_attention_2 not available, using default")

            if self.quant_config:
                load_kwargs["quantization_config"] = self.quant_config

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id, **load_kwargs
            )
            self.model.eval()

            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                max_pixels=self.max_pixels,
                min_pixels=28 * 28
            )
            self.process_vision_info = process_vision_info
            print(f"Loaded {self.model_name} (dtype=bf16, frames={self.num_frames}, "
                  f"max_px={self.max_pixels_h}x{self.max_pixels_w}, "
                  f"4bit={self.quantize_4bit})")
            get_gpu_memory_info()
        except ImportError as e:
            print(f"Error loading {self.model_name}: {e}")
            print("Install with: pip install transformers qwen-vl-utils")
            raise

    def prepare_video(self, video_path: str) -> Any:
        from PIL import Image
        import decord
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        pil_frames = [Image.fromarray(f) for f in frames]
        del vr, frames

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": pil_frames,
                        "max_pixels": self.max_pixels,
                        "min_pixels": 28 * 28,
                        "nframes": self.num_frames,
                    },
                    {"type": "text", "text": "placeholder"}
                ]
            }
        ]
        image_inputs, video_inputs = self.process_vision_info(messages)
        return (image_inputs, video_inputs)

    def inference_with_cache(self, video_cache: Any, question: str, options) -> str:
        prompt = self.format_mcq_prompt(question, options)
        image_inputs, video_inputs = video_cache

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": "cached",
                        "max_pixels": self.max_pixels,
                        "nframes": self.num_frames,
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        del inputs, generated_ids, generated_ids_trimmed
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))

    def inference(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        video_cache = self.prepare_video(video_path)
        result = self.inference_with_cache(video_cache, question, options)
        del video_cache
        self.cleanup_inference()
        return result

    def inference_text_only(self, question: str, options) -> str:
        prompt = self.format_mcq_prompt(question, options)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=8, do_sample=False
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        del inputs, generated_ids, generated_ids_trimmed
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))

    def batch_inference_text_only(self, questions, options_list):
        self.processor.tokenizer.padding_side = "left"
        prompts = [self.format_mcq_prompt(q, o) for q, o in zip(questions, options_list)]
        texts = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": p}]}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in prompts
        ]
        inputs = self.processor(text=texts, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=8, do_sample=False
            )
        prompt_lens = inputs.input_ids.shape[1]
        trimmed = generated_ids[:, prompt_lens:]
        responses = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        out = [self.extract_answer(r, num_options=len(o)) for r, o in zip(responses, options_list)]
        del inputs, generated_ids, trimmed
        self.cleanup_inference()
        return out


class Ovis(BaseVLM):
    """Ovis model implementation."""

    def __init__(self, model_version: str = "1.6", model_name_full: str = "Ovis1.6-Gemma2-9B",
                 device: str = "cuda", num_frames: int = 8,
                 max_pixels: Tuple[int, int] = (256, 256),
                 quantize_4bit: bool = False):
        super().__init__(f"Ovis-{model_version}", device, num_frames, max_pixels, quantize_4bit)
        self.model_version = model_version
        self.model_id = f"AIDC-AI/{model_name_full}"

    def load_model(self):
        try:
            from transformers import AutoModelForCausalLM

            load_kwargs = dict(
                dtype=torch.bfloat16,
                device_map="auto",
                multimodal_max_length=8192,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            if self.quant_config:
                load_kwargs["quantization_config"] = self.quant_config

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, **load_kwargs
            )
            self.model.eval()

            self.text_tokenizer = getattr(self.model, 'text_tokenizer', None) or self.model.get_text_tokenizer()
            self.visual_tokenizer = getattr(self.model, 'visual_tokenizer', None) or self.model.get_visual_tokenizer()
            print(f"Loaded {self.model_name} (dtype=bf16, 4bit={self.quantize_4bit})")
            get_gpu_memory_info()
        except ImportError as e:
            print(f"Error loading {self.model_name}: {e}")
            raise

    def prepare_video(self, video_path: str) -> Any:
        from PIL import Image
        import decord
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        images = [Image.fromarray(f) for f in frames]
        del vr, frames
        return images

    def inference_with_cache(self, video_cache: Any, question: str, options) -> str:
        prompt = self.format_mcq_prompt(question, options)
        images = video_cache
        image_placeholders = "".join(["<image>\n"] * len(images))
        query = f"{image_placeholders}{prompt}"

        # Ovis1.6 API: preprocess_inputs returns (prompt, input_ids, pixel_values)
        _, input_ids, pixel_values = self.model.preprocess_inputs(query, images)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        pixel_values = [pixel_values.to(
            dtype=self.visual_tokenizer.dtype,
            device=self.visual_tokenizer.device
        )]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True,
            )[0]

        response = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
        del input_ids, attention_mask, pixel_values, output_ids
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))

    def inference(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        video_cache = self.prepare_video(video_path)
        result = self.inference_with_cache(video_cache, question, options)
        del video_cache
        self.cleanup_inference()
        return result

    def inference_text_only(self, question: str, options) -> str:
        prompt = self.format_mcq_prompt(question, options)
        # No image placeholders, no images — text-only query
        _, input_ids, pixel_values = self.model.preprocess_inputs(prompt, [])
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        # pixel_values may be None or empty for text-only; wrap in list for generate API
        if pixel_values is not None:
            pixel_values = [pixel_values.to(
                dtype=self.visual_tokenizer.dtype,
                device=self.visual_tokenizer.device
            )]
        else:
            pixel_values = [torch.zeros(0, dtype=self.visual_tokenizer.dtype,
                                        device=self.visual_tokenizer.device)]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=8,
                do_sample=False,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True,
            )[0]

        response = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
        del input_ids, attention_mask, pixel_values, output_ids
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))


class Ovis25(BaseVLM):
    """Ovis2.5 model implementation."""

    def __init__(self, model_name_full: str = "Ovis2.5-14B",
                 device: str = "cuda", num_frames: int = 8,
                 max_pixels: Tuple[int, int] = (256, 256),
                 quantize_4bit: bool = False):
        super().__init__("Ovis2.5", device, num_frames, max_pixels, quantize_4bit)
        self.model_id = f"AIDC-AI/{model_name_full}"

    def load_model(self):
        try:
            from transformers import AutoModelForCausalLM
            load_kwargs = dict(
                dtype=torch.bfloat16, device_map="auto",
                multimodal_max_length=8192, trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            if self.quant_config:
                load_kwargs["quantization_config"] = self.quant_config

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
            self.model.eval()
            self.text_tokenizer = getattr(self.model, 'text_tokenizer', None) or self.model.get_text_tokenizer()
            self.visual_tokenizer = getattr(self.model, 'visual_tokenizer', None) or self.model.get_visual_tokenizer()
            print(f"Loaded {self.model_name} (dtype=bf16, 4bit={self.quantize_4bit})")
            get_gpu_memory_info()
        except ImportError as e:
            print(f"Error loading {self.model_name}: {e}")
            raise

    def prepare_video(self, video_path: str) -> Any:
        from PIL import Image
        import decord
        vr = decord.VideoReader(video_path)
        indices = np.linspace(0, len(vr) - 1, self.num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        images = [Image.fromarray(f) for f in frames]
        # REID_IMG_SIZE (if set) downsizes frames so high frame counts fit memory;
        # unset = native resolution (unchanged default behavior).
        _sz = os.environ.get("REID_IMG_SIZE")
        if _sz:
            _sz = int(_sz)
            images = [im.resize((_sz, _sz)) for im in images]
        del vr, frames
        return images

    def inference_with_cache(self, video_cache: Any, question: str, options) -> str:
        prompt = self.format_mcq_prompt(question, options)
        images = video_cache

        # Ovis2.5 new API: build messages list with image + text content
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = input_ids.to(self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        if grid_thws is not None:
            grid_thws = grid_thws.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                enable_thinking=False,
                max_new_tokens=128,
            )

        response = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        del outputs, input_ids, pixel_values, grid_thws
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))

    def inference(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        video_cache = self.prepare_video(video_path)
        result = self.inference_with_cache(video_cache, question, options)
        del video_cache
        self.cleanup_inference()
        return result

    def inference_text_only(self, question: str, options) -> str:
        prompt = self.format_mcq_prompt(question, options)
        # Text-only: no image content in messages
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = input_ids.to(self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        if grid_thws is not None:
            grid_thws = grid_thws.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                enable_thinking=False,
                max_new_tokens=8,
                do_sample=False,
            )

        response = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        del outputs, input_ids, pixel_values, grid_thws
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))


class LLaVANextVideo(BaseVLM):
    """LLaVA-NeXT-Video model implementation."""

    def __init__(self, model_size: str = "7B", device: str = "cuda",
                 num_frames: int = 8, max_pixels: Tuple[int, int] = (256, 256),
                 quantize_4bit: bool = False):
        super().__init__(f"LLaVA-NeXT-Video-{model_size}", device, num_frames, max_pixels, quantize_4bit)
        self.model_size = model_size
        self.model_id = f"llava-hf/LLaVA-NeXT-Video-{model_size}-hf"

    def load_model(self):
        try:
            from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
            load_kwargs = dict(dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
            try:
                import flash_attn
                load_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                pass
            if self.quant_config:
                load_kwargs["quantization_config"] = self.quant_config

            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(self.model_id, **load_kwargs)
            self.model.eval()
            self.processor = LlavaNextVideoProcessor.from_pretrained(self.model_id)
            print(f"Loaded {self.model_name} (dtype=bf16, 4bit={self.quantize_4bit})")
            get_gpu_memory_info()
        except ImportError as e:
            print(f"Error loading {self.model_name}: {e}")
            raise

    def prepare_video(self, video_path: str) -> Any:
        import decord
        vr = decord.VideoReader(video_path)
        indices = np.linspace(0, len(vr) - 1, self.num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        del vr
        return frames

    def inference_with_cache(self, video_cache: Any, question: str, options) -> str:
        prompt = self.format_mcq_prompt(question, options)
        conversation = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=text, videos=[video_cache], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        del inputs, generated_ids, trimmed
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))

    def inference(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        video_cache = self.prepare_video(video_path)
        result = self.inference_with_cache(video_cache, question, options)
        del video_cache
        self.cleanup_inference()
        return result

    def inference_text_only(self, question: str, options) -> str:
        # Text-only via the standard multimodal pipeline using a single black frame.
        # Bypassing the LM directly would lose chat-template formatting and
        # artifactually depress accuracy (false-negative bias signal).
        from PIL import Image
        prompt = self.format_mcq_prompt(question, options)
        dummy_frame = np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
        conversation = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=text, videos=[dummy_frame], return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=8, do_sample=False
            )
        trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        del inputs, generated_ids, trimmed
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))


class InternVL2(BaseVLM):
    """InternVL2 model implementation."""

    def __init__(self, model_size: str = "8B", device: str = "cuda",
                 num_frames: int = 8, max_pixels: Tuple[int, int] = (256, 256),
                 quantize_4bit: bool = False):
        super().__init__(f"InternVL2-{model_size}", device, num_frames, max_pixels, quantize_4bit)
        self.model_size = model_size
        self.model_id = f"OpenGVLab/InternVL2-{model_size}"

    def load_model(self):
        try:
            from transformers import AutoModel, AutoTokenizer
            load_kwargs = dict(dtype=torch.bfloat16, device_map="auto",
                             trust_remote_code=True, low_cpu_mem_usage=True)
            try:
                import flash_attn
                load_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                pass
            if self.quant_config:
                load_kwargs["quantization_config"] = self.quant_config

            self.model = AutoModel.from_pretrained(self.model_id, **load_kwargs)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            print(f"Loaded {self.model_name} (dtype=bf16, 4bit={self.quantize_4bit})")
            get_gpu_memory_info()
        except ImportError as e:
            print(f"Error loading {self.model_name}: {e}")
            raise

    def prepare_video(self, video_path: str) -> Any:
        from PIL import Image
        import decord
        vr = decord.VideoReader(video_path)
        indices = np.linspace(0, len(vr) - 1, self.num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        images = [Image.fromarray(f) for f in frames]
        del vr, frames
        return images

    def inference_with_cache(self, video_cache: Any, question: str, options) -> str:
        prompt = self.format_mcq_prompt(question, options)
        images = video_cache
        frame_ph = "".join([f"Frame {i+1}: <image>\n" for i in range(len(images))])
        full_prompt = f"{frame_ph}\n{prompt}"

        pixel_values = None
        try:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((448, 448)), transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pixel_values = torch.stack([transform(img) for img in images]).to(
                dtype=torch.bfloat16, device=self.device)
        except Exception:
            pass

        gen_config = dict(max_new_tokens=128, do_sample=False)
        with torch.no_grad():
            response = self.model.chat(self.tokenizer, pixel_values, full_prompt, gen_config)

        if pixel_values is not None:
            del pixel_values
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))

    def inference(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        video_cache = self.prepare_video(video_path)
        result = self.inference_with_cache(video_cache, question, options)
        del video_cache
        self.cleanup_inference()
        return result

    def inference_text_only(self, question: str, options) -> str:
        # Text-only via standard multimodal pipeline with a single black image,
        # preserving chat template / <image> token formatting.
        from PIL import Image
        prompt = self.format_mcq_prompt(question, options)
        dummy_image = Image.new("RGB", (448, 448), color=(0, 0, 0))
        try:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((448, 448)), transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            pixel_values = transform(dummy_image).unsqueeze(0).to(
                dtype=torch.bfloat16, device=self.device
            )
        except Exception:
            pixel_values = None
        full_prompt = f"<image>\n{prompt}"
        gen_config = dict(max_new_tokens=8, do_sample=False)
        with torch.no_grad():
            response = self.model.chat(self.tokenizer, pixel_values, full_prompt, gen_config)
        if pixel_values is not None:
            del pixel_values
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))


class VideoLLaVA(BaseVLM):
    """Video-LLaVA model implementation."""

    def __init__(self, device: str = "cuda", num_frames: int = 8,
                 max_pixels: Tuple[int, int] = (256, 256), quantize_4bit: bool = False):
        super().__init__("Video-LLaVA", device, num_frames, max_pixels, quantize_4bit)
        self.model_id = "LanguageBind/Video-LLaVA-7B-hf"

    def load_model(self):
        try:
            from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
            load_kwargs = dict(dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
            if self.quant_config:
                load_kwargs["quantization_config"] = self.quant_config
            self.model = VideoLlavaForConditionalGeneration.from_pretrained(self.model_id, **load_kwargs)
            self.model.eval()
            self.processor = VideoLlavaProcessor.from_pretrained(self.model_id)
            print(f"Loaded {self.model_name} (dtype=bf16, 4bit={self.quantize_4bit})")
            get_gpu_memory_info()
        except ImportError as e:
            print(f"Error loading {self.model_name}: {e}")
            raise

    def prepare_video(self, video_path: str) -> Any:
        import decord
        vr = decord.VideoReader(video_path)
        indices = np.linspace(0, len(vr) - 1, min(self.num_frames, 8), dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        del vr
        return frames

    def inference_with_cache(self, video_cache: Any, question: str, options) -> str:
        prompt = self.format_mcq_prompt(question, options)
        full_prompt = f"USER: <video>\n{prompt}\nASSISTANT:"
        inputs = self.processor(text=full_prompt, videos=[video_cache], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        del inputs, generated_ids, trimmed
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))

    def inference(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        video_cache = self.prepare_video(video_path)
        result = self.inference_with_cache(video_cache, question, options)
        del video_cache
        self.cleanup_inference()
        return result

    def inference_text_only(self, question: str, options) -> str:
        # Video-LLaVA expects 8 frames; feed black ones to retain template format.
        prompt = self.format_mcq_prompt(question, options)
        dummy_frames = np.zeros((8, 224, 224, 3), dtype=np.uint8)
        full_prompt = f"USER: <video>\n{prompt}\nASSISTANT:"
        inputs = self.processor(text=full_prompt, videos=[dummy_frames], return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=8, do_sample=False
            )
        trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        del inputs, generated_ids, trimmed
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))


class Qwen3VLReal(BaseVLM):
    """Real Qwen3-VL (not Qwen2.5-VL aliased). Requires transformers>=4.50."""

    def __init__(self, model_size: str = "8B", device: str = "cuda",
                 num_frames: int = 8, max_pixels: Tuple[int, int] = (256, 256),
                 quantize_4bit: bool = False):
        super().__init__(f"Qwen3-VL-{model_size}", device, num_frames, max_pixels, quantize_4bit)
        self.model_size = model_size
        self.model_id = f"Qwen/Qwen3-VL-{model_size}-Instruct"

    def load_model(self):
        from transformers import AutoProcessor
        load_kwargs = dict(dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
        if self.quant_config:
            load_kwargs["quantization_config"] = self.quant_config
        try:
            from transformers import Qwen3VLForConditionalGeneration
            model_cls = Qwen3VLForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForCausalLM
            model_cls = AutoModelForCausalLM
            load_kwargs["trust_remote_code"] = True
        self.model = model_cls.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, max_pixels=self.max_pixels, min_pixels=28 * 28
            )
        except Exception:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        print(f"Loaded {self.model_name} (dtype=bf16, 4bit={self.quantize_4bit}, "
              f"max_pixels={self.max_pixels})")
        get_gpu_memory_info()

    def prepare_video(self, video_path: str) -> Any:
        from PIL import Image
        import decord
        vr = decord.VideoReader(video_path)
        indices = np.linspace(0, len(vr) - 1, self.num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        # IMPORTANT: resize each frame down to the model's max_pixels budget
        # before tokenization. Without this, default processor budget can blow
        # up to >5000 visual tokens per frame and kill throughput.
        target_h, target_w = self.max_pixels_h, self.max_pixels_w
        pil_frames = [Image.fromarray(f).resize((target_w, target_h)) for f in frames]
        del vr, frames
        return pil_frames

    def inference_with_cache(self, video_cache: Any, question: str, options) -> str:
        prompt = self.format_mcq_prompt(question, options)
        pil_frames = video_cache
        # Build a proper VIDEO content block (not 8 separate images) so the
        # Qwen3-VL processor treats this as a temporal sequence and uses the
        # cheaper video token packing path.
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": pil_frames,
                 "max_pixels": self.max_pixels, "nframes": self.num_frames},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Try qwen_vl_utils if available - same path as Qwen2.5-VL
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(self.device)
        except Exception:
            # Fallback: pass frames directly as videos kwarg
            inputs = self.processor(
                text=[text], videos=[pil_frames],
                padding=True, return_tensors="pt",
            ).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, generated_ids)]
        response = self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0]
        del inputs, generated_ids, trimmed
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))

    def inference(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        video_cache = self.prepare_video(video_path)
        result = self.inference_with_cache(video_cache, question, options)
        del video_cache
        self.cleanup_inference()
        return result

    def inference_text_only(self, question: str, options) -> str:
        prompt = self.format_mcq_prompt(question, options)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, generated_ids)]
        response = self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0]
        del inputs, generated_ids, trimmed
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))

    def batch_inference_text_only(self, questions, options_list):
        self.processor.tokenizer.padding_side = "left"
        prompts = [self.format_mcq_prompt(q, o) for q, o in zip(questions, options_list)]
        texts = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": p}]}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in prompts
        ]
        inputs = self.processor(text=texts, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        prompt_lens = inputs.input_ids.shape[1]
        trimmed = generated_ids[:, prompt_lens:]
        responses = self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
        out = [self.extract_answer(r, num_options=len(o)) for r, o in zip(responses, options_list)]
        del inputs, generated_ids, trimmed
        self.cleanup_inference()
        return out


class InternVL3(BaseVLM):
    """InternVL3 (2B / 8B / 14B). Text-only via standard multimodal pipeline with dummy image."""

    def __init__(self, model_size: str = "8B", device: str = "cuda",
                 num_frames: int = 8, max_pixels: Tuple[int, int] = (256, 256),
                 quantize_4bit: bool = False):
        super().__init__(f"InternVL3-{model_size}", device, num_frames, max_pixels, quantize_4bit)
        self.model_size = model_size
        self.model_id = f"OpenGVLab/InternVL3-{model_size}"

    def load_model(self):
        from transformers import AutoModel, AutoTokenizer
        load_kwargs = dict(dtype=torch.bfloat16, device_map="auto",
                           trust_remote_code=True, low_cpu_mem_usage=True)
        if self.quant_config:
            load_kwargs["quantization_config"] = self.quant_config
        self.model = AutoModel.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        print(f"Loaded {self.model_name} (dtype=bf16, 4bit={self.quantize_4bit})")
        get_gpu_memory_info()

    def prepare_video(self, video_path: str) -> Any:
        from PIL import Image
        import decord
        vr = decord.VideoReader(video_path)
        indices = np.linspace(0, len(vr) - 1, self.num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        images = [Image.fromarray(f) for f in frames]
        del vr, frames
        return images

    def inference_with_cache(self, video_cache: Any, question: str, options) -> str:
        prompt = self.format_mcq_prompt(question, options)
        images = video_cache
        frame_ph = "".join([f"Frame {i+1}: <image>\n" for i in range(len(images))])
        full_prompt = f"{frame_ph}\n{prompt}"

        pixel_values = None
        try:
            from torchvision import transforms
            # REID_IMG_SIZE lets the frame-count ablation lower per-frame resolution
            # at high frame counts to fit memory (default 448 = unchanged behavior).
            _sz = int(os.environ.get("REID_IMG_SIZE", 448))
            transform = transforms.Compose([
                transforms.Resize((_sz, _sz)), transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            pixel_values = torch.stack([transform(img) for img in images]).to(
                dtype=torch.bfloat16, device=self.device
            )
        except Exception:
            pass

        gen_config = dict(max_new_tokens=8, do_sample=False)
        with torch.no_grad():
            response = self.model.chat(self.tokenizer, pixel_values, full_prompt, gen_config)
        if pixel_values is not None:
            del pixel_values
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))

    def inference(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        video_cache = self.prepare_video(video_path)
        result = self.inference_with_cache(video_cache, question, options)
        del video_cache
        self.cleanup_inference()
        return result

    def describe_images(self, pil_images, prompt: str, max_new_tokens: int = 256) -> str:
        """Free-text description of a LIST of images (e.g. one identity's crops) + prompt.
        Returns the raw generated text (NOT an MCQ letter). Used by the dossier pipeline
        to write one per-identity slot at a time."""
        from torchvision import transforms
        _sz = int(os.environ.get("REID_IMG_SIZE", 448))
        transform = transforms.Compose([
            transforms.Resize((_sz, _sz)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        pixel_values = torch.stack([transform(img.convert("RGB")) for img in pil_images]).to(
            dtype=torch.bfloat16, device=self.device)
        ph = "".join([f"Image-{i+1}: <image>\n" for i in range(len(pil_images))])
        full_prompt = f"{ph}\n{prompt}"
        gen_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
        with torch.no_grad():
            response = self.model.chat(self.tokenizer, pixel_values, full_prompt, gen_config)
        del pixel_values
        self.cleanup_inference()
        return response.strip()

    def inference_text_only(self, question: str, options) -> str:
        from PIL import Image
        prompt = self.format_mcq_prompt(question, options)
        dummy_image = Image.new("RGB", (448, 448), color=(0, 0, 0))
        try:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((448, 448)), transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            pixel_values = transform(dummy_image).unsqueeze(0).to(
                dtype=torch.bfloat16, device=self.device
            )
        except Exception:
            pixel_values = None
        full_prompt = f"<image>\n{prompt}"
        gen_config = dict(max_new_tokens=8, do_sample=False)
        with torch.no_grad():
            response = self.model.chat(self.tokenizer, pixel_values, full_prompt, gen_config)
        if pixel_values is not None:
            del pixel_values
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))


class Gemma3VL(BaseVLM):
    """Gemma 3 multimodal (4B-it, 12B-it). Text-only via chat template."""

    def __init__(self, model_size: str = "4b", device: str = "cuda",
                 num_frames: int = 8, max_pixels: Tuple[int, int] = (256, 256),
                 quantize_4bit: bool = False):
        super().__init__(f"Gemma3-{model_size}", device, num_frames, max_pixels, quantize_4bit)
        self.model_size = model_size
        self.model_id = f"google/gemma-3-{model_size}-it"

    def load_model(self):
        from transformers import AutoProcessor
        load_kwargs = dict(dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
        if self.quant_config:
            load_kwargs["quantization_config"] = self.quant_config
        try:
            from transformers import Gemma3ForConditionalGeneration
            self.model = Gemma3ForConditionalGeneration.from_pretrained(self.model_id, **load_kwargs)
        except ImportError:
            from transformers import AutoModelForCausalLM
            load_kwargs["trust_remote_code"] = True
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        print(f"Loaded {self.model_name} (dtype=bf16, 4bit={self.quantize_4bit})")
        get_gpu_memory_info()

    def prepare_video(self, video_path: str) -> Any:
        from PIL import Image
        import decord
        vr = decord.VideoReader(video_path)
        # Gemma3 SigLIP encoder produces 256 tokens per 896x896 image. Eight
        # frames at that res = 2048 tokens plus text - feasible but slow.
        # We resize to 448x448 to halve token count and use only the
        # configured num_frames.
        indices = np.linspace(0, len(vr) - 1, self.num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        pil_frames = [Image.fromarray(f).resize((448, 448)) for f in frames]
        del vr, frames
        return pil_frames

    def inference_with_cache(self, video_cache: Any, question: str, options) -> str:
        prompt = self.format_mcq_prompt(question, options)
        pil_frames = video_cache
        # Gemma3 supports interleaved images; treat each video frame as a separate image.
        content = [{"type": "image", "image": img} for img in pil_frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        trimmed = generated_ids[:, prompt_len:]
        response = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        del inputs, generated_ids, trimmed
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))

    def inference(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        video_cache = self.prepare_video(video_path)
        result = self.inference_with_cache(video_cache, question, options)
        del video_cache
        self.cleanup_inference()
        return result

    def inference_text_only(self, question: str, options) -> str:
        prompt = self.format_mcq_prompt(question, options)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        trimmed = generated_ids[:, prompt_len:]
        response = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        del inputs, generated_ids, trimmed
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))

    def batch_inference_text_only(self, questions, options_list):
        # Gemma3 processor uses an inner tokenizer; force left-pad for batched gen.
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None:
            tok.padding_side = "left"
        prompts = [self.format_mcq_prompt(q, o) for q, o in zip(questions, options_list)]
        all_msgs = [[{"role": "user", "content": [{"type": "text", "text": p}]}] for p in prompts]
        inputs = self.processor.apply_chat_template(
            all_msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True,
        ).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        trimmed = generated_ids[:, prompt_len:]
        responses = self.processor.batch_decode(trimmed, skip_special_tokens=True)
        out = [self.extract_answer(r, num_options=len(o)) for r, o in zip(responses, options_list)]
        del inputs, generated_ids, trimmed
        self.cleanup_inference()
        return out


class VideoChatFlash(BaseVLM):
    """OpenGVLab VideoChat-Flash (Qwen2.5 backbone, 2B or 7B at res448).

    API note: model.chat() expects a video PATH (it decodes internally with
    decord). Our prepare_video therefore returns the path as the cache; no
    pre-decoding on our side. For text-only mode we point chat() at a small
    pre-generated black MP4 (data/dummy_black.mp4)."""

    DEFAULT_DUMMY_MP4 = "/home/ab260989/gen-reid/data/dummy_black.mp4"

    def __init__(self, model_size: str = "2B", device: str = "cuda",
                 num_frames: int = 8, max_pixels: Tuple[int, int] = (448, 448),
                 quantize_4bit: bool = False):
        super().__init__(f"VideoChat-Flash-{model_size}", device, num_frames, max_pixels, quantize_4bit)
        self.model_size = model_size
        # HF family naming is non-uniform: 2B uses Qwen2.5 backbone, 7B uses Qwen2.
        if model_size.upper() == "2B":
            self.model_id = "OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448"
        elif model_size.upper() == "7B":
            self.model_id = "OpenGVLab/VideoChat-Flash-Qwen2-7B_res448"
        else:
            self.model_id = f"OpenGVLab/VideoChat-Flash-Qwen2-{model_size}_res448"

    def load_model(self):
        # NOTE: VideoChat-Flash custom modeling code is pinned to transformers ~4.40
        # cache APIs. Run this class in the `videochat-flash` conda env
        # (transformers==4.40.1, with a stub flash_attn package), not in `reid`.
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        load_kwargs = dict(trust_remote_code=True)
        if self.quant_config:
            load_kwargs["quantization_config"] = self.quant_config
        self.model = AutoModel.from_pretrained(self.model_id, **load_kwargs)
        self.model = self.model.to(torch.bfloat16).to(self.device)
        self.model.eval()
        print(f"Loaded {self.model_name} (dtype=bf16, 4bit={self.quantize_4bit})")
        get_gpu_memory_info()

    def prepare_video(self, video_path: str) -> Any:
        # VideoChat-Flash decodes the mp4 itself; we pass the path through.
        return video_path

    def _chat(self, video_path: str, prompt: str) -> str:
        with torch.no_grad():
            out = self.model.chat(
                video_path=video_path,
                tokenizer=self.tokenizer,
                user_prompt=prompt,
                return_history=False,
                max_num_frames=self.num_frames,
                generation_config=dict(
                    do_sample=False, max_new_tokens=8, num_beams=1,
                ),
            )
        return out if isinstance(out, str) else out[0]

    def inference_with_cache(self, video_cache: Any, question: str, options) -> str:
        prompt = self.format_mcq_prompt(question, options)
        response = self._chat(video_cache, prompt)
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))

    def inference(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        return self.inference_with_cache(video_path, question, options)

    def inference_text_only(self, question: str, options) -> str:
        prompt = self.format_mcq_prompt(question, options)
        response = self._chat(self.DEFAULT_DUMMY_MP4, prompt)
        self.cleanup_inference()
        return self.extract_answer(response, num_options=len(options))


# =============================================================================
# Model Registry
# =============================================================================

def create_model(model_name, device="cuda", num_frames=8,
                 max_pixels=(256, 256), quantize_4bit=False):
    model_map = {
        "qwen2-vl": lambda: Qwen2VL("7B", device, num_frames, max_pixels, quantize_4bit),
        "qwen2-vl-2b": lambda: Qwen2VL("2B", device, num_frames, max_pixels, quantize_4bit),
        "qwen2-vl-72b": lambda: Qwen2VL("72B", device, num_frames, max_pixels, quantize_4bit),
        "qwen3-vl": lambda: Qwen3VL("7B", device, num_frames, max_pixels, quantize_4bit),
        "qwen2.5-vl": lambda: Qwen3VL("7B", device, num_frames, max_pixels, quantize_4bit),
        "qwen2.5-vl-3b": lambda: Qwen3VL("3B", device, num_frames, max_pixels, quantize_4bit),
        "qwen2.5-vl-7b": lambda: Qwen3VL("7B", device, num_frames, max_pixels, quantize_4bit),
        "qwen2.5-vl-72b": lambda: Qwen3VL("72B", device, num_frames, max_pixels, quantize_4bit),
        "qwen3-vl-real-2b": lambda: Qwen3VLReal("2B", device, num_frames, max_pixels, quantize_4bit),
        "qwen3-vl-real-4b": lambda: Qwen3VLReal("4B", device, num_frames, max_pixels, quantize_4bit),
        "qwen3-vl-real-8b": lambda: Qwen3VLReal("8B", device, num_frames, max_pixels, quantize_4bit),
        "ovis": lambda: Ovis("1.6", "Ovis1.6-Gemma2-9B", device, num_frames, max_pixels, quantize_4bit),
        "ovis2.5": lambda: Ovis25("Ovis2.5-9B", device, num_frames, max_pixels, quantize_4bit),
        "ovis2.5-2b": lambda: Ovis25("Ovis2.5-2B", device, num_frames, max_pixels, quantize_4bit),
        "ovis2.5-9b": lambda: Ovis25("Ovis2.5-9B", device, num_frames, max_pixels, quantize_4bit),
        "llava-next-video": lambda: LLaVANextVideo("7B", device, num_frames, max_pixels, quantize_4bit),
        "llava-next-video-34b": lambda: LLaVANextVideo("34B", device, num_frames, max_pixels, quantize_4bit),
        "internvl2": lambda: InternVL2("8B", device, num_frames, max_pixels, quantize_4bit),
        "internvl2-26b": lambda: InternVL2("26B", device, num_frames, max_pixels, quantize_4bit),
        "internvl3-2b": lambda: InternVL3("2B", device, num_frames, max_pixels, quantize_4bit),
        "internvl3-8b": lambda: InternVL3("8B", device, num_frames, max_pixels, quantize_4bit),
        "internvl3-14b": lambda: InternVL3("14B", device, num_frames, max_pixels, quantize_4bit),
        "gemma3-4b": lambda: Gemma3VL("4b", device, num_frames, max_pixels, quantize_4bit),
        "gemma3-12b": lambda: Gemma3VL("12b", device, num_frames, max_pixels, quantize_4bit),
        "video-llava": lambda: VideoLLaVA(device, num_frames, max_pixels, quantize_4bit),
        "videochat-flash-2b": lambda: VideoChatFlash("2B", device, num_frames, max_pixels, quantize_4bit),
        "videochat-flash-7b": lambda: VideoChatFlash("7B", device, num_frames, max_pixels, quantize_4bit),
    }
    key = model_name.lower().strip()
    if key not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {', '.join(sorted(model_map.keys()))}")
    return model_map[key]()


# =============================================================================
# Evaluation Engine (Single Job - No Batching)
# =============================================================================

class BenchmarkEvaluator:
    """Evaluates VLMs on the full benchmark in a single job."""

    def __init__(self, benchmark_path, video_dir, output_dir):
        self.benchmark_path = benchmark_path
        self.video_dir = video_dir
        # Auto-detect: if video_dir contains a 'videos' subdirectory, use it
        videos_subdir = os.path.join(video_dir, "videos")
        if os.path.isdir(videos_subdir):
            print(f"  Auto-detected 'videos' subdirectory, using: {videos_subdir}")
            self.video_dir = videos_subdir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(benchmark_path, 'r') as f:
            raw = json.load(f)

        # Normalize benchmark to a list of dicts regardless of JSON structure
        if isinstance(raw, list):
            self.benchmark = raw
        elif isinstance(raw, dict):
            # Check for common wrapper keys first (e.g. {"data": [...]})
            for key in ("videos", "data", "questions", "items", "benchmark"):
                if key in raw and isinstance(raw[key], list):
                    self.benchmark = raw[key]
                    break
            else:
                # Dict keyed by video filename -> questions
                self.benchmark = []
                for video_key, value in raw.items():
                    if isinstance(value, dict):
                        value.setdefault("video", video_key)
                        self.benchmark.append(value)
                    elif isinstance(value, list):
                        # Each value is a list of questions for that video
                        self.benchmark.append({
                            "video": video_key,
                            "questions": value,
                        })
                    else:
                        print(f"  WARNING: Skipping unexpected value type for key '{video_key}': {type(value)}")
        else:
            raise ValueError(f"Unexpected benchmark JSON type: {type(raw)}")

        print(f"Loaded benchmark: {len(self.benchmark)} items from {benchmark_path}")

    def evaluate_model(self, model, max_samples=None):
        results = []
        correct = total = errors = 0
        category_results = defaultdict(lambda: {"correct": 0, "total": 0})

        items = self.benchmark[:max_samples] if max_samples else self.benchmark
        num_items = len(items)
        print(f"\nEvaluating {model.model_name} on {num_items} items...")

        # Debug: show first item keys so we know the JSON structure
        if items:
            print(f"  Benchmark item keys: {list(items[0].keys())}")
            print(f"  First item preview: {str(items[0])[:200]}")

        get_gpu_memory_info()
        eval_start = time.time()

        for idx, item in enumerate(items):
            # Build video path: try video/video_path keys first, then video_id + extension
            video_file = item.get("video", item.get("video_path", ""))
            if not video_file:
                vid_id = item.get("video_id", "")
                if vid_id:
                    for ext in (".mp4", ".avi", ".mkv", ".mov", ".webm", ""):
                        candidate = os.path.join(self.video_dir, f"{vid_id}{ext}")
                        if os.path.isfile(candidate):
                            video_file = f"{vid_id}{ext}"
                            break
                    else:
                        video_file = f"{vid_id}.mp4"  # default guess

            if video_file and os.path.isabs(video_file):
                video_path = video_file
            else:
                video_path = os.path.join(self.video_dir, video_file)

            if not os.path.isfile(video_path):
                if idx < 3:
                    print(f"  WARNING: Video not found: {video_path}")
                    if idx == 0 and os.path.isdir(self.video_dir):
                        sample = os.listdir(self.video_dir)[:5]
                        print(f"    Sample files in video_dir: {sample}")
                errors += 1
                continue

            questions = item.get("questions", [item])

            try:
                video_cache = model.prepare_video(video_path)
            except Exception as e:
                print(f"  ERROR preparing video {video_path}: {e}")
                errors += 1
                clear_gpu_memory()
                continue

            for q_idx, q in enumerate(questions):
                question_text = q.get("question", q.get("question_text", q.get("text", "")))
                options = q.get("options", {})
                correct_answer = q.get("answer", q.get("correct_answer", ""))
                meta = q.get("metadata", {})
                category = q.get("category", q.get("type", meta.get("capability", "unknown")))

                try:
                    t0 = time.time()
                    predicted = model.inference_with_cache(video_cache, question_text, options)
                    elapsed = time.time() - t0

                    is_correct = predicted.upper() == correct_answer.upper()
                    if is_correct:
                        correct += 1
                    total += 1
                    category_results[category]["total"] += 1
                    if is_correct:
                        category_results[category]["correct"] += 1

                    results.append({
                        "video": video_path, "question": question_text,
                        "correct_answer": correct_answer, "predicted": predicted,
                        "is_correct": is_correct, "category": category,
                        "time_seconds": elapsed,
                    })
                except Exception as e:
                    print(f"  ERROR on Q{q_idx} for {video_path}: {e}")
                    errors += 1
                    results.append({
                        "video": video_path, "question": question_text,
                        "correct_answer": correct_answer, "predicted": "ERROR",
                        "is_correct": False, "category": category, "error": str(e),
                    })

            del video_cache
            clear_gpu_memory()

            if (idx + 1) % 5 == 0 or idx == num_items - 1 or idx < 3:
                acc = correct / total * 100 if total > 0 else 0
                elapsed_total = time.time() - eval_start
                rate = (idx + 1) / elapsed_total * 60 if elapsed_total > 0 else 0
                eta = (num_items - idx - 1) / rate if rate > 0 else 0
                print(f"  [{idx+1}/{num_items}] Acc: {acc:.1f}% ({correct}/{total}) | "
                      f"Errors: {errors} | {rate:.1f} vid/min | ETA: {eta:.0f} min")
                get_gpu_memory_info()

        total_time = time.time() - eval_start
        accuracy = correct / total * 100 if total > 0 else 0

        category_accuracies = {}
        for cat, counts in category_results.items():
            cat_acc = counts["correct"] / counts["total"] * 100 if counts["total"] > 0 else 0
            category_accuracies[cat] = {"accuracy": cat_acc, "correct": counts["correct"], "total": counts["total"]}

        metrics = {
            "model": model.model_name, "overall_accuracy": accuracy,
            "correct": correct, "total": total, "errors": errors,
            "total_time_minutes": total_time / 60,
            "videos_per_minute": num_items / total_time * 60 if total_time > 0 else 0,
            "category_accuracies": category_accuracies,
            "config": {"num_frames": model.num_frames,
                       "max_pixels": f"{model.max_pixels_h}x{model.max_pixels_w}",
                       "quantize_4bit": model.quantize_4bit},
            "timestamp": datetime.now().isoformat(),
        }

        print(f"\n{'='*60}")
        print(f"RESULTS: {model.model_name}")
        print(f"  Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"  Errors: {errors}")
        print(f"  Total Time: {total_time/60:.1f} minutes")
        for cat, cm in sorted(category_accuracies.items()):
            print(f"  {cat}: {cm['accuracy']:.1f}% ({cm['correct']}/{cm['total']})")
        print(f"{'='*60}")

        return {"metrics": metrics, "results": results}

    def save_results(self, model_name, eval_output):
        safe_name = model_name.replace("/", "_").replace(" ", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        for suffix, data in [("metrics", eval_output["metrics"]), ("results", eval_output["results"])]:
            path = os.path.join(self.output_dir, f"{safe_name}_{suffix}_{ts}.json")
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  Saved: {path}")

        csv_path = os.path.join(self.output_dir, f"{safe_name}_results_{ts}.csv")
        pd.DataFrame(eval_output["results"]).to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")


# =============================================================================
# Visualization
# =============================================================================

def plot_results(all_metrics, output_dir):
    if not all_metrics:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    models = [m["model"] for m in all_metrics]
    accuracies = [m["overall_accuracy"] for m in all_metrics]
    bars = ax.bar(models, accuracies, color=sns.color_palette("husl", len(models)))
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("VLM Benchmark: Overall Accuracy")
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()+1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_accuracy.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate VLMs on Video Re-ID Benchmark")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSON")
    parser.add_argument("--video_dir", required=True, help="Path to video directory")
    parser.add_argument("--models", nargs="+", required=True, help="Models to evaluate")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples (debug)")
    parser.add_argument("--num_frames", type=int, default=8, help="Frames per video")
    parser.add_argument("--max_pixels", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))
    parser.add_argument("--quantize_4bit", action="store_true", help="4-bit quantization")
    args = parser.parse_args()

    print("=" * 60)
    print("Video Re-ID Benchmark Evaluation")
    print("=" * 60)
    print(f"Benchmark:  {args.benchmark}")
    print(f"Video dir:  {args.video_dir}")
    print(f"Models:     {args.models}")
    print(f"Frames:     {args.num_frames}")
    print(f"Max pixels: {args.max_pixels[0]}x{args.max_pixels[1]}")
    print(f"4-bit:      {args.quantize_4bit}")
    if torch.cuda.is_available():
        print(f"GPU:        {torch.cuda.get_device_name(0)}")
        tmem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU VRAM:   {tmem:.1f} GB")
        if tmem <= 34 and not args.quantize_4bit:
            print("\n*** WARNING: <=34GB VRAM. Consider --quantize_4bit ***")
    print("=" * 60)

    evaluator = BenchmarkEvaluator(args.benchmark, args.video_dir, args.output_dir)
    all_metrics = []

    for model_name in args.models:
        print(f"\n{'='*60}\nLoading: {model_name}\n{'='*60}")
        model = None
        try:
            model = create_model(model_name, args.device, args.num_frames,
                                tuple(args.max_pixels), args.quantize_4bit)
            model.load_model()
            eval_output = evaluator.evaluate_model(model, args.max_samples)
            evaluator.save_results(model_name, eval_output)
            all_metrics.append(eval_output["metrics"])
        except Exception as e:
            print(f"FAILED: {model_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if model is not None:
                del model
            clear_gpu_memory()

    if all_metrics:
        plot_results(all_metrics, args.output_dir)
        summary_path = os.path.join(args.output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nSummary: {summary_path}")
    print("Done!")


if __name__ == "__main__":
    main()