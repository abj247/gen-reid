#!/usr/bin/env python3
"""
Video Re-ID Benchmark Evaluation Script for VLMs

This script evaluates open-source Vision-Language Models on the Video Person
Re-Identification benchmark. It supports multiple VLMs and provides comprehensive
metrics and visualizations for analysis.

Supported Models:
1. Qwen2-VL (Alibaba)
2. Ovis (AIDC-AI)
3. LLaVA-NeXT-Video (Microsoft/LLaVA)
4. InternVL2 (OpenGVLab)
5. Video-LLaVA (PKU)

Usage:
    python evaluate_vlm_benchmark.py \
        --benchmark benchmark_questions.json \
        --video_dir /path/to/videos \
        --models qwen2-vl ovis \
        --output_dir results
"""

import argparse
import json
import os
import re
import sys
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
        """No-op replacement for torch.compile on Python 3.14+"""
        return model
    torch.compile = _no_compile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# Base VLM Interface
# =============================================================================

class BaseVLM(ABC):
    """Abstract base class for Video Language Models."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self):
        """Load the model and processor."""
        pass

    @abstractmethod
    def inference(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        """
        Run inference on a video with a question.

        Args:
            video_path: Path to the video file
            question: The question text
            options: Dictionary of options (A, B, C, D)

        Returns:
            The predicted answer (A, B, C, or D)
        """
        pass

    def format_mcq_prompt(self, question: str, options: Dict[str, str]) -> str:
        """Format the MCQ prompt for the model."""
        prompt = f"{question}\n\nOptions:\n"
        for key, value in sorted(options.items()):
            prompt += f"{key}. {value}\n"
        prompt += "\nAnswer with only the letter (A, B, C, or D) of the correct option."
        return prompt

    def extract_answer(self, response: str) -> str:
        """Extract the answer letter from model response."""
        response = response.strip().upper()

        # Direct match
        if response in ['A', 'B', 'C', 'D']:
            return response

        # Look for patterns like "A.", "A)", "(A)", "Answer: A"
        patterns = [
            r'^([ABCD])\.',
            r'^([ABCD])\)',
            r'^\(([ABCD])\)',
            r'^answer[:\s]*([ABCD])',
            r'^the answer is[:\s]*([ABCD])',
            r'^([ABCD])\s*[-:]',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # Search anywhere in response
        match = re.search(r'\b([ABCD])\b', response)
        if match:
            return match.group(1).upper()

        return "INVALID"


# =============================================================================
# VLM Implementations
# =============================================================================

class Qwen2VL(BaseVLM):
    """Qwen2-VL model implementation."""

    def __init__(self, model_size: str = "7B", device: str = "cuda"):
        super().__init__(f"Qwen2-VL-{model_size}", device)
        self.model_size = model_size
        self.model_id = f"Qwen/Qwen2-VL-{model_size}-Instruct"

    def load_model(self):
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.process_vision_info = process_vision_info
            print(f"Loaded {self.model_name}")
        except ImportError as e:
            print(f"Error loading {self.model_name}: {e}")
            print("Install with: pip install transformers qwen-vl-utils")
            raise

    def inference(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        prompt = self.format_mcq_prompt(question, options)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "max_pixels": 360*420, "fps": 1.0},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

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

        return self.extract_answer(response)


class OvisVLM(BaseVLM):
    """Ovis model implementation."""

    def __init__(self, model_size: str = "1.6", device: str = "cuda"):
        super().__init__(f"Ovis-{model_size}", device)
        self.model_id = f"AIDC-AI/Ovis{model_size}-Gemma2-9B"

    def _disable_torch_compile(self, model):
        """Recursively disable torch.compile on all submodules for Python 3.14+ compatibility."""
        # Disable on main model
        if hasattr(model, 'generation_config') and model.generation_config is not None:
            model.generation_config.compile_config = None
        
        # Disable on LLM if present
        if hasattr(model, 'llm') and model.llm is not None:
            if hasattr(model.llm, 'generation_config') and model.llm.generation_config is not None:
                model.llm.generation_config.compile_config = None
        
        # Disable on get_llm() if method exists
        if hasattr(model, 'get_llm'):
            try:
                llm = model.get_llm()
                if hasattr(llm, 'generation_config') and llm.generation_config is not None:
                    llm.generation_config.compile_config = None
            except:
                pass

    def load_model(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                multimodal_max_length=8192,
                trust_remote_code=True,
                attn_implementation="eager"
            ).to(self.device)
            self.text_tokenizer = self.model.get_text_tokenizer()
            self.visual_tokenizer = self.model.get_visual_tokenizer()
            
            # Disable torch.compile for Python 3.14+ compatibility
            self._disable_torch_compile(self.model)
            
            print(f"Loaded {self.model_name}")
        except ImportError as e:
            print(f"Error loading {self.model_name}: {e}")
            raise

    def inference(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        import torch
        from PIL import Image
        import cv2

        prompt = self.format_mcq_prompt(question, options)

        # Extract frames from video
        frames = self._extract_frames(video_path, num_frames=8)

        # Format conversation with images
        query = "<image>\n" * len(frames) + prompt

        prompt_text, input_ids, pixel_values = self.model.preprocess_inputs(
            query, frames
        )
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        # pixel_values is already a concatenated tensor from preprocess_inputs, wrap in list for generate
        if pixel_values is not None:
            pixel_values = [pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.device)]

        with torch.inference_mode():
            gen_kwargs = {
                "max_new_tokens": 128,
                "do_sample": False,
                "pad_token_id": self.text_tokenizer.pad_token_id,
                "eos_token_id": self.text_tokenizer.eos_token_id,
            }
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **gen_kwargs
            )[0]

        response = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return self.extract_answer(response)

    def _extract_frames(self, video_path: str, num_frames: int = 8) -> List:
        from PIL import Image
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")
        
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"Could not extract any frames from: {video_path}")
        
        return frames


class LLaVANeXTVideo(BaseVLM):
    """LLaVA-NeXT-Video model implementation."""

    def __init__(self, model_size: str = "7B", device: str = "cuda"):
        super().__init__(f"LLaVA-NeXT-Video-{model_size}", device)
        self.model_id = f"llava-hf/LLaVA-NeXT-Video-{model_size}-hf"

    def load_model(self):
        try:
            import torch
            from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

            self.processor = LlavaNextVideoProcessor.from_pretrained(self.model_id)
            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print(f"Loaded {self.model_name}")
        except ImportError as e:
            print(f"Error loading {self.model_name}: {e}")
            print("Install with: pip install transformers")
            raise

    def inference(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        import av
        import numpy as np

        prompt = self.format_mcq_prompt(question, options)

        # Read video frames
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames // 8).tolist()[:8]

        frames = []
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                frames.append(frame.to_ndarray(format="rgb24"))

        video_array = np.stack(frames)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(
            text=prompt_text,
            videos=video_array,
            return_tensors="pt"
        ).to(self.device)

        output = self.model.generate(**inputs, max_new_tokens=128)
        response = self.processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        return self.extract_answer(response)


class InternVL2(BaseVLM):
    """InternVL2 model implementation.
    
    NOTE: InternVL2 has compatibility issues with transformers >= 4.45.
    Consider using transformers 4.44.x or skipping this model.
    """

    def __init__(self, model_size: str = "8B", device: str = "cuda"):
        super().__init__(f"InternVL2-{model_size}", device)
        self.model_id = f"OpenGVLab/InternVL2-{model_size}"

    def load_model(self):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            import transformers
            
            # Check transformers version for compatibility warning
            version = transformers.__version__
            major, minor = map(int, version.split('.')[:2])
            if major >= 4 and minor >= 45:
                print(f"WARNING: InternVL2 may have compatibility issues with transformers {version}")
                print("         Consider downgrading to transformers 4.44.x or skipping this model.")

            self.model = AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto"
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            print(f"Loaded {self.model_name}")
        except ImportError as e:
            print(f"Error loading {self.model_name}: {e}")
            raise

    def inference(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        import torch
        from PIL import Image
        import cv2

        prompt = self.format_mcq_prompt(question, options)

        # Extract frames
        frames = self._extract_frames(video_path, num_frames=8)
        pixel_values = self._preprocess_frames(frames)

        # Build query with frame placeholders
        frame_placeholders = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(frames))])
        query = frame_placeholders + prompt

        generation_config = {
            "max_new_tokens": 128,
            "do_sample": False,
            "use_cache": False,  # Disable caching to avoid compatibility issues with newer transformers
        }

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            query,
            generation_config
        )

        return self.extract_answer(response)

    def _extract_frames(self, video_path: str, num_frames: int = 8) -> List:
        from PIL import Image
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")
        
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"Could not extract any frames from: {video_path}")
        
        return frames

    def _preprocess_frames(self, frames: List) -> Any:
        import torch
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        if not frames:
            raise ValueError("No frames to preprocess")

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        pixel_values = [transform(frame) for frame in frames]
        pixel_values = torch.stack(pixel_values).to(torch.bfloat16).to(self.device)
        return pixel_values


class VideoLLaVA(BaseVLM):
    """Video-LLaVA model implementation."""

    def __init__(self, device: str = "cuda"):
        super().__init__("Video-LLaVA", device)
        self.model_id = "LanguageBind/Video-LLaVA-7B-hf"

    def load_model(self):
        try:
            import torch
            from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

            self.model = VideoLlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.processor = VideoLlavaProcessor.from_pretrained(self.model_id)
            print(f"Loaded {self.model_name}")
        except ImportError as e:
            print(f"Error loading {self.model_name}: {e}")
            raise

    def inference(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        import av
        import numpy as np

        prompt = self.format_mcq_prompt(question, options)

        # Read video
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames // 8).tolist()[:8]

        frames = []
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                frames.append(frame.to_ndarray(format="rgb24"))

        clip = np.stack(frames)

        full_prompt = f"USER: <video>\n{prompt}\nASSISTANT:"

        inputs = self.processor(
            text=full_prompt,
            videos=clip,
            return_tensors="pt"
        ).to(self.device)

        output = self.model.generate(**inputs, max_new_tokens=128)
        response = self.processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        return self.extract_answer(response)


# =============================================================================
# Mock VLM for Testing (returns random answers)
# =============================================================================

class MockVLM(BaseVLM):
    """Mock VLM for testing without actual models."""

    def __init__(self, model_name: str = "MockVLM", device: str = "cpu"):
        super().__init__(model_name, device)

    def load_model(self):
        print(f"Loaded {self.model_name} (Mock Mode)")

    def inference(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        import random
        # Simulate some processing time
        return random.choice(['A', 'B', 'C', 'D'])


# =============================================================================
# Model Factory
# =============================================================================

def get_model(model_name: str, device: str = "cuda", mock: bool = False) -> BaseVLM:
    """Factory function to get the appropriate model."""

    if mock:
        return MockVLM(f"Mock-{model_name}", device)

    model_map = {
        "qwen2-vl": lambda: Qwen2VL("7B", device),
        "qwen2-vl-2b": lambda: Qwen2VL("2B", device),
        "qwen2-vl-72b": lambda: Qwen2VL("72B", device),
        "ovis": lambda: OvisVLM("1.6", device),
        "llava-next-video": lambda: LLaVANeXTVideo("7B", device),
        "llava-next-video-34b": lambda: LLaVANeXTVideo("34B", device),
        "internvl2": lambda: InternVL2("8B", device),
        "internvl2-26b": lambda: InternVL2("26B", device),
        "video-llava": lambda: VideoLLaVA(device),
    }

    model_key = model_name.lower()
    if model_key not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")

    return model_map[model_key]()


# =============================================================================
# Evaluation Engine
# =============================================================================

class BenchmarkEvaluator:
    """Evaluates VLMs on the Re-ID benchmark."""

    def __init__(self, benchmark_path: str, video_dir: str, output_dir: str):
        self.benchmark_path = benchmark_path
        self.video_dir = video_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load benchmark
        with open(benchmark_path) as f:
            self.benchmark = json.load(f)

        self.results = {}

    def get_video_path(self, video_id: str) -> str:
        """Get the full path to a video file."""
        # Try common video extensions
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            path = os.path.join(self.video_dir, f"{video_id}{ext}")
            if os.path.exists(path):
                return path
        # Return default path even if doesn't exist (for mock mode)
        return os.path.join(self.video_dir, f"{video_id}.mp4")

    def validate_prerequisites(self, models: List[BaseVLM], mock: bool = False) -> Tuple[bool, Dict]:
        """
        Pre-flight validation to check all prerequisites before evaluation.

        Checks:
        1. Video directory exists
        2. All benchmark videos are accessible
        3. Model availability (in non-mock mode)

        Returns:
            Tuple of (is_valid, validation_report)
        """
        print("\n" + "=" * 60)
        print("PRE-FLIGHT VALIDATION")
        print("=" * 60)

        validation_report = {
            "video_dir_exists": False,
            "videos": {},
            "models": {},
            "total_videos": 0,
            "videos_found": 0,
            "videos_missing": 0,
            "total_questions": 0,
            "questions_with_video": 0,
        }

        is_valid = True

        # 1. Check video directory
        print(f"\n[1/3] Checking video directory: {self.video_dir}")
        if os.path.isdir(self.video_dir):
            validation_report["video_dir_exists"] = True
            print(f"  [OK] Directory exists")
        else:
            validation_report["video_dir_exists"] = False
            print(f"  [X] Directory does not exist!")
            is_valid = False

        # 2. Check all videos in benchmark
        print(f"\n[2/3] Checking video files...")
        total_questions = 0
        questions_with_video = 0

        for video in self.benchmark["videos"]:
            video_id = video["video_id"]
            num_questions = len(video["questions"])
            total_questions += num_questions
            validation_report["total_videos"] += 1

            # Check if video exists
            video_path = None
            for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                test_path = os.path.join(self.video_dir, f"{video_id}{ext}")
                if os.path.exists(test_path):
                    video_path = test_path
                    break

            if video_path:
                file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
                validation_report["videos"][video_id] = {
                    "found": True,
                    "path": video_path,
                    "size_mb": round(file_size, 2),
                    "questions": num_questions
                }
                validation_report["videos_found"] += 1
                questions_with_video += num_questions
                print(f"  [OK] {video_id} ({num_questions} questions) - {file_size:.1f} MB")
            else:
                validation_report["videos"][video_id] = {
                    "found": False,
                    "path": None,
                    "questions": num_questions
                }
                validation_report["videos_missing"] += 1
                print(f"  [X] {video_id} ({num_questions} questions) - NOT FOUND")
                if not mock:
                    is_valid = False

        validation_report["total_questions"] = total_questions
        validation_report["questions_with_video"] = questions_with_video

        # 3. Check models
        print(f"\n[3/3] Checking models...")
        for model in models:
            model_name = model.model_name
            if mock or "Mock" in model_name:
                validation_report["models"][model_name] = {"available": True, "mock": True}
                print(f"  [OK] {model_name} (Mock Mode)")
            else:
                # For real models, we just note they'll be loaded later
                validation_report["models"][model_name] = {"available": True, "mock": False}
                print(f"  ? {model_name} (Will attempt to load)")

        # Summary
        print("\n" + "-" * 60)
        print("VALIDATION SUMMARY")
        print("-" * 60)
        print(f"  Video Directory: {'[OK]' if validation_report['video_dir_exists'] else '[X] MISSING'}")
        print(f"  Videos: {validation_report['videos_found']}/{validation_report['total_videos']} found")
        print(f"  Questions with video: {questions_with_video}/{total_questions}")
        print(f"  Models to evaluate: {len(models)}")

        if validation_report["videos_missing"] > 0 and not mock:
            print(f"\n  [!] WARNING: {validation_report['videos_missing']} video(s) missing!")
            print(f"    Models will answer {total_questions - questions_with_video} questions without video context.")

        if is_valid:
            print("\n  [OK] All pre-flight checks passed!")
        else:
            print("\n  [X] Some pre-flight checks failed!")

        print("=" * 60 + "\n")

        return is_valid, validation_report

    def evaluate_model(self, model: BaseVLM) -> Dict:
        """Evaluate a single model on the benchmark."""
        print(f"\n{'='*60}")
        print(f"Evaluating: {model.model_name}")
        print(f"{'='*60}")

        model.load_model()

        results = {
            "model_name": model.model_name,
            "predictions": [],
            "timestamp": datetime.now().isoformat()
        }

        total_questions = sum(len(v["questions"]) for v in self.benchmark["videos"])
        current = 0

        for video in self.benchmark["videos"]:
            video_id = video["video_id"]
            video_path = self.get_video_path(video_id)

            print(f"\nProcessing video: {video_id}")

            for question in video["questions"]:
                current += 1

                try:
                    predicted = model.inference(
                        video_path,
                        question["question_text"],
                        question["options"]
                    )
                except Exception as e:
                    import traceback
                    print(f"  Error on Q{question['question_id']}: {e}")
                    traceback.print_exc()
                    predicted = "ERROR"

                correct = question["correct_answer"].strip().upper()[0]
                is_correct = predicted == correct

                result = {
                    "video_id": video_id,
                    "question_id": question["question_id"],
                    "capability": question["capability"],
                    "referral_strategy": question["referral_strategy"],
                    "difficulty": question["difficulty"],
                    "predicted": predicted,
                    "correct": correct,
                    "is_correct": is_correct
                }
                results["predictions"].append(result)

                status = "[OK]" if is_correct else "[X]"
                print(f"  [{current}/{total_questions}] Q{question['question_id']}: {status} (pred={predicted}, gt={correct})")

        return results

    def run_evaluation(self, models: List[BaseVLM]) -> Dict:
        """Run evaluation on multiple models."""
        for model in models:
            results = self.evaluate_model(model)
            self.results[model.model_name] = results

        # Save raw results
        results_path = self.output_dir / "raw_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nRaw results saved to: {results_path}")

        return self.results

    def compute_metrics(self) -> pd.DataFrame:
        """Compute comprehensive metrics from results."""
        all_metrics = []

        for model_name, result in self.results.items():
            predictions = result["predictions"]
            df = pd.DataFrame(predictions)

            # Overall accuracy
            overall_acc = df["is_correct"].mean() * 100

            # Per-capability accuracy
            cap_acc = df.groupby("capability")["is_correct"].mean() * 100

            # Per-difficulty accuracy
            diff_acc = df.groupby("difficulty")["is_correct"].mean() * 100

            # Per-referral strategy accuracy
            ref_acc = df.groupby("referral_strategy")["is_correct"].mean() * 100

            # Per-video accuracy
            video_acc = df.groupby("video_id")["is_correct"].mean() * 100

            metrics = {
                "model": model_name,
                "overall_accuracy": overall_acc,
                "total_correct": df["is_correct"].sum(),
                "total_questions": len(df),
            }

            # Add capability metrics
            for cap in cap_acc.index:
                metrics[f"cap_{cap}"] = cap_acc[cap]

            # Add difficulty metrics
            for diff in diff_acc.index:
                metrics[f"diff_{diff}"] = diff_acc[diff]

            # Add referral strategy metrics
            for ref in ref_acc.index:
                metrics[f"ref_{ref.replace(' ', '_')}"] = ref_acc[ref]

            # Add video metrics
            for vid in video_acc.index:
                metrics[f"video_{vid}"] = video_acc[vid]

            all_metrics.append(metrics)

        metrics_df = pd.DataFrame(all_metrics)

        # Save metrics
        metrics_path = self.output_dir / "metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to: {metrics_path}")

        return metrics_df


# =============================================================================
# Visualization Functions
# =============================================================================

class BenchmarkVisualizer:
    """Creates visualizations for benchmark results."""

    def __init__(self, results: Dict, output_dir: Path):
        self.results = results
        self.output_dir = output_dir
        self.colors = plt.cm.Set2(np.linspace(0, 1, 10))

    def plot_all(self):
        """Generate all visualization plots."""
        print("\nGenerating visualizations...")

        self.plot_overall_accuracy()
        self.plot_accuracy_by_capability()
        self.plot_accuracy_by_difficulty()
        self.plot_accuracy_by_referral_strategy()
        self.plot_accuracy_by_video()
        self.plot_capability_radar()
        self.plot_confusion_matrices()
        self.plot_difficulty_capability_heatmap()
        self.plot_error_analysis()
        self.plot_performance_distribution()
        self.plot_comparative_bar_chart()

        print(f"All visualizations saved to: {self.output_dir}")

    def _get_all_predictions_df(self) -> pd.DataFrame:
        """Combine all predictions into a single DataFrame."""
        all_preds = []
        for model_name, result in self.results.items():
            for pred in result["predictions"]:
                pred_copy = pred.copy()
                pred_copy["model"] = model_name
                all_preds.append(pred_copy)
        return pd.DataFrame(all_preds)

    def plot_overall_accuracy(self):
        """Plot overall accuracy comparison across models."""
        fig, ax = plt.subplots(figsize=(10, 6))

        models = []
        accuracies = []

        for model_name, result in self.results.items():
            preds = result["predictions"]
            acc = sum(p["is_correct"] for p in preds) / len(preds) * 100
            models.append(model_name)
            accuracies.append(acc)

        bars = ax.bar(models, accuracies, color=self.colors[:len(models)], edgecolor='black', linewidth=1.2)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='Random Baseline (25%)')
        ax.legend()

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'overall_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_accuracy_by_capability(self):
        """Plot accuracy breakdown by capability."""
        df = self._get_all_predictions_df()

        capabilities = df['capability'].unique()
        models = df['model'].unique()

        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(len(capabilities))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            model_df = df[df['model'] == model]
            accs = [model_df[model_df['capability'] == cap]['is_correct'].mean() * 100
                   for cap in capabilities]
            bars = ax.bar(x + i * width, accs, width, label=model, alpha=0.85)

        ax.set_xlabel('Capability', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy by Capability', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(capabilities, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 105)
        ax.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='Random Baseline')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_by_capability.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_accuracy_by_difficulty(self):
        """Plot accuracy breakdown by difficulty level."""
        df = self._get_all_predictions_df()

        difficulties = ['Easy', 'Medium', 'Hard']
        models = df['model'].unique()

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(difficulties))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            model_df = df[df['model'] == model]
            accs = []
            for diff in difficulties:
                diff_df = model_df[model_df['difficulty'] == diff]
                acc = diff_df['is_correct'].mean() * 100 if len(diff_df) > 0 else 0
                accs.append(acc)
            ax.bar(x + i * width, accs, width, label=model, alpha=0.85)

        ax.set_xlabel('Difficulty', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy by Difficulty Level', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(difficulties)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 105)
        ax.axhline(y=25, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_by_difficulty.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_accuracy_by_referral_strategy(self):
        """Plot accuracy breakdown by referral strategy."""
        df = self._get_all_predictions_df()

        strategies = df['referral_strategy'].unique()
        models = df['model'].unique()

        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(len(strategies))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            model_df = df[df['model'] == model]
            accs = [model_df[model_df['referral_strategy'] == strat]['is_correct'].mean() * 100
                   for strat in strategies]
            ax.bar(x + i * width, accs, width, label=model, alpha=0.85)

        ax.set_xlabel('Referral Strategy', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy by Referral Strategy', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 105)
        ax.axhline(y=25, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_by_referral_strategy.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_accuracy_by_video(self):
        """Plot accuracy breakdown by video."""
        df = self._get_all_predictions_df()

        videos = df['video_id'].unique()
        models = df['model'].unique()

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(videos))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            model_df = df[df['model'] == model]
            accs = [model_df[model_df['video_id'] == vid]['is_correct'].mean() * 100
                   for vid in videos]
            ax.bar(x + i * width, accs, width, label=model, alpha=0.85)

        ax.set_xlabel('Video ID', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy by Video', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(videos, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 105)
        ax.axhline(y=25, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_by_video.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_capability_radar(self):
        """Create radar chart comparing capabilities across models."""
        df = self._get_all_predictions_df()

        capabilities = sorted(df['capability'].unique())
        models = df['model'].unique()

        # Number of variables
        N = len(capabilities)

        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

        for idx, model in enumerate(models):
            model_df = df[df['model'] == model]
            values = [model_df[model_df['capability'] == cap]['is_correct'].mean() * 100
                     for cap in capabilities]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(capabilities, fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_title('Capability Comparison Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'capability_radar.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrices(self):
        """Plot confusion matrices for each model."""
        df = self._get_all_predictions_df()
        models = df['model'].unique()

        n_models = len(models)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        options = ['A', 'B', 'C', 'D']

        for idx, model in enumerate(models):
            model_df = df[df['model'] == model]

            # Create confusion matrix
            cm = np.zeros((4, 4), dtype=int)
            for _, row in model_df.iterrows():
                pred = row['predicted']
                true = row['correct']
                if pred in options and true in options:
                    cm[options.index(true)][options.index(pred)] += 1

            # Normalize
            cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm_normalized = np.nan_to_num(cm_normalized)

            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=options, yticklabels=options, ax=axes[idx],
                       cbar_kws={'label': 'Proportion'})
            axes[idx].set_xlabel('Predicted', fontsize=10)
            axes[idx].set_ylabel('True', fontsize=10)
            axes[idx].set_title(f'{model}', fontsize=11, fontweight='bold')

        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Confusion Matrices (Normalized)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_difficulty_capability_heatmap(self):
        """Create heatmap of accuracy across difficulty and capability dimensions."""
        df = self._get_all_predictions_df()
        models = df['model'].unique()

        n_models = len(models)
        cols = min(2, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        capabilities = sorted(df['capability'].unique())
        difficulties = ['Easy', 'Medium', 'Hard']

        for idx, model in enumerate(models):
            model_df = df[df['model'] == model]

            # Create accuracy matrix
            acc_matrix = np.zeros((len(difficulties), len(capabilities)))
            for i, diff in enumerate(difficulties):
                for j, cap in enumerate(capabilities):
                    subset = model_df[(model_df['difficulty'] == diff) &
                                     (model_df['capability'] == cap)]
                    if len(subset) > 0:
                        acc_matrix[i, j] = subset['is_correct'].mean() * 100
                    else:
                        acc_matrix[i, j] = np.nan

            sns.heatmap(acc_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                       xticklabels=capabilities, yticklabels=difficulties,
                       ax=axes[idx], vmin=0, vmax=100,
                       cbar_kws={'label': 'Accuracy (%)'})
            axes[idx].set_xlabel('Capability', fontsize=10)
            axes[idx].set_ylabel('Difficulty', fontsize=10)
            axes[idx].set_title(f'{model}', fontsize=11, fontweight='bold')
            plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Accuracy by Difficulty & Capability', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'difficulty_capability_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_error_analysis(self):
        """Analyze and visualize common error patterns."""
        df = self._get_all_predictions_df()

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Error rate by capability
        ax1 = axes[0, 0]
        error_by_cap = df.groupby('capability')['is_correct'].apply(lambda x: (1 - x.mean()) * 100)
        error_by_cap.sort_values(ascending=False).plot(kind='barh', ax=ax1, color='coral')
        ax1.set_xlabel('Error Rate (%)')
        ax1.set_title('Error Rate by Capability', fontweight='bold')
        ax1.axvline(x=75, color='red', linestyle='--', alpha=0.5, label='Random Error (75%)')

        # 2. Error rate by difficulty
        ax2 = axes[0, 1]
        error_by_diff = df.groupby('difficulty')['is_correct'].apply(lambda x: (1 - x.mean()) * 100)
        order = ['Easy', 'Medium', 'Hard']
        error_by_diff = error_by_diff.reindex(order)
        error_by_diff.plot(kind='bar', ax=ax2, color=['green', 'orange', 'red'], alpha=0.7)
        ax2.set_xlabel('Difficulty')
        ax2.set_ylabel('Error Rate (%)')
        ax2.set_title('Error Rate by Difficulty', fontweight='bold')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

        # 3. Most common wrong answer patterns
        ax3 = axes[1, 0]
        wrong_df = df[~df['is_correct']]
        if len(wrong_df) > 0:
            wrong_df['pattern'] = wrong_df['correct'] + ' -> ' + wrong_df['predicted']
            pattern_counts = wrong_df['pattern'].value_counts().head(10)
            pattern_counts.plot(kind='barh', ax=ax3, color='steelblue')
            ax3.set_xlabel('Count')
            ax3.set_title('Most Common Wrong Answer Patterns', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No Errors!', ha='center', va='center', fontsize=14)
            ax3.set_title('Most Common Wrong Answer Patterns', fontweight='bold')

        # 4. Error distribution across models
        ax4 = axes[1, 1]
        models = df['model'].unique()
        model_errors = []
        for model in models:
            model_df = df[df['model'] == model]
            error_rate = (1 - model_df['is_correct'].mean()) * 100
            model_errors.append({'model': model, 'error_rate': error_rate})

        error_df = pd.DataFrame(model_errors)
        error_df.set_index('model')['error_rate'].plot(kind='bar', ax=ax4, color='crimson', alpha=0.7)
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Error Rate (%)')
        ax4.set_title('Error Rate by Model', fontweight='bold')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.axhline(y=75, color='red', linestyle='--', alpha=0.5, label='Random Baseline')

        plt.suptitle('Error Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_performance_distribution(self):
        """Plot distribution of correct answers across questions."""
        df = self._get_all_predictions_df()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Distribution of accuracy across questions
        ax1 = axes[0]
        question_acc = df.groupby(['video_id', 'question_id'])['is_correct'].mean()
        ax1.hist(question_acc, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(x=question_acc.mean(), color='red', linestyle='--',
                   label=f'Mean: {question_acc.mean()*100:.1f}%')
        ax1.set_xlabel('Accuracy per Question')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of Per-Question Accuracy', fontweight='bold')
        ax1.legend()

        # 2. Questions ranked by difficulty (how many models got them right)
        ax2 = axes[1]
        question_success = df.groupby(['video_id', 'question_id']).agg({
            'is_correct': 'sum',
            'capability': 'first',
            'difficulty': 'first'
        }).reset_index()
        question_success['success_rate'] = question_success['is_correct'] / len(df['model'].unique())

        # Color by difficulty
        colors = {'Easy': 'green', 'Medium': 'orange', 'Hard': 'red'}
        for diff in ['Easy', 'Medium', 'Hard']:
            subset = question_success[question_success['difficulty'] == diff]
            ax2.scatter(range(len(subset)), sorted(subset['success_rate']),
                       label=diff, alpha=0.7, c=colors[diff], s=50)

        ax2.set_xlabel('Questions (sorted)')
        ax2.set_ylabel('Success Rate (across models)')
        ax2.set_title('Question Difficulty Analysis', fontweight='bold')
        ax2.legend()
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_comparative_bar_chart(self):
        """Create a comprehensive comparative bar chart."""
        df = self._get_all_predictions_df()
        models = df['model'].unique()

        metrics = {
            'Overall': df.groupby('model')['is_correct'].mean() * 100,
            'Easy': df[df['difficulty'] == 'Easy'].groupby('model')['is_correct'].mean() * 100,
            'Medium': df[df['difficulty'] == 'Medium'].groupby('model')['is_correct'].mean() * 100,
            'Hard': df[df['difficulty'] == 'Hard'].groupby('model')['is_correct'].mean() * 100,
        }

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(models))
        width = 0.2

        for i, (metric_name, metric_values) in enumerate(metrics.items()):
            values = [metric_values.get(m, 0) for m in models]
            bars = ax.bar(x + i * width, values, width, label=metric_name, alpha=0.85)

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=8, rotation=90)

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Comprehensive Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 110)
        ax.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='Random Baseline')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparative_bar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()


# =============================================================================
# Report Generator
# =============================================================================

def generate_report(results: Dict, metrics_df: pd.DataFrame, output_dir: Path):
    """Generate a summary report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("VIDEO RE-ID BENCHMARK EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Output Directory: {output_dir}")

    report_lines.append("\n" + "-" * 40)
    report_lines.append("OVERALL RESULTS")
    report_lines.append("-" * 40)

    for _, row in metrics_df.iterrows():
        report_lines.append(f"\n{row['model']}:")
        report_lines.append(f"  Overall Accuracy: {row['overall_accuracy']:.2f}%")
        report_lines.append(f"  Correct: {int(row['total_correct'])}/{int(row['total_questions'])}")

    report_lines.append("\n" + "-" * 40)
    report_lines.append("ACCURACY BY CAPABILITY")
    report_lines.append("-" * 40)

    cap_cols = [c for c in metrics_df.columns if c.startswith('cap_')]
    for cap_col in cap_cols:
        cap_name = cap_col.replace('cap_', '')
        report_lines.append(f"\n{cap_name}:")
        for _, row in metrics_df.iterrows():
            report_lines.append(f"  {row['model']}: {row[cap_col]:.2f}%")

    report_lines.append("\n" + "-" * 40)
    report_lines.append("ACCURACY BY DIFFICULTY")
    report_lines.append("-" * 40)

    diff_cols = [c for c in metrics_df.columns if c.startswith('diff_')]
    for diff_col in diff_cols:
        diff_name = diff_col.replace('diff_', '')
        report_lines.append(f"\n{diff_name}:")
        for _, row in metrics_df.iterrows():
            report_lines.append(f"  {row['model']}: {row[diff_col]:.2f}%")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)

    # Save report
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {report_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate VLMs on Video Re-ID Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with mock models for testing
    python evaluate_vlm_benchmark.py \\
        --benchmark benchmark_questions.json \\
        --video_dir /path/to/videos \\
        --models qwen2-vl ovis \\
        --mock

    # Run actual evaluation
    python evaluate_vlm_benchmark.py \\
        --benchmark benchmark_questions.json \\
        --video_dir /path/to/videos \\
        --models qwen2-vl ovis llava-next-video internvl2 video-llava

Available Models:
    - qwen2-vl (Qwen2-VL-7B-Instruct)
    - qwen2-vl-2b (Qwen2-VL-2B-Instruct)
    - qwen2-vl-72b (Qwen2-VL-72B-Instruct)
    - ovis (Ovis1.6-Gemma2-9B)
    - llava-next-video (LLaVA-NeXT-Video-7B)
    - llava-next-video-34b (LLaVA-NeXT-Video-34B)
    - internvl2 (InternVL2-8B)
    - internvl2-26b (InternVL2-26B)
    - video-llava (Video-LLaVA-7B)
        """
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Path to benchmark JSON file"
    )

    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing video files"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen2-vl", "ovis", "llava-next-video", "internvl2", "video-llava"],
        help="List of models to evaluate"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results and plots"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock models for testing (random predictions)"
    )

    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip pre-flight validation checks"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("VIDEO RE-ID BENCHMARK EVALUATION")
    print("=" * 60)
    print(f"Benchmark: {args.benchmark}")
    print(f"Video Directory: {args.video_dir}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Mock Mode: {args.mock}")
    print("=" * 60)

    # Initialize evaluator
    evaluator = BenchmarkEvaluator(
        benchmark_path=args.benchmark,
        video_dir=args.video_dir,
        output_dir=args.output_dir
    )

    # Load models
    models = []
    for model_name in args.models:
        try:
            model = get_model(model_name, args.device, mock=args.mock)
            models.append(model)
        except ValueError as e:
            print(f"Warning: {e}")
            continue

    if not models:
        print("Error: No valid models specified!")
        sys.exit(1)

    # Pre-flight validation
    if not args.skip_validation:
        is_valid, validation_report = evaluator.validate_prerequisites(models, mock=args.mock)

        # Save validation report
        validation_path = evaluator.output_dir / "validation_report.json"
        with open(validation_path, 'w') as f:
            json.dump(validation_report, f, indent=2)
        print(f"Validation report saved to: {validation_path}")

        if not is_valid and not args.mock:
            print("\nERROR: Pre-flight validation failed!")
            print("Fix the issues above or use --skip_validation to proceed anyway.")
            sys.exit(1)
    else:
        print("\n[!] Skipping pre-flight validation (--skip_validation)")

    # Run evaluation
    results = evaluator.run_evaluation(models)

    # Compute metrics
    metrics_df = evaluator.compute_metrics()

    # Generate visualizations
    visualizer = BenchmarkVisualizer(results, evaluator.output_dir)
    visualizer.plot_all()

    # Generate report
    generate_report(results, metrics_df, evaluator.output_dir)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()