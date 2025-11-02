#!/usr/bin/env python3
"""
Evaluate Qwen on your video-QA template and write an augmented JSON with model answers.

Inputs
  - Template.json   (list of items: {id, question_type, question, answer})
  - Template Videos/ (contains <id>.mp4, or a folder <id> with the video)

Output
  - Template_qwen_answers.json (same entries + model_answer, model_raw, optional model_rationale, video_path)

Backends
  --backend transformers  -> local HF model (e.g., Qwen/Qwen2-VL-2B-Instruct)
  --backend openai        -> OpenAI-compatible API (frames sent as images)
"""

import argparse
import json
import re
import cv2
import math
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

print("[BANNER] evaluate_qwen_videos.py:", __file__)

# ---------------- Utilities ----------------

TIMECODE_RE = re.compile(r"\b(\d{2}):(\d{2})(?::(\d{2}))?\b")

def parse_timecode_to_seconds(tc: str) -> float:
    m = TIMECODE_RE.search(tc)
    if not m:
        return -1
    hh, mm, ss = m.group(1), m.group(2), m.group(3) or "00"
    return int(hh) * 3600 + int(mm) * 60 + int(ss)

def extract_timecodes(text: str) -> List[str]:
    return [m.group(0) for m in TIMECODE_RE.finditer(text)]

def has_timecode(text: str) -> bool:
    return bool(TIMECODE_RE.search(text or ""))

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def find_video_path(vid_id: str, videos_dir: Path) -> Optional[Path]:
    # Try <id>.mp4, <id>.mkv, <id>.mov; also allow a subfolder "<id>/*.ext"
    candidates = [
        videos_dir / f"{vid_id}.mp4",
        videos_dir / f"{vid_id}.mkv",
        videos_dir / f"{vid_id}.mov",
    ]
    subdir = videos_dir / vid_id
    if subdir.exists():
        for ext in (".mp4", ".mkv", ".mov"):
            for p in subdir.glob(f"*{ext}"):
                candidates.append(p)
    for c in candidates:
        if c.exists():
            return c
    return None

# ---------------- Frame sampling ----------------

@dataclass
class FrameSpec:
    t: float
    idx: int
    path: Path

def _save_frame(img, out_dir: Path, stem: str, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{stem}_{tag}.jpg"
    cv2.imwrite(str(fname), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return fname

def _ffmpeg_grab_first_frame(video_path: Path, out_dir: Path) -> Optional[FrameSpec]:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{video_path.stem}_ff0.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-ss", "0", "-i", str(video_path), "-frames:v", "1", str(out)],
            check=True,
        )
        if out.exists():
            return FrameSpec(t=0.0, idx=0, path=out)
    except Exception:
        pass
    return None

def sample_frames(
    video_path: Path,
    hinted_times: List[float],
    fallback_n: int = 12,
    radius: float = 2.0,
    max_total: int = 24,
    out_dir: Optional[Path] = None,
    debug: bool = False,
) -> List[FrameSpec]:
    """
    Strategy:
    1) Random-access seeks around hinted times or evenly spaced times.
    2) If no frames, do a linear scan with stride to harvest frames.
    3) If still no frames, grab frame 0 with ffmpeg.
    Always return >=1 frame when the file is readable.
    """
    out_dir = out_dir or (video_path.parent / f"_frames_{video_path.stem}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames > 0 else 0.0

    # Desired timestamps
    wanted: List[float] = []
    if hinted_times:
        for t in hinted_times:
            for dt in (-radius, 0.0, +radius):
                tt = clamp(t + dt, 0.0, max(0.0, duration - 1e-6))
                wanted.append(tt)
    else:
        for i in range(max(1, fallback_n)):
            tt = (i + 0.5) / max(1, fallback_n) * max(0.0, duration - 1e-6)
            wanted.append(tt)

    wanted_sorted = sorted(set(round(x, 2) for x in wanted))
    if len(wanted_sorted) > max_total:
        step = math.ceil(len(wanted_sorted) / max_total)
        wanted_sorted = wanted_sorted[::step]

    frames: List[FrameSpec] = []

    # 1) Random-access tries
    for t in wanted_sorted:
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = cap.read()
        if ok and img is not None:
            tag = str(int(round(t * 1000)))
            path = _save_frame(img, out_dir, video_path.stem, tag)
            frames.append(FrameSpec(t=t, idx=frame_idx, path=path))

    # 2) Linear scan fallback if needed
    if not frames and total_frames > 0:
        want = max(1, min(max_total, fallback_n))
        stride = max(1, total_frames // (want + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        got = 0
        while True:
            ok, img = cap.read()
            if not ok or img is None:
                break
            if idx % stride == 0:
                tag = f"ls{idx}"
                path = _save_frame(img, out_dir, video_path.stem, tag)
                frames.append(FrameSpec(t=idx / max(1.0, fps), idx=idx, path=path))
                got += 1
                if got >= want:
                    break
            idx += 1

    cap.release()

    # 3) ffmpeg final fallback
    if not frames:
        ff = _ffmpeg_grab_first_frame(video_path, out_dir)
        if ff:
            frames.append(ff)

    if debug:
        print(f"[DEBUG] sample_frames: {video_path.name} fps={fps:.2f} total={total_frames} "
              f"duration={duration:.2f}s -> kept {len(frames)} frames")

    return frames

# ---------------- Backends ----------------

class QwenBackend:
    def answer(self, question: str, images: List[Path], question_type: str):
        raise NotImplementedError

class TransformersQwenStrict(QwenBackend):
    """
    Strict loader for Qwen2-VL. Uses apply_chat_template so image tokens match image features.
    """
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct", debug: bool = False, max_images: int = 12):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, __version__ as hf_ver

        self.model_id = model_id
        self.debug = debug
        self.uses_chat_api = False
        self.torch = torch
        self.max_images = max_images

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_kwargs = dict(device_map="auto")

        if self.debug:
            print(f"[DEBUG] transformers={hf_ver}")
            print("[DEBUG] Strict Qwen2VLForConditionalGeneration path with chat template")

        try:
            try:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    dtype=dtype,
                    **device_kwargs
                )
            except TypeError:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    **device_kwargs
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {self.model_id} via Qwen2VLForConditionalGeneration. Inner: {e}"
            )

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load AutoProcessor for {self.model_id}: {e}")

        if not hasattr(self.model, "generate"):
            raise AttributeError(
                f"{type(self.model).__name__} loaded but has no .generate(). Upgrade transformers."
            )

        if self.debug:
            print("[DEBUG] Loaded class:", type(self.model).__name__)
            print("[DEBUG] has_generate:", hasattr(self.model, "generate"))

    def _build_chat(self, question: str, question_type: str, num_images: int) -> List[dict]:
        guide = {
            "Yes/No": "Answer strictly with Yes or No. If unsure, guess the most likely.",
            "True/False": "Answer strictly with True or False. If unsure, guess the most likely.",
            "Timestamp": "Answer strictly with a timestamp in HH:MM:SS. If no clear answer, return the most plausible timestamp in HH:MM:SS."
        }.get(question_type, "Answer concisely.")

        sys_prompt = (
            "You are a precise video QA assistant. You see key frames extracted from a video. "
            "Use only visible evidence; be consistent with identities across frames. "
            "If the question mentions timestamps, assume frames come from around those times.\n"
            f"{guide} Then add one short sentence rationale starting with 'Because'."
        )
        content = [{"type": "text", "text": question}]
        for _ in range(num_images):
            content.append({"type": "image"})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user",   "content": content}
        ]
        return messages

    def answer(self, question: str, images: List[Path], question_type: str):
        from PIL import Image

        if images:
            images = images[: self.max_images]

        pil_images = [Image.open(p).convert("RGB") for p in images] if images else []

        if pil_images:
            messages = self._build_chat(question, question_type, num_images=len(pil_images))
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[pil_images], return_tensors="pt").to(self.model.device)
        else:
            messages = self._build_chat(question, question_type, num_images=0)
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)

        gen_ids = self.model.generate(**inputs, max_new_tokens=128)
        text_out = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
        full_text = (text_out[0] if text_out else "").strip()

        if not full_text:
            messages = self._build_chat(question, question_type, num_images=0)
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)
            gen_ids = self.model.generate(**inputs, max_new_tokens=128)
            text_out = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
            full_text = (text_out[0] if text_out else "").strip()

        # Return only one string; we'll parse assistant/rationale in caller
        return full_text

class OpenAICompatQwen(QwenBackend):
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None, debug: bool = False):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model = model
        self.debug = debug

    def _image_to_dataurl(self, path: Path) -> str:
        import base64, mimetypes
        mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def answer(self, question: str, images: List[Path], question_type: str):
        content = [{"type": "text", "text": question}]
        for p in images[:20]:
            content.append({"type": "input_image", "image_url": self._image_to_dataurl(p)})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content":
                       "You are a precise video QA assistant. You see key frames extracted from a video. "
                       "Use only visible evidence; be consistent with identities across frames. "
                       "If the question mentions timestamps, assume frames come from around those times. "
                       "Answer strictly as requested and add one short sentence rationale starting with 'Because'."},
                      {"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=128
        )
        return (resp.choices[0].message.content or "").strip()

# ---------------- Parsing & Postprocess ----------------

def extract_after_assistant(full_text: str) -> str:
    """
    If the model echoes 'system\\n... user\\n... assistant\\n<answer>',
    strip everything up to and including the 'assistant' marker.
    """
    if not full_text:
        return ""
    m = re.search(r'(?:^|\n)assistant\s*\n?', full_text, flags=re.IGNORECASE)
    if m:
        return full_text[m.end():].strip()
    return full_text.strip()

def split_answer_and_rationale(assistant_text: str) -> Tuple[str, str]:
    """
    Split assistant text into the main answer and trailing rationale
    starting with 'Because ...'. Returns (answer_only, rationale_text).
    """
    if not assistant_text:
        return "", ""
    m = re.search(r"(Because .+)$", assistant_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return assistant_text[:m.start()].strip(), m.group(1).strip()
    return assistant_text.strip(), ""

def normalize_answer(raw: str, question_type: str) -> str:
    txt = (raw or "").strip()
    low = txt.lower()

    if question_type == "Yes/No":
        if "yes" in low and "no" not in low: return "Yes"
        if "no" in low and "yes" not in low: return "No"
        if low.startswith("y"): return "Yes"
        if low.startswith("n"): return "No"
        return txt

    if question_type == "True/False":
        if "true" in low and "false" not in low: return "True"
        if "false" in low and "true" not in low: return "False"
        if low.startswith("t"): return "True"
        if low.startswith("f"): return "False"
        return txt

    if question_type == "Timestamp":
        m = TIMECODE_RE.search(txt)
        if m:
            hh, mm, ss = m.group(1), m.group(2), m.group(3) or "00"
            return f"{int(hh):02d}:{int(mm):02d}:{int(ss):02d}"
        return txt

    return txt

# ---------------- Core ----------------

def run(
    qa_json_path: Path,
    videos_dir: Path,
    out_json_path: Path,
    backend: str = "transformers",
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    openai_model: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    frames_per_question: int = 12,
    context_radius_sec: float = 2.0,
    max_frames: int = 24,
    add_rationale: bool = True,
    debug: bool = False,
):
    data = json.loads(Path(qa_json_path).read_text())
    frames_cache: Dict[Tuple[str, Tuple[float, ...]], List[Path]] = {}

    # Init backend
    if backend == "transformers":
        qwen: QwenBackend = TransformersQwenStrict(model_id=model_id, debug=debug, max_images=min(12, max_frames))
        if debug:
            print("[DEBUG] Using loader class:", type(qwen).__name__)
    elif backend == "openai":
        if not openai_model or not openai_api_key:
            raise ValueError("backend=openai requires --openai-model and --openai-api-key (optionally --openai-base-url).")
        qwen = OpenAICompatQwen(model=openai_model, api_key=openai_api_key, base_url=openai_base_url, debug=debug)
    else:
        raise ValueError("backend must be 'transformers' or 'openai'")

    augmented: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        vid_id = item.get("id")
        qtype = item.get("question_type")
        question = item.get("question")

        if not vid_id or not question or not qtype:
            if debug:
                print(f"[DEBUG] Skipping malformed item index {i}: {item}")
            continue

        vpath = find_video_path(vid_id, videos_dir)
        if not vpath:
            if debug:
                print(f"[DEBUG] No video for id={vid_id} under {videos_dir}. Answering without images.")
            images: List[Path] = []
        else:
            tcs = extract_timecodes(question)
            hinted = [parse_timecode_to_seconds(tc) for tc in tcs if parse_timecode_to_seconds(tc) >= 0]
            key = (str(vpath), tuple(sorted(hinted)))
            if key not in frames_cache:
                frames = sample_frames(
                    vpath,
                    hinted_times=hinted,
                    fallback_n=frames_per_question,
                    radius=context_radius_sec,
                    max_total=max_frames,
                    debug=debug
                )
                frames_cache[key] = [f.path for f in frames]
            images = frames_cache[key]

        directive = {
            "Yes/No": "Answer strictly Yes or No.",
            "True/False": "Answer strictly True or False.",
            "Timestamp": "Answer strictly with a timestamp in HH:MM:SS."
        }.get(qtype, "Answer concisely.")
        full_q = f"{question}\n\nConstraints: {directive}"

        try:
            full_text = qwen.answer(full_q, images, qtype)

            # Keep full text for debugging regardless
            model_raw_text = full_text

            # 1) Cut to the assistant segment
            assistant_text = extract_after_assistant(full_text)

            # 2) Split off rationale
            ans_main, rationale = split_answer_and_rationale(assistant_text)

            # 3) Normalize main
            model_answer = normalize_answer(ans_main, qtype).strip()

            # 4) Fallbacks
            if qtype in ("Yes/No", "True/False"):
                if not model_answer or model_answer.lower() in ("system", "user", "assistant"):
                    # try pulling from rationale if it accidentally held the token
                    fallback = normalize_answer(rationale, qtype).strip()
                    if fallback and fallback.lower() not in ("system", "user", "assistant"):
                        model_answer = fallback

            elif qtype == "Timestamp":
                # If assistant text has no time, try rationale for a time; else set 00:00:00
                if not has_timecode(model_answer):
                    alt = normalize_answer(rationale, qtype).strip()
                    if has_timecode(alt):
                        model_answer = alt
                    else:
                        model_answer = "00:00:00"

        except Exception as e:
            model_raw_text = f"[ERROR] {type(e).__name__}: {e}"
            rationale = ""
            model_answer = "[ERROR]"

        out_item = dict(item)
        out_item["model_answer"] = model_answer
        out_item["model_raw"] = model_raw_text
        if add_rationale and rationale:
            out_item["model_rationale"] = rationale
        if vpath:
            out_item["video_path"] = str(vpath)
        augmented.append(out_item)

        if debug and (i % 10 == 0):
            print(f"[DEBUG] processed {i+1}/{len(data)} items")

    Path(out_json_path).write_text(json.dumps(augmented, indent=2))
    print(f"[OK] wrote {out_json_path} with {len(augmented)} items.")

# ---------------- CLI ----------------

def main():
    parser = argparse.ArgumentParser(description="Run Qwen over Template.json and Template Videos/ to create Template_qwen_answers.json")
    parser.add_argument("--qa-json", default="Template.json", type=Path)
    parser.add_argument("--videos-dir", default=Path("Template Videos"), type=Path)
    parser.add_argument("--out-json", default="Template_qwen_answers.json", type=Path)
    parser.add_argument("--backend", choices=["transformers","openai"], default="transformers")
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--openai-model")
    parser.add_argument("--openai-base-url")
    parser.add_argument("--openai-api-key")
    parser.add_argument("--frames-per-question", type=int, default=12)
    parser.add_argument("--context-radius-sec", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--no-rationale", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run(
        qa_json_path=args.qa_json,
        videos_dir=args.videos_dir,
        out_json_path=args.out_json,
        backend=args.backend,
        model_id=args.model_id,
        openai_model=args.openai_model,
        openai_base_url=args.openai_base_url,
        openai_api_key=args.openai_api_key,
        frames_per_question=args.frames_per_question,
        context_radius_sec=args.context_radius_sec,
        max_frames=args.max_frames,
        add_rationale=not args.no_rationale,
        debug=args.debug,
    )

if __name__ == "__main__":
    main()

