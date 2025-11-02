#!/usr/bin/env python3
"""
Evaluate Ovis 2.5 on your video-QA template and write an augmented JSON with model answers.

Inputs
  - Template.json   (list of items: {id, question_type, question, answer})
  - Template Videos/ (contains <id>.mp4, or a folder <id> with the video)

Output
  - Template_ovis_answers.json (same entries + model_answer, model_raw, optional model_rationale, video_path)

Backend: Hugging Face transformers with trust_remote_code using Ovis 2.5
Recommended public IDs:  AIDC-AI/Ovis2.5-2B  or  AIDC-AI/Ovis2.5-9B

Example:
  python evaluate_ovis.py \
    --qa-json "Template.json" \
    --videos-dir "Template Videos" \
    --out-json "Template_ovis_answers.json" \
    --model-id AIDC-AI/Ovis2.5-2B \
    --debug
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

print("[BANNER] evaluate_ovis.py:", __file__)

# ---------------- Utilities ----------------

TIMECODE = re.compile(r"\b(\d{2}):(\d{2})(?::(\d{2}))?\b")

def parse_timecode_to_seconds(tc: str) -> float:
    m = TIMECODE.search(tc)
    if not m:
        return -1
    hh, mm, ss = m.group(1), m.group(2), m.group(3) or "00"
    return int(hh) * 3600 + int(mm) * 60 + int(ss)

def extract_timecodes(text: str) -> List[str]:
    return [m.group(0) for m in TIMECODE.finditer(text)]

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
    1) Try random-access seeks around hinted times or evenly spaced times.
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

# ---------------- Backend (Ovis 2.5) ----------------

class VisionChatBackend:
    def answer(self, question: str, images: List[Path], question_type: str):
        raise NotImplementedError

class TransformersOvisStrict(VisionChatBackend):
    """
    Ovis 2.5 loader (AIDC-AI/*). Use model.preprocess_inputs(messages=...) with content types:
    - {"type": "video", "video": [PIL.Image,...]}
    - {"type": "image", "image": PIL.Image}
    - {"type": "text", "text": "..."}
    """
    def __init__(self, model_id: str = "AIDC-AI/Ovis2.5-2B", debug: bool = False, max_images: int = 12):
        import os
        import torch
        from transformers import AutoModelForCausalLM, __version__ as hf_ver
        self.debug = debug
        self.model_id = model_id
        self.max_images = max_images

        # dtype preference
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32

        if self.debug:
            print(f"[DEBUG] transformers={hf_ver}")
            print("[DEBUG] Loading Ovis via AutoModelForCausalLM + trust_remote_code")

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
                token=token  # works for public as well
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_id}: {e}")

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # Sanity: text_tokenizer presence
        if not hasattr(self.model, "text_tokenizer"):
            raise RuntimeError(f"Model {self.model_id} missing .text_tokenizer (Ovis API expected).")

    def _format_instructions(self, question_type: str) -> str:
        return {
            "Yes/No": "Answer strictly with Yes or No. If unsure, guess the most likely. Then add one short sentence rationale starting with 'Because'.",
            "True/False": "Answer strictly with True or False. If unsure, guess the most likely. Then add one short sentence rationale starting with 'Because'.",
            "Timestamp": "Answer strictly with a timestamp in HH:MM:SS. If no clear answer, return the most plausible timestamp in HH:MM:SS. Then add one short sentence rationale starting with 'Because'."
        }.get(question_type, "Answer concisely. Then add one short sentence rationale starting with 'Because'.")

    def _split_rationale(self, text: str):
        import re
        m = re.search(r"(Because .+)$", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return text[:m.start()].strip(), m.group(1).strip()
        return text.strip(), ""

    def answer(self, question: str, images: List[Path], question_type: str):
        from PIL import Image
        import torch

        guide = self._format_instructions(question_type)
        user_text = f"{question}\n\nConstraints: {guide}"

        # Build messages: prefer a single "video" block
        content = []
        frames_pil = []
        if images:
            use = images[: self.max_images]
            for p in use:
                frames_pil.append(Image.open(p).convert("RGB"))
            content.append({"type": "video", "video": frames_pil})
        content.append({"type": "text", "text": user_text})
        messages = [{"role": "user", "content": content}]

        # Preprocess (Ovis custom)
        try:
            input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
                messages=messages,
                add_generation_prompt=True,
                max_pixels=896*896,         # tune if you see OOM or empty outputs; try 768*768 if needed
                enable_thinking=False
            )
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] preprocess_inputs failed: {e}")
            # last-resort text-only
            messages = [{"role": "user", "content": question}]
            input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
                messages=messages, add_generation_prompt=True
            )

        # Move to GPU and dtype
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            if pixel_values is not None:
                pixel_values = pixel_values.cuda().to(self.model.dtype)
            if grid_thws is not None:
                grid_thws = grid_thws.cuda()

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=self.model.text_tokenizer.eos_token_id,
                pad_token_id=self.model.text_tokenizer.pad_token_id,
            )

        text = self.model.text_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return self._split_rationale(text)

# ---------------- Postprocess ----------------

def normalize_answer(raw: str, question_type: str) -> str:
    """
    Squash model text to exactly Yes/No, True/False, or HH:MM:SS.
    If nothing parseable, return '' to flag the miss.
    """
    txt = (raw or "").strip()

    if question_type == "Yes/No":
        low = txt.lower()
        if "yes" in low and "no" not in low:
            return "Yes"
        if "no" in low and "yes" not in low:
            return "No"
        if low.startswith("y"):
            return "Yes"
        if low.startswith("n"):
            return "No"
        return ""

    if question_type == "True/False":
        low = txt.lower()
        if "true" in low and "false" not in low:
            return "True"
        if "false" in low and "true" not in low:
            return "False"
        if low.startswith("t"):
            return "True"
        if low.startswith("f"):
            return "False"
        return ""

    if question_type == "Timestamp":
        m = re.search(r"\b(\d{2}):(\d{2})(?::(\d{2}))?\b", txt)
        if m:
            hh, mm, ss = m.group(1), m.group(2), m.group(3) or "00"
            return f"{int(hh):02d}:{int(mm):02d}:{int(ss):02d}"
        # Sometimes the model only outputs "Because 00:00:00" or similar.
        m2 = re.search(r"(?:Because\s+)?(\d{2}):(\d{2})(?::(\d{2}))?\b", txt)
        if m2:
            hh, mm, ss = m2.group(1), m2.group(2), m2.group(3) or "00"
            return f"{int(hh):02d}:{int(mm):02d}:{int(ss):02d}"
        return ""

    # default passthrough
    return txt

# ---------------- Core ----------------

def run(
    qa_json_path: Path,
    videos_dir: Path,
    out_json_path: Path,
    model_id: str = "AIDC-AI/Ovis2.5-2B",
    frames_per_question: int = 12,
    context_radius_sec: float = 2.0,
    max_frames: int = 16,
    add_rationale: bool = True,
    debug: bool = False,
):
    data = json.loads(Path(qa_json_path).read_text())
    frames_cache: Dict[Tuple[str, Tuple[float, ...]], List[Path]] = {}

    # Init backend (Ovis)
    ovis: VisionChatBackend = TransformersOvisStrict(model_id=model_id, debug=debug, max_images=min(16, max_frames))
    if debug:
        print("[DEBUG] Using loader class:", type(ovis).__name__)

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

        # Build Ovis prompt once (no system role needed here; keep instruction in the user text)
        directive = {
            "Yes/No": "Answer strictly Yes or No.",
            "True/False": "Answer strictly True or False.",
            "Timestamp": "Answer strictly with a timestamp in HH:MM:SS."
        }.get(qtype, "Answer concisely.")
        full_q = f"{question}\n\nConstraints: {directive}"

        try:
            raw_text, rationale = ovis.answer(full_q, images, qtype)
            model_answer = normalize_answer(raw_text, qtype)

            # If normalization failed (empty), attempt a tiny retry using fewer frames or text-only
            if not model_answer:
                if debug:
                    print(f"[DEBUG] Empty normalized answer for id={vid_id}, type={qtype}. Raw: '{raw_text}'")
                # Retry 1: use at most 6 frames
                retry_imgs = images[:6]
                raw_text2, rationale2 = ovis.answer(full_q, retry_imgs, qtype)
                model_answer = normalize_answer(raw_text2, qtype)
                if not model_answer:
                    # Retry 2: text-only
                    raw_text3, rationale3 = ovis.answer(full_q, [], qtype)
                    model_answer = normalize_answer(raw_text3, qtype)
                    # Prefer the most informative raw/rationale
                    if raw_text3:
                        raw_text, rationale = raw_text3, rationale3
                    elif raw_text2:
                        raw_text, rationale = raw_text2, rationale2

        except Exception as e:
            raw_text = f"[ERROR] {type(e).__name__}: {e}"
            rationale = ""
            model_answer = "[ERROR]"

        out_item = dict(item)  # keep original fields
        out_item["model_answer"] = model_answer
        out_item["model_raw"] = raw_text
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
    parser = argparse.ArgumentParser(description="Run Ovis 2.5 over Template.json and Template Videos/ to create *_ovis_answers.json")
    parser.add_argument("--qa-json", default="Template.json", type=Path)
    parser.add_argument("--videos-dir", default=Path("Template Videos"), type=Path)
    parser.add_argument("--out-json", default="Template_ovis_answers.json", type=Path)
    parser.add_argument("--model-id", default="AIDC-AI/Ovis2.5-2B")
    parser.add_argument("--frames-per-question", type=int, default=12)
    parser.add_argument("--context-radius-sec", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--no-rationale", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run(
        qa_json_path=args.qa_json,
        videos_dir=args.videos_dir,
        out_json_path=args.out_json,
        model_id=args.model_id,
        frames_per_question=args.frames_per_question,
        context_radius_sec=args.context_radius_sec,
        max_frames=args.max_frames,
        add_rationale=not args.no_rationale,
        debug=args.debug,
    )

if __name__ == "__main__":
    main()

