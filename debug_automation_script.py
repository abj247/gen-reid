#!/usr/bin/env python3
"""
Automated ReID Benchmark Generation Pipeline V2 - DEBUG VERSION
Temporal tracking questions with relative time references
"""

import os
import json
import time
import re
import base64
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

try:
    import anthropic
except ImportError:
    print("Warning: anthropic not installed. Run: pip install anthropic")

try:  
    import openai
except ImportError:
    print("Warning: openai not installed. Run: pip install openai")

try:
    import google.generativeai as genai
except ImportError:
    print("Warning: google-generativeai not installed. Run: pip install google-generativeai")

try:
    import imageio
except ImportError:
    print("Warning: imageio not installed. Run: pip install imageio")


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    VIDEO_INPUT_DIR = "./videos"
    OUTPUT_DIR = "./benchmarks_v2"
    LOGS_DIR = "./logs"
    
    MAX_VIDEO_DURATION = 600
    MIN_VIDEO_DURATION = 60
    BATCH_SIZE = 10
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 5
    
    DEFAULT_VLM = "gemini"
    
    MIN_QUESTIONS = 18
    REQUIRED_QUESTION_TYPES = [
        "Activity tracking",
        "Location tracking", 
        "Interaction tracking",
        "Object/state changes",
        "Cross-temporal"
    ]
    
    SAVE_JSON = True
    SAVE_MARKDOWN = True


REID_PROMPT_V2 = """TASK: Analyze this video and generate 20 person re-identification (ReID) questions that emphasize TRACKING people across time.

CRITICAL: This video is sampled at 30 frames per second (30fps). All temporal references should be relative to events, NOT absolute timestamps.

KEY REQUIREMENTS:
1. NEVER use absolute timestamps (like "at 1:23")
2. ALWAYS use relative time (like "ten seconds later", "when person stands up")
3. Questions must TRACK the same person across different times
4. Options describe possible ANSWERS (not different people)
5. Use 4-part structure: ID context + ID description + state + question context

QUESTION STRUCTURE (MANDATORY):
**[ID CONTEXT], [ID DESCRIPTION] is [STATE]. [QUESTION CONTEXT], what [QUESTION TYPE]?**

Example:
"At the beginning of the video, the man wearing a gray shirt and eyeglasses is sitting on the couch. Ten seconds later, what is this person doing?"

GENERATE 20 QUESTIONS

Distribution:
- 6 Activity tracking questions (what person is doing at different times)
- 4 Location tracking questions (where person is at different times)
- 4 Interaction tracking questions (who person interacts with)
- 3 Object/state change questions (what person holds or state changes)
- 3 Cross-temporal comparison questions (comparing across times)

Difficulty:
- 6 Easy (short time span, < 10 seconds)
- 8 Medium (medium time span, 10-30 seconds)
- 6 Hard (long span or cross-temporal reasoning)

FORMAT (EXACT):

**Question [N]**: [Full question using 4-part structure]

**Options:**
A. [Possible answer describing activity/location/interaction]
B. [Different possible answer]
C. [Different possible answer]
D. [Different possible answer]

**Correct Answer:** [A/B/C/D]
**Question Type:** [Activity tracking / Location tracking / Interaction tracking / Object-state changes / Cross-temporal]
**Difficulty:** [Easy / Medium / Hard]
**ID Strategy:** [How person was identified: appearance/biometric/activity]
**Temporal Span:** [e.g., "beginning -> 10 sec later" or "early -> late"]

EXAMPLES:

**Question 1**: At the beginning of the video, the man wearing a gray shirt and eyeglasses is sitting on the couch. Ten seconds later, what is this person doing?

**Options:**
A. Standing near the door and talking to another person
B. Still sitting on the couch but now holding a drink
C. Walking toward the kitchen area
D. Sitting on the couch and gesturing with his hands

**Correct Answer:** B
**Question Type:** Activity tracking
**Difficulty:** Easy
**ID Strategy:** Appearance (gray shirt and eyeglasses)
**Temporal Span:** beginning -> 10 seconds later

---

**Question 2**: When three people are sitting on the couch, the woman wearing a blue dress is sitting in the middle. Later when the group is standing near the door, where is this person positioned?

**Options:**
A. On the left side of the group, closest to the door
B. In the middle of the group, between two other people
C. On the right side of the group, furthest from the door
D. Standing separately, away from the main group

**Correct Answer:** A
**Question Type:** Location tracking
**Difficulty:** Medium
**ID Strategy:** Appearance (blue dress) + initial position (middle)
**Temporal Span:** couch scene -> door scene

---

**Question 3**: Early in the video, the man wearing a gray shirt is sitting on the left side of the couch. Later when only two people remain on the couch, what has changed about this person's position?

**Options:**
A. He has moved to the right side of the couch
B. He has moved to the middle position on the couch
C. He remains in the same position on the left
D. He is no longer on the couch

**Correct Answer:** B
**Question Type:** Cross-temporal
**Difficulty:** Hard
**ID Strategy:** Appearance + initial location
**Temporal Span:** early -> later (requires comparison)

CRITICAL CHECKS before submitting:
- NO absolute timestamps in any question
- All questions use relative time with clear origin events
- All questions track the SAME person across time
- Options describe possible answers, NOT different people
- 4-part structure is followed for each question
- Correct distribution: 6/4/4/3/3 by type
- Correct difficulty: 6 easy, 8 medium, 6 hard

BEGIN GENERATION NOW."""


class VideoProcessor:
    def __init__(self, config: Config):
        self.config = config
        
    def get_video_duration(self, video_path: str) -> Optional[float]:
        print(f"[DEBUG] Getting duration for: {video_path}")
        try:
            # Use imageio to get duration without relying on system ffprobe
            with imageio.get_reader(video_path) as reader:
                meta = reader.get_meta_data()
                return float(meta['duration'])
        except Exception as e:
            print(f"Error getting duration for {video_path}: {e}")
            return None
    
    def validate_video(self, video_path: str) -> bool:
        duration = self.get_video_duration(video_path)
        
        if duration is None:
            return False
        
        if duration < self.config.MIN_VIDEO_DURATION:
            print(f"[SKIP] {video_path}: Too short ({duration}s)")
            return False
            
        if duration > self.config.MAX_VIDEO_DURATION:
            print(f"[WARN] {video_path}: Long ({duration}s) - may be expensive")
        
        return True
    
    def get_valid_videos(self, directory: str) -> List[str]:
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(Path(directory).glob(f"*{ext}"))
        
        valid_videos = []
        for video in video_files:
            if self.validate_video(str(video)):
                valid_videos.append(str(video))
                print(f"[OK] Valid: {video.name}")
        
        return valid_videos


class ClaudeHandler:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate_questions(self, video_path: str, prompt: str) -> Optional[str]:
        try:
            print("Uploading video to Claude...")
            
            with open(video_path, "rb") as f:
                video_data = base64.standard_b64encode(f.read()).decode("utf-8")
            
            ext = Path(video_path).suffix.lower()
            media_type_map = {
                '.mp4': 'video/mp4',
                '.mov': 'video/quicktime',
                '.avi': 'video/x-msvideo',
                '.mkv': 'video/x-matroska',
                '.webm': 'video/webm'
            }
            media_type = media_type_map.get(ext, 'video/mp4')
            
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": video_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            return message.content[0].text
            
        except Exception as e:
            print(f"[ERROR] Claude: {e}")
            return None


class GeminiHandler:
    def __init__(self, api_key: str, model: str = "gemini-3-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
    
    def generate_questions(self, video_path: str, prompt: str) -> Optional[str]:
        try:
            print(f"Uploading video to Gemini ({self.model_name})...")
            
            video_file = genai.upload_file(path=video_path)
            
            while video_file.state.name == "PROCESSING":
                print("Processing video...")
                time.sleep(5)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name == "FAILED":
                print("[ERROR] Video processing failed")
                return None
            
            response = self.model.generate_content(
                [video_file, prompt],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=8192,
                    temperature=0.7
                )
            )
            
            return response.text
            
        except Exception as e:
            print(f"[ERROR] Gemini: {e}")
            return None


class OpenAIHandler:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_questions(self, video_path: str, prompt: str) -> Optional[str]:
        try:
            print("Processing with OpenAI...")
            
            # Note: OpenAI GPT-4V does not directly support video
            # Would need to extract frames and send as images
            response = self.client.chat.completions.create(
                model="gpt-4o",
                max_tokens=8192,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"[ERROR] OpenAI: {e}")
            return None


class MockHandler:
    """Mock handler for testing without API keys"""
    def __init__(self):
        print("[DEBUG] Initializing MockHandler")
        pass
    
    def generate_questions(self, video_path, prompt) -> Optional[str]:
        print(f"[DEBUG] MockHandler: Generating dummy questions for {video_path}")
        time.sleep(0.5) 
        
        # Generator for valid question types
        dummy_questions = []
        types = [
            ("Activity tracking", "What is the person doing?", "Walking"),
            ("Activity tracking", "What is the person holding?", "Nothing"),
            ("Activity tracking", "Is the person sitting?", "No"),
            ("Activity tracking", "Is the person running?", "Yes"),
            ("Activity tracking", "Is the person eating?", "No"),
            ("Activity tracking", "Is the person writing?", "Yes"),
            ("Location tracking", "Where is the person?", "Room"),
            ("Location tracking", "Is the person near the door?", "Yes"),
            ("Location tracking", "Is the person outside?", "No"),
            ("Location tracking", "Is the person at the desk?", "Yes"),
            ("Interaction tracking", "Who is the person with?", "Alone"),
            ("Interaction tracking", "Is the person talking?", "No"),
            ("Interaction tracking", "Is the person listening?", "Yes"),
            ("Interaction tracking", "Is the person waving?", "No"),
            ("Object/state changes", "What changed?", "Hat removed"),
            ("Object/state changes", "What changed?", "Door opened"),
            ("Object/state changes", "What changed?", "Light on"),
            ("Cross-temporal", "What changed from start?", "Moved"),
            ("Cross-temporal", "What changed from middle?", "Sat down"),
            ("Cross-temporal", "What changed from end?", "Left")
        ]
        
        for i, (qtype, qtext, ans) in enumerate(types):
            dummy_questions.append(f"\\n**Question {i+1}**: [ID Context] Person [ID Desc] is active. [Context] {qtext}\\n\\n**Options:**\\nA. {ans}\\nB. Incorrect Option 1\\nC. Incorrect Option 2\\nD. Incorrect Option 3\\n\\n**Correct Answer:** A\\n**Question Type:** {qtype}\\n**Difficulty:** Medium\\n**ID Strategy:** Appearance\\n**Temporal Span:** Start -> End\\n")
            
        return "\\n".join(dummy_questions)


class ResponseValidatorV2:
    def __init__(self, config: Config):
        self.config = config
    
    def parse_response(self, response: str) -> Dict:
        questions = []
        parts = response.split("**Question")
        
        for part in parts[1:]:
            try:
                question = self._parse_single_question(part)
                if question:
                    questions.append(question)
            except Exception as e:
                print(f"[WARN] Error parsing question: {e}")
                continue
        
        return {
            "total_questions": len(questions),
            "questions": questions,
            "raw_response": response
        }
    
    def _parse_single_question(self, text: str) -> Optional[Dict]:
        q_match = re.search(r'(\d+)\*\*:\s*(.+?)(?=\n\n|\*\*Options)', text, re.DOTALL)
        if not q_match:
            return None
        
        question_num = q_match.group(1)
        question_text = q_match.group(2).strip()
        
        options = {}
        for letter in ['A', 'B', 'C', 'D']:
            opt_match = re.search(rf'{letter}\.\s*(.+?)(?=\n[A-D]\.|[\n\*])', 
                                text, re.DOTALL)
            if opt_match:
                options[letter] = opt_match.group(1).strip()
        
        answer_match = re.search(r'\*\*Correct Answer:\*\*\s*([A-D])', text)
        qtype_match = re.search(r'\*\*Question Type:\*\*\s*(.+?)(?:\n|$)', text)
        difficulty_match = re.search(r'\*\*Difficulty:\*\*\s*(.+?)(?:\n|$)', text)
        id_strategy_match = re.search(r'\*\*ID Strategy:\*\*\s*(.+?)(?:\n|$)', text)
        temporal_span_match = re.search(r'\*\*Temporal Span:\*\*\s*(.+?)(?:\n|$)', text)
        
        if not all([answer_match, qtype_match, difficulty_match]):
            return None
        
        return {
            "number": int(question_num),
            "question": question_text,
            "options": options,
            "correct_answer": answer_match.group(1),
            "question_type": qtype_match.group(1).strip(),
            "difficulty": difficulty_match.group(1).strip(),
            "id_strategy": id_strategy_match.group(1).strip() if id_strategy_match else "N/A",
            "temporal_span": temporal_span_match.group(1).strip() if temporal_span_match else "N/A"
        }
    
    def validate_questions(self, parsed_data: Dict) -> Dict:
        issues = []
        warnings = []
        questions = parsed_data["questions"]
        
        if len(questions) < self.config.MIN_QUESTIONS:
            issues.append(f"Only {len(questions)} questions (need {self.config.MIN_QUESTIONS})")
        
        for q in questions:
            if self._has_absolute_timestamp(q["question"]):
                issues.append(f"Q{q['number']}: Contains absolute timestamp (FORBIDDEN)")
        
        qtypes = [q["question_type"] for q in questions]
        type_counter = Counter(qtypes)
        
        if "Activity tracking" not in type_counter or type_counter["Activity tracking"] < 4:
            warnings.append("Too few Activity tracking questions")
        
        for q in questions:
            if len(q["options"]) != 4:
                issues.append(f"Q{q['number']}: Only {len(q['options'])} options")
            
            if q["correct_answer"] not in q["options"]:
                issues.append(f"Q{q['number']}: Invalid correct answer")
            
            if not q.get("temporal_span") or q["temporal_span"] == "N/A":
                warnings.append(f"Q{q['number']}: Missing temporal span")
        
        stats = self._calculate_stats(questions)
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "statistics": stats,
            "quality_score": self._calculate_quality_score(questions, issues, warnings)
        }
    
    def _has_absolute_timestamp(self, text: str) -> bool:
        patterns = [
            r'\d{1,2}:\d{2}',
            r'at\s+\d+\s+minutes?',
            r'at\s+\d+:\d+',
        ]
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _calculate_stats(self, questions: List[Dict]) -> Dict:
        qtypes = Counter([q["question_type"] for q in questions])
        difficulties = Counter([q["difficulty"] for q in questions])
        return {
            "by_type": dict(qtypes),
            "by_difficulty": dict(difficulties),
            "total": len(questions)
        }
    
    def _calculate_quality_score(self, questions: List[Dict], 
                                 issues: List, warnings: List) -> float:
        score = 100.0
        score -= len(issues) * 15
        score -= len(warnings) * 5
        
        if len(questions) == 20:
            score += 10
        
        for q in questions:
            if self._has_absolute_timestamp(q["question"]):
                score -= 20
        
        return max(0, min(100, score))


class BatchProcessor:
    def __init__(self, config: Config, vlm_handler):
        self.config = config
        self.vlm_handler = vlm_handler
        self.validator = ResponseValidatorV2(config)
        
        Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
        Path(config.LOGS_DIR).mkdir(exist_ok=True)
    
    def process_video(self, video_path: str) -> Optional[Dict]:
        print(f"[DEBUG] process_video start: {video_path}")
        video_name = Path(video_path).stem
        
        print(f"\\n{'='*60}")
        print(f"Processing: {video_name}")
        print(f"{'='*60}")
        
        for attempt in range(self.config.RETRY_ATTEMPTS):
            try:
                print(f"Attempt {attempt + 1}/{self.config.RETRY_ATTEMPTS}")
                
                response = self.vlm_handler.generate_questions(
                    video_path, REID_PROMPT_V2
                )
                
                print(f"[DEBUG] Response received (Length: {len(response) if response else 0} chars)")
                
                if not response:
                    print("[ERROR] No response from VLM")
                    continue
                
                parsed = self.validator.parse_response(response)
                validation = self.validator.validate_questions(parsed)
                
                print(f"Generated: {parsed['total_questions']} questions")
                print(f"Quality Score: {validation['quality_score']:.1f}/100")
                
                if validation["issues"]:
                    print("Issues found:")
                    for issue in validation["issues"]:
                        print(f"  [X] {issue}")
                
                if validation["warnings"]:
                    print("Warnings:")
                    for warning in validation["warnings"]:
                        print(f"  [!] {warning}")
                
                result = {
                    "video_name": video_name,
                    "video_path": video_path,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "format_version": "v2_temporal_tracking",
                    "parsed_data": parsed,
                    "validation": validation,
                    "attempt": attempt + 1
                }
                
                self._save_result(result, video_name)
                
                if validation["valid"]:
                    print("[SUCCESS] Valid benchmark generated")
                    return result
                else:
                    print("[WARN] Generated but has issues")
                    if attempt < self.config.RETRY_ATTEMPTS - 1:
                        print(f"Retrying in {self.config.RETRY_DELAY}s...")
                        time.sleep(self.config.RETRY_DELAY)
                    
            except Exception as e:
                print(f"[ERROR] {e}")
                if attempt < self.config.RETRY_ATTEMPTS - 1:
                    time.sleep(self.config.RETRY_DELAY)
        
        print("[FAILED] Could not generate valid benchmark")
        return None
    
    def _save_result(self, result: Dict, video_name: str):
        if self.config.SAVE_JSON:
            json_path = Path(self.config.OUTPUT_DIR) / f"{video_name}.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Saved: {json_path}")
        
        if self.config.SAVE_MARKDOWN:
            md_path = Path(self.config.OUTPUT_DIR) / f"{video_name}.md"
            self._save_markdown(result, md_path)
            print(f"Saved: {md_path}")
    
    def _save_markdown(self, result: Dict, path: Path):
        with open(path, 'w') as f:
            f.write(f"# ReID Benchmark V2 (Temporal Tracking): {result['video_name']}\\n\\n")
            f.write(f"**Generated:** {result['timestamp']}\\n")
            f.write(f"**Format Version:** {result['format_version']}\\n")
            f.write(f"**Quality Score:** {result['validation']['quality_score']:.1f}/100\\n\\n")
            
            stats = result['validation']['statistics']
            f.write("## Statistics\\n\\n")
            f.write(f"- Total Questions: {stats['total']}\\n")
            f.write(f"- By Type: {stats['by_type']}\\n")
            f.write(f"- By Difficulty: {stats['by_difficulty']}\\n\\n")
            
            if result['validation']['issues']:
                f.write("## Issues\\n\\n")
                for issue in result['validation']['issues']:
                    f.write(f"- [X] {issue}\\n")
                f.write("\\n")
            
            if result['validation']['warnings']:
                f.write("## Warnings\\n\\n")
                for warning in result['validation']['warnings']:
                    f.write(f"- [!] {warning}\\n")
                f.write("\\n")
            
            f.write("## Questions\\n\\n")
            for q in result['parsed_data']['questions']:
                f.write(f"### Question {q['number']}\\n\\n")
                f.write(f"{q['question']}\\n\\n")
                f.write("**Options:**\\n\\n")
                for letter in ['A', 'B', 'C', 'D']:
                    if letter in q['options']:
                        f.write(f"{letter}. {q['options'][letter]}\\n\\n")
                f.write(f"**Correct Answer:** {q['correct_answer']}\\n\\n")
                f.write(f"**Question Type:** {q['question_type']}\\n\\n")
                f.write(f"**Difficulty:** {q['difficulty']}\\n\\n")
                f.write(f"**ID Strategy:** {q.get('id_strategy', 'N/A')}\\n\\n")
                f.write(f"**Temporal Span:** {q.get('temporal_span', 'N/A')}\\n\\n")
                f.write("---\\n\\n")
    
    def process_batch(self, video_paths: List[str]) -> List[Dict]:
        results = []
        
        for i, video_path in enumerate(video_paths):
            print(f"\\n[{i+1}/{len(video_paths)}]")
            result = self.process_video(video_path)
            results.append(result)
            
            if i < len(video_paths) - 1:
                time.sleep(2)
        
        return results


def save_consolidated_benchmark(results: List[Dict], output_dir: str):
    """Save all videos and questions to a single consolidated JSON file"""
    consolidated = {
        "benchmark_name": "ReID_Temporal_Tracking_V2",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_videos": 0,
        "total_questions": 0,
        "videos": []
    }
    
    for result in results:
        if result is None:
            continue
        
        video_entry = {
            "video_id": result["video_name"],
            "video_path": result["video_path"],
            "quality_score": result["validation"]["quality_score"],
            "questions": []
        }
        
        for q in result["parsed_data"]["questions"]:
            question_entry = {
                "question_number": q["number"],
                "question": q["question"],
                "question_type": q["question_type"],
                "difficulty": q["difficulty"],
                "options": q["options"],
                "correct_answer": q["correct_answer"],
                "id_strategy": q.get("id_strategy", "N/A"),
                "temporal_span": q.get("temporal_span", "N/A")
            }
            video_entry["questions"].append(question_entry)
        
        consolidated["videos"].append(video_entry)
        consolidated["total_questions"] += len(video_entry["questions"])
    
    consolidated["total_videos"] = len(consolidated["videos"])
    
    output_path = Path(output_dir) / "reid_benchmark_all.json"
    with open(output_path, 'w') as f:
        json.dump(consolidated, f, indent=2)
    
    print(f"\\nSaved consolidated benchmark: {output_path}")
    print(f"  - Videos: {consolidated['total_videos']}")
    print(f"  - Total Questions: {consolidated['total_questions']}")


def get_vlm_handler(vlm_name: str, config: Config):
    print(f"[DEBUG] Requesting VLM Handler: {vlm_name}")
    vlm_name = vlm_name.lower()
    
    if vlm_name == "mock":
        return MockHandler()
    
    if vlm_name == "claude":
        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set")
        return ClaudeHandler(config.ANTHROPIC_API_KEY)
    
    elif vlm_name == "gemini" or vlm_name.startswith("gemini"):
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set")
        model = "gemini-2.5-pro"
        if "-" in vlm_name:
            model = vlm_name
        return GeminiHandler(config.GEMINI_API_KEY, model)
    
    elif vlm_name == "openai" or vlm_name == "gpt":
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")
        return OpenAIHandler(config.OPENAI_API_KEY)
    
    else:
        raise ValueError(f"Unknown VLM: {vlm_name}. Supported: claude, gemini, openai, mock")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Automated ReID Benchmark Generation V2 (Temporal Tracking) - DEBUG"
    )
    # Changed default to Template Videos for testing
    parser.add_argument("--input-dir", default="./Template Videos")
    parser.add_argument("--output-dir", default="./benchmarks_v2_debug")
    parser.add_argument("--batch-size", type=int, default=10)
    # Changed default to mock
    parser.add_argument("--vlm", default="mock", 
                        help="VLM to use: claude, gemini, gemini-2.5-pro, openai, mock")
    
    args = parser.parse_args()
    
    config = Config()
    config.VIDEO_INPUT_DIR = args.input_dir
    config.OUTPUT_DIR = args.output_dir
    config.BATCH_SIZE = args.batch_size
    
    print("ReID Benchmark Automation V2 (Temporal Tracking) - DEBUG MODE")
    print(f"Input: {config.VIDEO_INPUT_DIR}")
    print(f"Output: {config.OUTPUT_DIR}")
    print(f"VLM: {args.vlm}")
    print()
    
    video_processor = VideoProcessor(config)
    
    print(f"[DEBUG] Searching for videos in: {config.VIDEO_INPUT_DIR}")
    if not os.path.exists(config.VIDEO_INPUT_DIR):
        print(f"[ERROR] Input directory does not exist: {config.VIDEO_INPUT_DIR}")
        print(f"[INFO] Current working directory: {os.getcwd()}")
        return

    video_paths = video_processor.get_valid_videos(config.VIDEO_INPUT_DIR)
    
    if not video_paths:
        print("[ERROR] No valid videos found!")
        return
    
    print(f"[OK] Found {len(video_paths)} valid videos")
    for i, v in enumerate(video_paths):
        duration = video_processor.get_video_duration(v)
        duration_str = f"{duration:.2f}s" if duration is not None else "Unknown"
        print(f"  {i+1}. {Path(v).name} (Duration: {duration_str})")
    print()
    
    print("-" * 60)
    print("MANUAL BREAK: Videos loaded successfully.")
    print("The script is currently configured to use MOCK VLM (no real API calls).")
    user_input = input("Press ENTER to continue with mock generation, or 'q' to quit: ")
    if user_input.lower() == 'q':
        print("[INFO] Exiting as requested.")
        return
    print("-" * 60)
    
    vlm_handler = get_vlm_handler(args.vlm, config)
    
    batch_processor = BatchProcessor(config, vlm_handler)
    
    total_results = []
    for i in range(0, len(video_paths), config.BATCH_SIZE):
        batch = video_paths[i:i + config.BATCH_SIZE]
        batch_results = batch_processor.process_batch(batch)
        total_results.extend(batch_results)
    
    successful = [r for r in total_results if r and r['validation']['valid']]
    print(f"\\n{'='*60}")
    print(f"Complete: {len(successful)}/{len(total_results)} successful")
    print(f"{'='*60}")
    
    # Save consolidated benchmark JSON with all videos
    save_consolidated_benchmark(total_results, config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
