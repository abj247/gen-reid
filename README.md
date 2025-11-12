# gen-reid


## Run the evals

### 1. Evaluate Qwen 

```
python evaluate_qwen_videos.py \
  --qa-json "Template.json" \
  --videos-dir "Template Videos" \
  --out-json "Template_qwen_answers.json" \
  --backend transformers \
  --model-id Qwen/Qwen2-VL-2B-Instruct \
  --debug
```

### 2. Evaluate Ovis

```
python evaluate_ovis.py \
  --qa-json Template.json \
  --videos-dir "Template Videos" \
  --out-json Template_ovis_answers.json \
  --model-id AIDC-AI/Ovis2.5-2B \
  --frames-per-question 12 \
  --max-frames 16 \
  --debug
```
