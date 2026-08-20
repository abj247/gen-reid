#!/usr/bin/env python
"""Prompt shapes for InternVL3: the collapsed block we have always used vs. one block per frame.

Every InternVL number in this project was produced by a prompt with NO frame delimiters, and
this module exists to make that testable rather than to retract it.

`evaluate_vlm_bm.py:1267` calls `model.chat(...)` without `num_patches_list`. In the snapshot's
`modeling_internvl_chat.py` that defaults to `[pixel_values.shape[0]]` -- a list of ONE -- and
the expansion loop then does `query.replace('<image>', image_tokens, 1)` exactly once. So for a
32-frame prompt "Frame 1: <image>\\nFrame 2: <image>\\n...", the FIRST placeholder swallows all
32 * 256 = 8192 IMG_CONTEXT tokens as a single undelimited <img>...</img> block, and the other
31 placeholders survive into the tokenizer as the literal text "<image>". The model sees one
visual blob, plus 31 frame labels pointing at nothing. `LetterScorer.build_query`
(analysis3/readout/contrastive_decode.py:134-144) reproduces that shape, so the read-out harness
is CONSISTENT with the published path (uniform32 = 31.03 over all 780 core keys) -- both are
collapsed. (vtm_eval.py's docstring still says 30.94; the JSONL it wrote measures 31.03 on 780/780
keys, and analyze_gates.py recomputes from that file rather than from any written-down constant,
so the gate rules are unaffected by the stale figure.)

InternVL's own video interface instead passes `num_patches_list=[1]*N`, giving each frame its own
<img>...</img> block. Gate A is that A/B, and it needs both prompt shapes to be built by the same
code path so the only difference is the delimiters:

    build_query_flat    today's shape, byte-identical to LetterScorer.build_query
    build_query_blocks  one block per frame, RAGGED token counts allowed (Gate B wants
                        [64]*128 + [256]*32 -- cheap tokens everywhere, full tokens near evidence)
    score_prebuilt      LetterScorer.score() with the query supplied by the caller

Nothing here loads a model: a `scorer` is any object with LetterScorer's attributes
(`model`, `tok`, `get_conv_template`, `letter_ids`), so the query builders are testable on CPU
with a tokenizer alone.
"""
import os
import sys

import os as _os
from pathlib import Path as _Path
ROOT = _os.environ.get("PERSISTQA_ROOT") or str(_Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_READOUT = os.path.join(ROOT, "analysis3", "readout")
if _READOUT not in sys.path:
    sys.path.insert(0, _READOUT)

from contrastive_decode import IMG_CTX, IMG_END, IMG_START  # noqa: E402  (re-exported)

PLACEHOLDER = "<image>"

__all__ = ["IMG_START", "IMG_END", "IMG_CTX", "PLACEHOLDER",
           "build_query_flat", "build_query_blocks", "score_prebuilt"]


def _templated(scorer, question):
    """The chat-template'd query, still holding its literal '<image>' placeholders.

    Both builders go through here so the two prompt shapes can never drift in anything except
    the placeholder expansion -- which is the one thing Gate A is measuring.
    """
    if PLACEHOLDER not in question:
        question = PLACEHOLDER + "\n" + question
    template = scorer.get_conv_template(scorer.model.template)
    template.system_message = scorer.model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    return template.get_prompt()


def build_query_flat(scorer, question, n_frames, tokens_per_frame):
    """Today's collapsed prompt: ONE <img>...</img> holding n_frames*tokens_per_frame contexts.

    Byte-identical to `LetterScorer.build_query(question, n_frames)` when
    `scorer.model.num_image_token == tokens_per_frame`. The count is taken from the argument
    rather than from the model so callers do not have to wrap this in TokenCountOverride.

    Placeholders 2..N are left as literal text, deliberately: that is the shape every InternVL
    number in this project came from, and the flat arm has to reproduce it exactly.
    """
    if n_frames < 1 or tokens_per_frame < 1:
        raise ValueError(f"n_frames and tokens_per_frame must be >= 1, "
                         f"got {n_frames} and {tokens_per_frame}")
    query = _templated(scorer, question)
    block = IMG_START + IMG_CTX * (tokens_per_frame * n_frames) + IMG_END
    return query.replace(PLACEHOLDER, block, 1)


def build_query_blocks(scorer, question, tokens_per_frame_list):
    """One <img>...</img> block per frame, i.e. InternVL's own `num_patches_list=[1]*N` shape.

    The k-th placeholder gets `tokens_per_frame_list[k]` IMG_CONTEXT tokens, so counts may be
    RAGGED across frames. No literal '<image>' survives.
    """
    counts = [int(t) for t in tokens_per_frame_list]
    if not counts:
        raise ValueError("tokens_per_frame_list is empty; a query needs at least one image block")
    bad = [i for i, t in enumerate(counts) if t < 1]
    if bad:
        raise ValueError(f"tokens_per_frame_list has non-positive counts at indices {bad[:8]}")

    query = _templated(scorer, question)
    n_ph = query.count(PLACEHOLDER)
    if n_ph != len(counts):
        raise ValueError(
            f"prompt has {n_ph} '{PLACEHOLDER}' placeholders but tokens_per_frame_list has "
            f"{len(counts)} entries; every frame needs exactly one placeholder "
            f"(build the prompt with one '{PLACEHOLDER}' per frame, e.g. "
            f"''.join(f'Frame {{i+1}}: {PLACEHOLDER}\\n' for i in range(n)))")

    for t in counts:
        query = query.replace(PLACEHOLDER, IMG_START + IMG_CTX * t + IMG_END, 1)
    assert PLACEHOLDER not in query, "placeholder survived block expansion"
    return query


def score_prebuilt(scorer, query, vit_flat):
    """`LetterScorer.score()` on a caller-built query: returns (raw_logits[8], logprobs[8]).

    `vit_flat` is [total_tokens, C], flattened in the SAME order the blocks appear in `query`
    (frame 0's tokens first, then frame 1's, ...). It is spliced onto the IMG_CONTEXT positions
    exactly as `InternVLChatModel.generate` does, then ONE forward pass with logits_to_keep=1 --
    generate() materialises all ~9.5k positions x 152k vocab (11 GB) and OOMs a 48 GB card.

    The img-context count check is kept as a hard error, not an assert: with ragged block sizes a
    mismatch is easy to introduce and would otherwise splice the bank silently misaligned, which
    reads as a plausible-looking accuracy drop instead of a crash.
    """
    torch = getattr(scorer, "torch", None)
    if torch is None:
        import torch

    if vit_flat.dim() != 2:
        raise ValueError(f"vit_flat must be [total_tokens, C], got shape {tuple(vit_flat.shape)}"
                         f"; flatten the bank with .reshape(-1, C) in block order first")

    mi = scorer.tok(query, return_tensors="pt")
    lm = scorer.model.language_model
    emb_layer = lm.get_input_embeddings()
    input_ids = mi["input_ids"].to(emb_layer.weight.device)
    with torch.no_grad():
        inp = emb_layer(input_ids)
        B, N, C = inp.shape
        inp = inp.reshape(B * N, C)
        flat = input_ids.reshape(B * N)
        sel = flat == scorer.model.img_context_token_id
        n_ctx, n_vis = int(sel.sum()), int(vit_flat.shape[0])
        if n_ctx != n_vis:
            raise ValueError(f"img-context/visual-token mismatch: prompt expects {n_ctx} "
                             f"{IMG_CTX} tokens, vit_flat provides {n_vis}")
        inp[sel] = vit_flat.reshape(-1, C).to(inp.dtype).to(inp.device)
        inp = inp.reshape(B, N, C)
        attn = mi["attention_mask"].to(inp.device)
        try:
            out = lm(inputs_embeds=inp, attention_mask=attn, use_cache=False, logits_to_keep=1)
        except TypeError:
            out = lm(inputs_embeds=inp, attention_mask=attn, use_cache=False, num_logits_to_keep=1)
        raw = out.logits[0, -1].float()
        lsm = torch.log_softmax(raw, dim=-1)
        lg = [float(raw[i]) for i in scorer.letter_ids]
        lp = [float(lsm[i]) for i in scorer.letter_ids]
    del inp, out, raw, lsm
    return lg, lp
