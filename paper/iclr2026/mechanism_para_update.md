# Mechanism section update (combined layer-wise figure)

`fig:layerwise` is now the combined figure (cosine similarity + logit lens on top, depth
attention on the bottom = `fig_mechanism_depth.pdf`). Two things change: the paragraph that
references it, and the figure block/caption. Style kept consistent with the surrounding text
(no em dashes, no scare quotes).

---

## 1. Replace the last paragraph with this

We then map attention onto the native sixteen by sixteen vision patch grid to see where inside each frame the model looks. Figure~\ref{fig:attn} depicts the attention maps to show relevancy on the peakiest correct and wrong examples. Correct answers place hot spots on the referent, while wrong answers scatter across background and outro cards, which is the spatial counterpart of the frame-level recency bias. Figure~\ref{fig:tsne} embeds a 64-frame pool per video with a CLIP encoder and marks the clusters an 8-frame and a 32-frame uniform sample reach, which visualizes the coverage gap. Figure~\ref{fig:layerwise} then follows the same failure inward through the language model. Panel (a) tracks the cosine similarity between visual-token representations at reduced budgets and the 32-frame reference across layers, and the dip in middle layers shows where missing coverage enters the representation. Panel (c) reads the answer-token attention on the referent frame at increasing depth, where it starts diffuse, sharpens onto the referent by the middle layers, and then stays on the person for correct answers but drifts to the background for wrong ones. Panel (b) applies a logit lens to the answer position and shows the correct option gains probability only in the last few layers when the model succeeds, while in failures it never rises and the chosen wrong option does, so the decision is committed late and downstream of evidence that is already lost. Together the probes trace one causal chain, namely sparse sampling misses scene clusters, recency-biased attention underreads the frames that arrive, spatial relevancy lands off the referent and drifts further off it with depth, the representation gap opens in the middle layers, and the answer is fixed in the last layers with the correct option no longer recoverable. All of this runs on one representative model, so it explains a mechanism rather than establishes a population law, and the chain is read as an existence proof that unread frames, not unavailable frames, cap accuracy. Attention mass is read from generated answer tokens back to visual tokens and averaged over heads, which understates any single sharp head but is robust to head-level noise.

---

## 2. Replace the figure block with this

```latex
\begin{figure}[tp]
\centering
\includegraphics[width=\linewidth]{figures/fig_mechanism_depth.pdf}
\caption{Following the failure inward on InternVL3-8B. (a) Cosine similarity of visual-token representations at reduced frame budgets against the 32-frame reference, layer by layer, where the middle-layer dip marks where missing coverage enters the representation. (b) Logit-lens readout of the option probabilities at each layer, averaged over questions the model answers right and wrong, where the correct option becomes probable only in the last few layers when the model succeeds and in failures never rises while the chosen wrong option does. (c) Answer-token spatial attention on the referent frame at a shallow, a middle, and a deep layer, sharpening onto the referent for the correct answer and drifting to the background for the wrong one.}
\label{fig:layerwise}
\end{figure}
```

Notes:
- The include now uses `fig_mechanism_depth.pdf` at `\linewidth` (was `fig_layerwise.pdf` at `0.5\linewidth`). If you prefer to keep the filename `fig_layerwise.pdf`, just rename `fig_mechanism_depth.pdf` to it and leave the width at `\linewidth`.
- Panel letters follow the composite layout: (a) top-left cosine curve, (b) top-right logit lens, (c) bottom depth-attention strip.
