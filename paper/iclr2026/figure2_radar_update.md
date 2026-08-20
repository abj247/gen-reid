# Figure 2 update: one combined figure (radar + accuracy-vs-size)

`fig_radar_bubble.pdf` is a single image: **left** = radar of per-capability / per-referral accuracy by
model group (proprietary / open-source VLMs / long-video), **right** = accuracy vs parameters (the old
`fig:scaling` panel). It replaces the whole of the old `fig:results2` (consensus + scaling). The
displaced **consensus** panel becomes its own small figure so its "22.9% solved by none" reference still
resolves.

Groups match Table~\ref{tab:leaderboard}: **3 proprietary, 15 open-source VLMs, 4 long-video**
(gemini-3.5-flash excluded). Group means over all axes: proprietary 22.8%, open-source VLMs 18.4%,
long-video 15.8%.

---

## 1. Replace the `fig:results2` figure block with this

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/fig_radar_bubble.pdf}
\caption{Left, accuracy on each capability and referral axis, averaged within the three model groups of Table~\ref{tab:leaderboard}, namely three proprietary models, fifteen open-source VLMs, and four long-video token-compression models, against the \chance{} chance level with shaded difficulty bands. Proprietary models reach the moderate band on every axis, open-source VLMs sit inside them, and long-video models sit inside both near chance, and all three groups are weakest on the same tracking-heavy axes. Right, accuracy against parameter count, where scale is a weak predictor at $r=0.05$ and family dominates.}
\label{fig:results2}
\end{figure}
```

## 2. Add this standalone figure for the displaced consensus panel

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.55\linewidth]{figures/fig_consensus_difficulty.pdf}
\caption{Number of the \nummodels{} models that answer each question, against the independent-chance expectation. No question is solved by all models and 22.9\% by none.}
\label{fig:consensus}
\end{figure}
```

## 3. Fix the two text references (the panels are now one image, so no `fig:scaling`/`fig:radar` labels)

In the "**Scale does not buy the skill**" paragraph, change
`shown in Figure~\ref{fig:scaling}` to
`shown in the right panel of Figure~\ref{fig:results2}`.

Replace the "**Long video specialists trail**" paragraph with:

\textbf{Long video specialists trail.} The four token-compression models occupy the bottom of the board between 10.4\% and 19.2\%, despite reading 128 frames against the default eight. The left panel of Figure~\ref{fig:results2} shows this per capability, where the long-video group sits inside the open-source VLM group on nearly every axis and both trail the proprietary group by a wide margin, yet all three collapse together on the tracking-heavy axes. Section~\ref{sec:coverage} attributes this to a fidelity cost of compression rather than a data or scale artifact.

---

## Notes
- Files: `figures/fig_radar_bubble.pdf` (new combined Figure 2). You can delete the standalone `fig_radar_groups.pdf` if you only use the combined one.
- The composition table (`tab:composition`) is unchanged; its counts are per-question, not per-model.
- The combined image is wide (roughly 2.4:1). At `\linewidth` the radar labels stay legible; if you want it taller, use `width=0.9\linewidth`.
