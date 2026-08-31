# scribe_lesson — scribe_lesson (model mistralai/codestral-2508, 1372 tokens)

## system block 1

# ROLE: Scribe (lesson)

You write ONE sentence (at most 20 words) that records the OUTCOME of a finished experiment for future
planning. You are given only harness-measured facts: the hypothesis, the status, the metrics, the
harness decision, the best score afterwards, and (when the experiment ran) the tail of its training
log. Use only those facts.

Decision vocabulary (use it exactly): "promoted" = the attempt became the new champion;
"kept_champion" = the attempt was DISCARDED and the previous champion stays ("kept" refers to the OLD
champion — never write that the attempt was "kept as champion"); "failed" = it never produced a score.

Hard rules:
- State WHAT happened, never WHY. No causal claims ("misaligned", "overfit", "too weak") unless that
  word appears in the facts themselves. A number that is in the facts may be quoted; nothing else.
- Never argue with the harness decision, never invent numbers, never recommend the next step.
- Observations from the training log are welcome when they are literal: "still improving at epoch 40",
  "early-stopped at epoch 7", "crashed twice before running".
Good: "Pairwise BPR: 0.5973 vs 0.6015, kept; still improving at epoch 40 with 4x fewer samples per epoch."
Good: "LightGBM on id features crashed on memory in all 3 debug attempts; no score."
Bad:  "BPR failed, suggesting misalignment with the ranking metric." (causal claim not in the facts)
Output only the sentence — no quotes, no prefix, no newline.

## user

# Facts (measured by the harness)
HYPOTHESIS: Upgrading the FM to a DeepFM (add a 1-hidden-layer MLP over the concatenated field embeddings) and feeding it standardized past-only numerical priors (user/video/author long_view & click rates plus user×tab historical rate) gives the model new, genuinely predictive signal beyond raw id crosses, since these levers were independently validated as the three largest measured wins in this project's history (+0.0078, +0.0037, +0.0035 stacked to 0.6563 from an FM baseline).
CATEGORY: model
RESULT: {"status": "scored", "gauc": 0.6703246197753816, "ndcg5": 0.5371876848672479, "primary": 0.6037561523213147, "runtime_s": 41.5, "error_excerpt": "", "vs_best": "+0.0023"}
DECISION (harness): promoted
BEST PRIMARY AFTER: 0.6037561523213147
TRAINING LOG TAIL (experiment stdout, verbatim):
[champion_equiv] epoch  7 | loss 0.5060 | valid GAUC 0.6488 nDCG@5 0.5282 primary 0.5885 | 0.5s
[champion_equiv] epoch  8 | loss 0.4906 | valid GAUC 0.6498 nDCG@5 0.5289 primary 0.5893 | 0.5s
ABLATION champion_equiv primary=0.5893 gauc=0.6498 ndcg5=0.5289
[deepfm_only] epoch  1 | loss 0.6688 | valid GAUC 0.6193 nDCG@5 0.5171 primary 0.5682 | 0.9s
[deepfm_only] epoch  2 | loss 0.5902 | valid GAUC 0.6461 nDCG@5 0.5274 primary 0.5867 | 0.9s
[deepfm_only] epoch  3 | loss 0.5270 | valid GAUC 0.6529 nDCG@5 0.5298 primary 0.5913 | 0.9s
[deepfm_only] epoch  4 | loss 0.4958 | valid GAUC 0.6566 nDCG@5 0.5314 primary 0.5940 | 1.1s
[deepfm_only] epoch  5 | loss 0.4803 | valid GAUC 0.6570 nDCG@5 0.5313 primary 0.5942 | 0.9s
[deepfm_only] epoch  6 | loss 0.4724 | valid GAUC 0.6567 nDCG@5 0.5314 primary 0.5941 | 0.9s
[deepfm_only] epoch  7 | loss 0.4680 | valid GAUC 0.6569 nDCG@5 0.5318 primary 0.5943 | 1.1s
[deepfm_only] epoch  8 | loss 0.4651 | valid GAUC 0.6557 nDCG@5 0.5312 primary 0.5935 | 0.9s
ABLATION deepfm_only primary=0.5943 gauc=0.6569 ndcg5=0.5318

# TASK
Write the one-sentence lesson (max 20 words) for the facts above. Output only the sentence.

## assistant (response)

DeepFM with standardized priors: 0.6037 vs 0.5893, promoted; early-stopped at epoch 8.
