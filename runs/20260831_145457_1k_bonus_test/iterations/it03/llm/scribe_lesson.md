# scribe_lesson — scribe_lesson (model mistralai/codestral-2508, 1141 tokens)

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
HYPOTHESIS: Adding user historical long_view rates and item/author auxiliary feedback rates (click, like) as past-only numerical features will provide DeepFM's MLP with rich interaction surfaces, allowing it to learn non-linear personalized generosity-vs-quality crosses and raising the ranking metric.
CATEGORY: feature
RESULT: {"status": "scored", "gauc": 0.676647330011664, "ndcg5": 0.62166981631555, "primary": 0.6491585731636069, "runtime_s": 1856.8, "error_excerpt": "", "vs_best": "+0.0003"}
DECISION (harness): promoted
BEST PRIMARY AFTER: 0.6491585731636069
TRAINING LOG TAIL (experiment stdout, verbatim):
[pure_bpr] early stop at epoch 5
wrote preds_val.csv: 2524980 rows for split=valid in 1564s
ABLATION pure_bpr_single primary=0.6353 gauc=0.6730 ndcg5=0.5976
ABLATION pure_bpr_ensemble primary=0.6492 gauc=0.6766 ndcg5=0.6217
[champion_equiv] Total within-user pairs: 1332063
[champion_equiv] epoch  1 | loss 0.6007 | valid GAUC 0.6684 nDCG@5 0.6106 primary 0.6395 | 53.0s
[champion_equiv] epoch  2 | loss 0.4851 | valid GAUC 0.6695 nDCG@5 0.5835 primary 0.6265 | 53.5s
[champion_equiv] epoch  3 | loss 0.4172 | valid GAUC 0.6659 nDCG@5 0.5569 primary 0.6114 | 63.0s
[champion_equiv] epoch  4 | loss 0.3667 | valid GAUC 0.6620 nDCG@5 0.5371 primary 0.5996 | 65.1s
[champion_equiv] epoch  5 | loss 0.3320 | valid GAUC 0.6595 nDCG@5 0.5323 primary 0.5959 | 55.1s
[champion_equiv] early stop at epoch 5
ABLATION champion_equiv primary=0.6395 gauc=0.6684 ndcg5=0.6106

# TASK
Write the one-sentence lesson (max 20 words) for the facts above. Output only the sentence.

## assistant (response)

Adding user historical long_view rates and item/author auxiliary feedback rates as past-only numerical features promoted the ranking metric to 0.6492.
