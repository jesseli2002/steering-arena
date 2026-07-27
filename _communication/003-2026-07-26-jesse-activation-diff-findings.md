author: Jesse Li
agent: Claude Code (Sonnet 5)
date: 2026-07-26
re: 002 (running the diagnostic you proposed)

# Activation diff results: doesn't match either expected pattern

Ran the diagnostic from your `002` message against a local (VastAI, transformers) capture
of the same 24 texts/layers. Neither of the two expected shapes ("mismatch already at
layer 1 -> tokenization" vs "agreement early, drift growing with depth -> numeric
divergence") is quite what showed up.

## What the data actually looks like

Per-layer Euclidean distance between NDIF and local activations, plus that distance
normalized by mean activation norm (relative distance):

```
 layer  ndif_norm  local_norm  abs_dist  rel_dist
     1      3.632       3.584    0.5620    15.72%
     4      5.448       5.372    0.8193    14.34%
     8      8.919       8.846    1.3788    13.11%
    12     11.607      11.496    1.8574    13.22%
    16     13.097      12.930    2.3120    16.49%
    20     15.301      15.276    2.4700    13.47%
    24     19.314      19.226    2.9087    11.96%
```

Absolute distance grows with depth (as expected, since the residual stream's own norm
grows), but **relative distance is flat**, ~12-16% at every layer, no trend. That's
already large by layer 1 (which reads `hidden_states[2]`, i.e. after 2 transformer
blocks) and doesn't compound further. This doesn't fit "numeric divergence accumulating
through the model" (would expect relative error to grow with depth) or "near machine
epsilon" (would expect ~1e-3% scale, not 12-16%) -- it looks like a single divergence
introduced early, then carried forward roughly proportionally by the residual stream.

## Ruled out

1. **Tokenization: NO.** `token_ids_i` compared directly (not through activation math)
   for all 24 texts -- identical in both snapshots.
2. **Batching/padding at read time: NO** (checked `ndif_client.py`). The NDIF snapshot
   uses `ResidualReader.last_resids_layers`, which traces one string at a time -- no
   batch dimension, no padding. The local capture also does one text per forward. Left-
   padding + gather-position bugs (like the kind `batch_last_resids`/
   `compute_scores_batch` guard against elsewhere) can't be the mechanism here, since
   neither snapshot script ever batches.
3. **model_id: matches exactly** (`allenai/Olmo-3-1125-32B`) in both snapshots' meta.
4. **dtype (bf16 vs fp32): NO, empirically.** Re-ran the local capture in float32 and
   diffed it directly against the original bf16 local capture (same GPU, same code,
   only `dtype` changed):

   ```
    layer  abs_mean(bf16 vs fp32)  rel_mean(bf16 vs fp32)
        1                0.0185                    0.52%
        4                0.0377                    0.68%
        8                0.0706                    0.72%
       12                0.1078                    0.78%
       16                0.1337                    0.84%
       20                0.1599                    0.85%
       24                0.1917                    0.80%
   ```

   This *is* the textbook numeric-drift signature -- small (~0.5-0.9%) and growing with
   depth. But it's ~20-30x smaller than the actual NDIF-vs-local gap (12-16%, flat), and
   a different shape (growing vs. flat). Whatever's causing the real gap, it isn't bf16
   vs fp32 precision.

## Still open

- Local snapshot's meta records `attn_implementation='sdpa'`, `torch=2.13.0+cu130`,
  `transformers=5.14.1`; NDIF's snapshot doesn't log any of these three fields, so we
  can't currently tell whether the serving stack differs. Worth adding to
  `snapshot_activations.py`'s meta if that's easy on your end.
- `ResidualReader.__init__` takes `prepend_bos: bool = True` (threaded from
  `settings.prepend_bos`) and stores it as `self.prepend_bos`, but none of
  `last_token_resid` / `batch_last_resids` / `last_resids_layers` appear to reference
  it. Since token ids already match, it isn't currently causing a live divergence
  either way -- but it reads as dead/vestigial config, which is the kind of thing worth
  a second look in case intended logic got dropped somewhere.
- Nobody's compared `hidden_states[0]` (raw embedding output, before any transformer
  block) yet -- current `LAYERS = [1, 4, 8, ...]` all start after block 0. If a future
  capture includes layer 0 and the ~13-16% relative gap is already there, that points
  at the embedding table itself (weights/dtype/scaling); if layer 0 matches and it only
  appears by layer 1, that points at block 0's computation instead. Cheap addition to
  both snapshot scripts if useful.

With tokenization, batching/padding, and dtype all ruled out, the remaining live
suspects are serving-stack differences (attn implementation, transformers/torch
version) or a weight/embedding-level difference -- the layer-0 capture above is the
cheapest way to distinguish those two.
