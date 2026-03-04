# CONTEXT_SUMMARY.md — Star Stencil + Pyramid Router Project

## Goal
Build a cost-effective, CPU-friendly alternative AI stack using:
- deterministic bitwise “context blender” + learned stencils (Star Stencil AI),
- and a hierarchical ternary routing “training pyramid” (Pyramid Router),
with a 3-phase gcd(K) oscillator as a recurring structural clock.

Key constraints:
- No GPUs required; should run on consumer laptops.
- Prefer single-file Rust binaries (no cargo) for fast iteration.
- Heavy emphasis on deterministic, integer-only arithmetic (bit ops + popcount).

---

## Track 1: Star Stencil AI (bitplane backprop)
### What it is
- A deterministic GF(2)-style blender `F`:
  - 4096-bit state updated by XOR + rotates + diffusion,
  - driven by a big read-only table (Bruhat-Tits “building” / chambers).
- A small gcd(K) oscillator that converges to a 3-cycle -> phase `φ ∈ {0,1,2}`.
- A learned stencil head with bitplane weights:
  - Score(token) = bias + Σ 2^p (popcount(state & POS_plane) - popcount(state & NEG_plane)).
  - Weights stored as bitplanes (4 bits => 0..15 magnitudes), updated with bit-sliced increment/decrement (integer-only “backprop-ish”).

### Major milestone
On Apple M5 MBP 14" 16GB, corpus ~2.27GB:
- vocab 30k, candidates 96, trained 50M steps
- shortlist accuracy ~64% valid
- generation coherent-ish but not fluent (still needs better negatives/decoding).

---

## Track 2: Pyramid Router (hierarchical routing)
### Intuition
Create a ternary “token pyramid”: corpus -> 3 arches -> 3 subparts -> ... down to chunks.
Routing should navigate the tree quickly (log_3 scale), not brute force.
GCD 3-cycle phase provides a natural ternary structural rhythm.

### v1 issues
- Built tree by sorting hashes numerically; routed by Hamming distance -> mismatch, ~0% recall.
- Early examples had zero context hashes -> fixed by warmup.

### v2
- Keys were based on boundary context snapshots.
- Tree built with Hamming-aware 3-pivot partitioning.
- Recall improved (~20–30%) but beam didn’t help much.

### v3
- Multi-lane keys: 4 lanes per phase => 768-bit total keys.
- Matched routing metric.
- Still limited (~24–38% recall depending on beam/leaf) due to index brittleness.

### v4 breakthrough
Forest + diverse beam + multi-probe:
- Build multiple randomized ternary metric trees (forest).
- Diverse beam selection prevents path collapse.
- Multi-probe: rotate phase alignment (0/1/2).
On Apple M5:
- `./main input.txt 300000 256 256 100000 32 1 64 8 3`
- recall@beam ≈ **95%** on exact boundary-key retrieval (100k examples), ~0.83s eval on 10 threads.
This validates the “pyramid as address system” foundation (retrieval stage).

### v5 learned routers
Add per-node learned 3-way routers (hashed-feature perceptron, integer-only):
- features derived from query key -> hashed indices + signs.
- hinge margin updates at each node along teacher-forced true path.
Initial learned-only routing collapsed (scores uncalibrated). Fixed with hybrid cost:
- baseline distance backbone + router as small penalty.
Then achieved:
- Exact-key recall ~baseline (or slightly better).
- Noisy-query proxy recall improved: baseline ~70.5% -> learned ~73% (stable plateau).
Performance tuning:
- Apply learned routing only at top levels (`learn_levels`), baseline below.
- Sweet spot found: `learn_levels=4` gives noisy gain with low overhead.

Ablations:
- trees=8/probes=3/beam=16: best accuracy.
- reducing trees to 4 crushed recall (exact and noisy); trees are important.

Training augmentations attempted:
- rotate phases, bit-flip noise, dropout-style masking, structured masking.
- Improved noisy recall slightly (to ~73.7%), but plateau persisted -> suggests key representation is bottleneck.

### Current v7 direction (key nuance upgrade)
Hypothesis: SimHash keys too bag-of-words; need order/position awareness to exceed ~73% noisy recall.
Patch implemented:
- Add positional salting in key construction:
  - ctx_queue changed from `VecDeque<u64>` to `VecDeque<(u64,u32)>`.
  - Each update assigns slot = slot_counter % context_tokens and salts hash by slot/phase/lane.
  - Removal uses stored old_slot so adds/removes are consistent.
- Helper added:
  - `pos_salt(slot, phase, lane) -> u64`.
This is expected to improve key robustness and break the ~73% ceiling.

---

## Commands and proven configs
### Exact addressability (routing foundation)
- v4 best: `beam=32 leaf=64 trees=8 probes=3` => ~95% recall.

### Learned routing speed/robustness tradeoff (sweet spot)
- `learn_levels=4` best ROI (small speed hit, keeps noisy gain).

Typical run used:
`./main input.txt 300000 256 256 200000 16 1 64 8 3 200000 12 1`
g
---

## Known pitfalls / fixes
- Rust borrow checker issues fixed by avoiding closures capturing mutable arrays; use helper functions and precompute lengths before borrowing mutably.
- Early examples had zero hashes -> start collecting examples only after context window warmed up.

---

## Next steps
1) Re-run after positional hashing patch:
   - Measure baseline noisy recall and learned noisy recall.
   - Expect baseline noisy to rise and learned noisy to rise further.
2) If improved, consider:
   - multi-resolution keys (separate bands for last 64/128/256 tokens),
   - add n-gram hashing (bigrams) cheaply.
3) If still plateaued, train routers with richer features:
   - include pivot-distance features (distance-to-child pivots) as router inputs.
4) Ultimately integrate routing pyramid with Star Stencil:
   - route to relevant chunks, then use stencil scoring/generation conditioned on retrieved paths.

Hardware: Apple M5 MacBook Pro 14", 16GB, 10-core CPU. Corpus size used: 2.27GB text, indexed 300k chunks.