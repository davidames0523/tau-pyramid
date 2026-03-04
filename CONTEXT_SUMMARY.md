# CONTEXT_SUMMARY.md — Star Stencil + Tau Pyramid Project (Updated)

## System Thesis (target endgame): Route-First Text AI
We aim to maximize novelty leverage by treating language as a *navigation + local computation* problem.

- **Tau Pyramid = memory + address space:** a ternary hierarchical routing substrate that maps context → top-K neighborhood IDs cheaply (CPU/integer-only).
- **Local model = tiny compute:** a compact predictor (Star Stencil head or small param model) that runs only on the routed neighborhood (shortlist), not over full vocab/corpus.
- **Training decomposes into two coupled parts:**
  1) **Routing training:** improve robustness under partial/noisy queries; reduce required trees/probes/beam for a target recall.
  2) **Local prediction training:** given (context, retrieved neighborhood IDs/chunks), predict next token with a shortlist objective (hard negatives from near-miss neighborhoods).
- **Inference loop:** prompt → route (top-K) → local score → sample/generate → update query key/path → repeat.
- **Novelty leverage:** replace global dense attention with cheap hierarchical routing + tiny local compute; “pyramid path tokens” are a candidate alternative tokenization/latent state.

### Next decisive milestone
End-to-end demo: use Tau Pyramid to retrieve neighborhoods for each training step, train a tiny local predictor to beat n-gram baselines at similar CPU budget, and show quality/latency gains versus a small transformer at equal compute.

## Goal
Build a cost-effective, CPU-friendly alternative AI stack using:
- deterministic bitwise “context blender” + learned stencils (Star Stencil AI),
- and a hierarchical ternary routing “training pyramid” (Tau Pyramid; formerly Pyramid Router),
with a 3-phase gcd(K) oscillator as a recurring structural clock.

Key constraints:
- No GPUs required; runs on consumer laptops.
- Prefer single-file Rust binaries (no cargo) for fast iteration.
- Emphasis on deterministic, integer-only arithmetic (bit ops + popcount).

Hardware used: Apple M5 MacBook Pro 14", 16GB, 10-core CPU.  
Corpus: ~2.27GB text (`input.txt`). Typical index: 300k chunk-boundaries.

---

## Track 1: Star Stencil AI (bitplane backprop)
### What it is
- Deterministic GF(2)-style blender `F`:
  - 4096-bit state updated by XOR + rotates + diffusion,
  - driven by a big read-only table (Bruhat-Tits “building” / chambers).
- Small gcd(K) oscillator converging to 3-cycle -> phase `φ ∈ {0,1,2}`.
- Learned stencil head with bitplane weights:
  - Score(token) = bias + Σ 2^p (popcount(state & POS_plane) - popcount(state & NEG_plane)).
  - Weights stored as bitplanes (4 bits => 0..15 magnitudes), updated with bit-sliced inc/dec (integer-only “backprop-ish”).

### Major milestone
On Apple M5 MBP 14" 16GB, corpus ~2.27GB:
- vocab 30k, candidates 96, trained 50M steps
- shortlist accuracy ~64% valid
- output coherent-ish but not fluent; needs better negatives + decoding.

---

## Track 2: Tau Pyramid (hierarchical ternary routing)
### Intuition
Create a ternary “token pyramid”: corpus → 3 arches → 3 subparts → ... → chunks.
Routing should navigate the tree cheaply (≈ log₃), not brute force.
GCD 3-cycle provides a natural ternary rhythm for phases/probes.

### Evolution (v1 → v7-ish)
#### v1 issues
- Tree built by numeric sort of hashes; routing used Hamming distance → geometry mismatch, near 0% recall.
- Early examples had zero context hashes → fixed by collecting examples only after context window warmup.

#### v2/v3
- Keys based on boundary context snapshots.
- Hamming-aware 3-pivot partitioning.
- Added multi-lane keys: 4 lanes per phase → 768-bit keys total.
- Still limited (~20–40% recall) due to index brittleness; beam not helping much.

#### v4 breakthrough: “address system”
Forest + diverse beam + multi-probe:
- Multiple randomized ternary metric trees (“forest”).
- Diverse beam selection avoids collapse into same subtree.
- Multi-probe: rotate phase alignment (0/1/2).
Example milestone:
- `./main input.txt 300000 256 256 100000 32 1 64 8 3`
- recall@beam ≈ **95%** on exact boundary-key retrieval, ~0.83s eval/100k examples on 10 threads.

This validated the foundation: Tau Pyramid can behave like a directory/address system for large corpora.

---

## Learned routing attempts and final alignment
### v5 (hashed-feature router) — worked pre-positional-salting
- Per-node 3-way router using hashed perceptron features, hinge updates.
- Learned-only routing initially collapsed; fixed by:
  - baseline distance backbone + router as penalty (not reward),
  - apply learned routing only at top levels (`learn_levels`), baseline below.
- Plateau reached: noisy recall improved from ~70.5% → ~73–74% and stuck.
- Conclusion at the time: key representation (bag-of-words-ish SimHash) was bottleneck.

### Key nuance upgrade (positional salting) — major improvement
Hypothesis: keys too bag-of-words; need order/position awareness.
Patch implemented in key construction:
- `ctx_queue: VecDeque<u64>` → `VecDeque<(u64,u32)>`
- add `slot_counter` and `slot = slot_counter % context_tokens`
- salt the token contribution by slot/phase/lane:
  - `hv = lane_variant(ph ^ pos_salt(slot, phase, lane), lane)`
- removal uses stored `old_slot` to remove exact prior contribution.
Result:
- Baseline exact recall jumped to ~**99%+**
- Baseline noisy proxy jumped to ~**79–80%**, breaking the ~73% ceiling.

(Important: ensure `pos_salt(...)` remains present and used in both ADD and REMOVE cases.)

### Metric router (per-depth learned metric weights) — replaced hashed-feature router after positional salting
After positional salting, hashed-feature router collapsed (~chance) because partitions became “pure geometry.”
Solution: learn the metric directly.

#### MetricRouter concept
- For each child pivot, compute component distances:
  - `d_{p,l} = ham(q[p][l], pivot[p][l])` for 3 phases × 4 lanes = 12 components (DCOMP)
- Learn per-depth weights `w[depth][component]` (tiny model).
- Use hinge updates based on true child vs rival child:
  - update uses `(d_rival - d_true)` deltas.
- Multiple iterations:
  - Replacing baseline entirely hurt noisy.
  - Switching to **penalty-only** integration fixed it:
    - compute learned costs for 3 children, take `min_l`,
    - apply `penalty = max(0, lcost[child] - min_l) >> SHIFT`,
    - add penalty to baseline distance (never reduces cost).
  - Fixed a major bug in `route_beam_metric_learned` (accidentally nested loops pushing children 3×).

#### Current effective learned routing behavior
- Learned routing now slightly improves exact and noisy recall and can be cost-neutral or faster.
Example (post-fix):
- Baseline exact: 99.31%
- Learned exact: 99.60%
- Baseline noisy: 79.73%
- Learned noisy: 80.49% (and sometimes slightly faster)

#### Tuning knobs
- `learn_levels`: apply learned penalty only at top levels (sweet spot ~4).
- `SHIFT`: penalty strength (right shift). SHIFT=4 generally best; SHIFT=5/6 weakens penalty.
- Forest parameters:
  - trees are critical (dropping 8→4 crushes noisy recall).
  - probes can be reduced to 1 once positional salting + learned penalty are stable.
  - beam controls robustness strongly; leaf size provides “slack.”

---

## Current best-known “consumer” configurations
(All use positional salting keys.)

### Strong + cheaper than early runs
- `trees=8, probes=1, beam=16, leaf=64, learn_levels=4, train_depth=12, SHIFT=4`
Typical:
- exact recall ≈ 99.6%
- noisy ≈ 79.8–80.5%

### Tradeoff experiments (beam/leaf)
- `beam=12, leaf=64, trees=8, probes=1`:
  - exact ≈ 99.2%
  - noisy ≈ 73–74% (learned adds ~+1)
- `beam=8, leaf=64`:
  - noisy drops significantly (~64–66%)
- reducing `leaf_size` below 64 (32, 16) significantly hurts noisy recall (less error-correction slack).

Key conclusion: **beam and leaf_size dominate noisy robustness**; learned metric adds consistent incremental gain.

---

## CLI interface (typical)
`./main input.txt <max_chunks> <chunk_tokens> <context_tokens> <examples> <beam> <token_stride> <leaf_size> <trees> <probes> <train_examples> <train_depth> <epochs>`

Common:
- `max_chunks=300000`
- `chunk_tokens=256`
- `context_tokens=256`
- `token_stride=1`
- `train_depth=12`
- `learn_levels` is an internal constant/variable (commonly 4).

---

## Known pitfalls / fixes
- Rust borrow checker: avoid closures capturing mutable arrays; use helper functions; precompute lengths before mut borrows.
- Example collection must begin only after context window warmed up (avoid all-zero keys).
- In `route_beam_metric_learned`, avoid nested `for ci` duplication: compute lcost once per node, push children once.
- Ensure positional salting is applied consistently in both add/remove.

---

## Next steps
1) Continue cost/quality frontier:
   - Reduce beam while maintaining ~80% noisy (try beam 12/10 with learned penalty).
   - Keep leaf=64 for now; it provides error-correction slack.
2) Add evaluation beyond bit-flip noise:
   - masked queries (50% / 25% keep) to simulate short/partial queries.
3) Consider multi-resolution keys (last 64 + last 256 bands) to improve robustness at lower beam.
4) Integrate Tau Pyramid routing with Star Stencil AI:
   - route to relevant chunks/paths, then use stencil scoring/generation conditioned on retrieved neighborhood.

---