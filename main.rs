// Pyramid Router v5 (Forest + Multi-Probe + Diverse Beam + LEARNED 3-way Routers per node)
// Single-file Rust, no Cargo.
//
// What v5 adds beyond v4:
//  - Each internal node gets a tiny learned 3-class router (hashed-feature perceptron).
//  - Training is teacher-forced along the true path (which child range contains the target id).
//  - Routing can run in two modes:
//      (A) baseline similarity routing (like v4)
//      (B) learned routing (routers score which child to expand)
//
// Why this is the real “training pyramid” step:
//  - v4 proved: approximate routing can find the exact address with high recall.
//  - v5 starts learning routers so routing can generalize when queries aren’t exact keys
//    (we include a noisy-query evaluation as a first proxy).
//
// Usage:
//   ./main <path> <max_chunks> <chunk_tokens> <context_tokens> <examples>
//          <beam> <token_stride> <leaf_size> <trees> <probes>
//          <train_examples> <train_depth> <epochs>
//
// Example (your strong baseline params + learning):
//   ./main input.txt 300000 256 256 100000 32 1 64 8 3 100000 12 1

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

const PHASES: usize = 3;
const LANES: usize = 4; // 4×64 bits per phase => 256 bits/phase => 768 total bits

// Router settings (small + fast + memory-safe)
const FEAT_DIM: usize = 512; // power of two
const FEATS_PER_QUERY: usize = 96;
const ROUTER_MARGIN: i32 = 8;
const DCOMP: usize = PHASES * LANES; // components per key
const METRIC_MARGIN: i32 = 8;
const W_MIN: i16 = -32;
const W_MAX: i16 =  32;
const SHIFT: i32 = 6;

// Noise test (generalization proxy): flip a small number of bits deterministically
const NOISE_MASK_BITS: u32 = 2; // higher = fewer flips. 2 => ~1/4 bits flipped in mask before AND density, but we sparsify.
const NOISE_WORDS: usize = 2;   // number of 64-bit words per phase-lane to perturb lightly

// -------------------- gcd(K) 3-cycle oscillator --------------------

#[inline(always)]
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

#[inline(always)]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

struct GcdThreeCycleOsc {
    a: u64,
    b: u64,
    k: u64,
}
impl GcdThreeCycleOsc {
    fn new(seed: u64, k: u64) -> Self {
        let a = splitmix64(seed) | 1;
        let b = splitmix64(seed ^ 0xD1B54A32D192ED03) | 1;
        Self { a, b, k: k.max(1) }
    }
    #[inline(always)]
    fn step_and_phase(&mut self) -> u8 {
        let next = gcd_u64(self.a, self.b).wrapping_add(self.k);
        self.a = self.b;
        self.b = next;

        let kp1 = self.k + 1;
        let k2p1 = 2 * self.k + 1;
        if self.a == kp1 && self.b == kp1 { 0 }
        else if self.a == kp1 && self.b == k2p1 { 1 }
        else if self.a == k2p1 && self.b == kp1 { 2 }
        else { ((self.a ^ self.b ^ self.k.wrapping_mul(0x9E3779B97F4A7C15)) % 3) as u8 }
    }
}

// -------------------- token hashing --------------------

#[inline(always)]
fn is_word_byte(b: u8) -> bool {
    (b'A'..=b'Z').contains(&b)
        || (b'a'..=b'z').contains(&b)
        || (b'0'..=b'9').contains(&b)
        || b == b'_' || b == b'\''
}

#[inline(always)]
fn token_hash_from_bytes(seed: u64, bytes: &[u8]) -> u64 {
    // FNV-1a-ish then splitmix for diffusion
    let mut h: u64 = 1469598103934665603u64 ^ seed;
    for &b in bytes {
        let lb = if (b'A'..=b'Z').contains(&b) { b + 32 } else { b };
        h ^= lb as u64;
        h = h.wrapping_mul(1099511628211u64);
    }
    splitmix64(h)
}

#[inline(always)]
fn phase_variant(h: u64, phase: u8) -> u64 {
    match phase {
        0 => h,
        1 => h.rotate_left(21) ^ 0x9E3779B97F4A7C15u64,
        _ => h.rotate_left(43) ^ 0xBF58476D1CE4E5B9u64,
    }
}

const LANE_SALTS: [u64; LANES] = [
    0x243F_6A88_85A3_08D3u64,
    0x1319_8A2E_0370_7344u64,
    0xA409_3822_299F_31D0u64,
    0x082E_FA98_EC4E_6C89u64,
];

#[inline(always)]
fn lane_variant(h: u64, lane: usize) -> u64 {
    splitmix64(h ^ LANE_SALTS[lane])
}

// -------------------- simhash core --------------------

#[inline(always)]
fn add_simhash(counters: &mut [i32; 64], h: u64, sign: i32) {
    for bit in 0..64 {
        let b = ((h >> bit) & 1) as i32;
        counters[bit] += sign * ((b << 1) - 1); // +1 if bit=1 else -1
    }
}

#[inline(always)]
fn counters_to_u64(counters: &[i32; 64]) -> u64 {
    let mut out = 0u64;
    for bit in 0..64 {
        if counters[bit] > 0 {
            out |= 1u64 << bit;
        }
    }
    out
}

#[inline(always)]
fn ham(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

// -------------------- multi-lane key --------------------

#[derive(Clone, Copy)]
struct MultiKey {
    h: [[u64; LANES]; PHASES],
}

#[inline(always)]
fn comp_dists(query: &MultiKey, piv: &MultiKey) -> [i16; DCOMP] {
    let mut out = [0i16; DCOMP];
    let mut t = 0usize;
    for p in 0..PHASES {
        for l in 0..LANES {
            out[t] = ham(query.h[p][l], piv.h[p][l]) as i16;
            t += 1;
        }
    }
    out
}

#[inline(always)]
fn dot_i16(w: &[i16; DCOMP], x: &[i16; DCOMP]) -> i32 {
    let mut s = 0i32;
    for i in 0..DCOMP {
        s += (w[i] as i32) * (x[i] as i32);
    }
    s
}

#[inline(always)]
fn dist_key(a: &MultiKey, b: &MultiKey) -> u32 {
    let mut d = 0u32;
    for p in 0..PHASES {
        for l in 0..LANES {
            d += ham(a.h[p][l], b.h[p][l]);
        }
    }
    d
}

// Rotate the phase axis of a query key (multi-probe).
#[inline(always)]
fn rotate_phases(q: &MultiKey, shift: usize) -> MultiKey {
    if shift % PHASES == 0 { return *q; }
    let mut out = MultiKey { h: [[0u64; LANES]; PHASES] };
    for p in 0..PHASES {
        out.h[p] = q.h[(p + shift) % PHASES];
    }
    out
}

// Deterministic small-noise perturbation (proxy for “non-exact query”)
#[inline(always)]
fn noisy_key(q: &MultiKey, seed: u64, strength: u32) -> MultiKey {
    let mut out = *q;
    for p in 0..PHASES {
        for l in 0..LANES {
            // perturb only a couple of words (keeps it "nearby")
            for w in 0..NOISE_WORDS {
                let m = splitmix64(seed ^ ((p as u64) << 32) ^ ((l as u64) << 16) ^ (w as u64))
                    & splitmix64(seed ^ 0xBAD5EED ^ (w as u64));
                // sparsify flips: keep only low-density bits
                let flip = m & (m >> strength);
                out.h[p][l] ^= flip;
            }
        }
    }
    out
}

#[inline(always)]
fn masked_key(q: &MultiKey, seed: u64, mode: u8) -> MultiKey {
    // mode:
    // 0 => keep ~75% bits (m1 | m2)
    // 1 => keep ~50% bits (m1)
    // 2 => keep ~25% bits (m1 & m2)
    let mut out = *q;
    for p in 0..PHASES {
        for l in 0..LANES {
            let m1 = splitmix64(seed ^ ((p as u64) << 32) ^ ((l as u64) << 16) ^ 0xA1);
            let m2 = splitmix64(seed ^ ((p as u64) << 32) ^ ((l as u64) << 16) ^ 0xB2);
            let keep = match mode {
                0 => m1 | m2,
                1 => m1,
                _ => m1 & m2,
            };
            out.h[p][l] &= keep;
        }
    }
    out
}

#[inline(always)]
fn dropout_key(q: &MultiKey, seed: u64) -> MultiKey {
    // Drop some 1-bits (simulate missing tokens/evidence).
    // We create a ~75% keep-mask by OR-ing two random 50% masks.
    let mut out = *q;
    for p in 0..PHASES {
        for l in 0..LANES {
            let m1 = splitmix64(seed ^ ((p as u64) << 32) ^ ((l as u64) << 16) ^ 0xA1);
            let m2 = splitmix64(seed ^ ((p as u64) << 32) ^ ((l as u64) << 16) ^ 0xB2);
            let keep = m1 | m2; // ~75% ones
            out.h[p][l] &= keep;
        }
    }
    out
}

#[inline(always)]
fn pos_salt(slot: u32, phase: u8, lane: usize) -> u64 {
    splitmix64((slot as u64)
        ^ ((phase as u64) << 32)
        ^ ((lane as u64) << 48)
        ^ 0x9E37_79B9_7F4A_7C15u64)
}

// -------------------- metric ternary tree --------------------

#[derive(Clone)]
struct Node {
    l: usize,
    r: usize,
    child: [Option<usize>; 3],
    piv: [usize; 3], // pivot key IDs
}

#[inline(always)]
fn is_leaf(n: &Node) -> bool {
    n.child[0].is_none() && n.child[1].is_none() && n.child[2].is_none()
}

#[inline(always)]
fn min3(a: u32, b: u32, c: u32) -> u32 {
    a.min(b).min(c)
}

fn choose_pivots(ids: &[usize], keys: &[MultiKey], l: usize, r: usize, rng: &mut u64) -> [usize; 3] {
    let n = r - l;
    let sample = n.min(256);
    let start = l + ((*rng as usize) % n);
    *rng = splitmix64(*rng);
    let a = ids[start];

    let mut samp: Vec<usize> = Vec::with_capacity(sample);
    for i in 0..sample {
        let idx = l + (((*rng as usize).wrapping_add(i * 997)) % n);
        samp.push(ids[idx]);
    }
    *rng = splitmix64(*rng);

    let mut b = a;
    let mut best = 0u32;
    for &x in &samp {
        let d = dist_key(&keys[a], &keys[x]);
        if d > best { best = d; b = x; }
    }

    let mut c = a;
    let mut best2 = 0u32;
    for &x in &samp {
        let da = dist_key(&keys[a], &keys[x]);
        let db = dist_key(&keys[b], &keys[x]);
        let d = da.min(db);
        if d > best2 { best2 = d; c = x; }
    }

    if a == b || b == c || a == c {
        let b2 = ids[l + n / 3];
        let c2 = ids[l + 2 * n / 3];
        return [a, b2, c2];
    }
    [a, b, c]
}

fn partition3(ids: &mut [usize], scratch: &mut [usize], keys: &[MultiKey], l: usize, r: usize, piv: [usize; 3])
    -> (usize, usize)
{
    let mut c0 = 0usize;
    let mut c1 = 0usize;
    let mut c2 = 0usize;

    for i in l..r {
        let id = ids[i];
        let d0 = dist_key(&keys[id], &keys[piv[0]]);
        let d1 = dist_key(&keys[id], &keys[piv[1]]);
        let d2 = dist_key(&keys[id], &keys[piv[2]]);
        let m = min3(d0, d1, d2);
        if m == d0 { c0 += 1; }
        else if m == d1 { c1 += 1; }
        else { c2 += 1; }
    }

    if c0 == 0 || c1 == 0 || c2 == 0 {
        let n = r - l;
        let s1 = l + n / 3;
        let s2 = l + 2 * n / 3;
        return (s1, s2);
    }

    let mut o0 = l;
    let mut o1 = l + c0;
    let mut o2 = l + c0 + c1;

    for i in l..r {
        let id = ids[i];
        let d0 = dist_key(&keys[id], &keys[piv[0]]);
        let d1 = dist_key(&keys[id], &keys[piv[1]]);
        let d2 = dist_key(&keys[id], &keys[piv[2]]);
        let m = min3(d0, d1, d2);

        if m == d0 { scratch[o0] = id; o0 += 1; }
        else if m == d1 { scratch[o1] = id; o1 += 1; }
        else { scratch[o2] = id; o2 += 1; }
    }

    ids[l..r].copy_from_slice(&scratch[l..r]);
    let m1 = l + c0;
    let m2 = l + c0 + c1;
    (m1, m2)
}

fn build_tree(
    nodes: &mut Vec<Node>,
    ids: &mut [usize],
    scratch: &mut [usize],
    keys: &[MultiKey],
    l: usize,
    r: usize,
    leaf_size: usize,
    rng: &mut u64,
) -> usize {
    let n = r - l;

    if n <= leaf_size {
        let rep_id = ids[l + n / 2];
        let node = Node { l, r, child: [None, None, None], piv: [rep_id, rep_id, rep_id] };
        let id = nodes.len();
        nodes.push(node);
        return id;
    }

    let piv = choose_pivots(ids, keys, l, r, rng);
    let (m1, m2) = partition3(ids, scratch, keys, l, r, piv);

    if m1 <= l || m2 <= m1 || r <= m2 {
        let rep_id = ids[l + n / 2];
        let node = Node { l, r, child: [None, None, None], piv: [rep_id, rep_id, rep_id] };
        let id = nodes.len();
        nodes.push(node);
        return id;
    }

    let c0 = build_tree(nodes, ids, scratch, keys, l, m1, leaf_size, rng);
    let c1 = build_tree(nodes, ids, scratch, keys, m1, m2, leaf_size, rng);
    let c2 = build_tree(nodes, ids, scratch, keys, m2, r, leaf_size, rng);

    let node = Node { l, r, child: [Some(c0), Some(c1), Some(c2)], piv };
    let id = nodes.len();
    nodes.push(node);
    id
}

// -------------------- Router (hashed-feature 3-class perceptron) --------------------

#[derive(Clone)]
struct RouterModel {
    // weights: [node][class][feat]
    w: Vec<i8>,
    // bias: [node][class]
    b: Vec<i16>,
    nodes: usize,
}

impl RouterModel {
    fn new(nodes: usize) -> Self {
        let w_len = nodes * 3 * FEAT_DIM;
        let b_len = nodes * 3;
        Self { w: vec![0i8; w_len], b: vec![0i16; b_len], nodes }
    }

    #[inline(always)]
    fn w_index(node: usize, class: usize, feat: usize) -> usize {
        (node * 3 + class) * FEAT_DIM + feat
    }
    #[inline(always)]
    fn b_index(node: usize, class: usize) -> usize {
        node * 3 + class
    }

    #[inline(always)]
    fn extract_features(q: &MultiKey, seed: u64, out_idx: &mut [u8; FEATS_PER_QUERY], out_sgn: &mut [i8; FEATS_PER_QUERY]) {
        // flatten 12 u64 words
        let mut words = [0u64; PHASES * LANES];
        let mut t = 0usize;
        for p in 0..PHASES {
            for l in 0..LANES {
                words[t] = q.h[p][l];
                t += 1;
            }
        }

        for i in 0..FEATS_PER_QUERY {
            let w = words[i % words.len()];
            let h = splitmix64(seed ^ w ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let idx = (h as usize) & (FEAT_DIM - 1);
            out_idx[i] = idx as u8;
            out_sgn[i] = if (h >> 63) == 0 { 1 } else { -1 };
        }
    }

    #[inline(always)]
    fn scores(&self, node: usize, feat_idx: &[u8; FEATS_PER_QUERY], feat_sgn: &[i8; FEATS_PER_QUERY]) -> [i32; 3] {
        let mut s = [0i32; 3];
        for c in 0..3 {
            let mut acc = self.b[Self::b_index(node, c)] as i32;
            for i in 0..FEATS_PER_QUERY {
                let f = feat_idx[i] as usize;
                let wv = self.w[Self::w_index(node, c, f)] as i32;
                acc += (feat_sgn[i] as i32) * wv;
            }
            s[c] = acc;
        }
        s
    }

    #[inline(always)]
    fn update(&mut self, node: usize, truth: usize, pred: usize, feat_idx: &[u8; FEATS_PER_QUERY], feat_sgn: &[i8; FEATS_PER_QUERY]) {
        if truth == pred { return; }
        // bias
        let bt = Self::b_index(node, truth);
        let bp = Self::b_index(node, pred);
        self.b[bt] = self.b[bt].saturating_add(1);
        self.b[bp] = self.b[bp].saturating_sub(1);

        // weights
        for i in 0..FEATS_PER_QUERY {
            let f = feat_idx[i] as usize;
            let sgn = feat_sgn[i] as i16;

            let it = Self::w_index(node, truth, f);
            let ip = Self::w_index(node, pred, f);

            // saturating i8 update
            let wt = self.w[it] as i16 + sgn;
            let wp = self.w[ip] as i16 - sgn;

            self.w[it] = wt.clamp(-127, 127) as i8;
            self.w[ip] = wp.clamp(-127, 127) as i8;
        }
    }
}

// -------------------- per depth learned metric router ------------------------------
#[derive(Clone)]
struct MetricRouter {
    // one weight vector per depth (small and stable)
    w: Vec<[i16; DCOMP]>,
}

impl MetricRouter {
    fn new(depths: usize) -> Self {
        let mut w = Vec::with_capacity(depths);
        for _ in 0..depths {
            w.push([0i16; DCOMP]); // start at zero
        }
        Self { w }
    }

    #[inline(always)]
    fn cost(&self, depth: usize, d: &[i16; DCOMP]) -> i32 {
        let di = depth.min(self.w.len().saturating_sub(1));
        dot_i16(&self.w[di], d)
    }

    #[inline(always)]
    fn update(&mut self, depth: usize, d_true: &[i16; DCOMP], d_rival: &[i16; DCOMP]) {
        let di = depth.min(self.w.len().saturating_sub(1));
        let w = &mut self.w[di];
        for i in 0..DCOMP {
            // w += (d_rival - d_true)
            let delta: i32 = (d_rival[i] as i32) - (d_true[i] as i32);
            let nw: i32 = (w[i] as i32 + delta).clamp(W_MIN as i32, W_MAX as i32);
            w[i] = nw as i16;
        }
    }
}

// -------------------- diverse beam routing (baseline + learned) --------------------

#[derive(Clone, Copy)]
struct BeamItemU32 {
    node: usize,
    cost: i32, // lower is better
}

#[inline(always)]
fn overlap_len(a_l: usize, a_r: usize, b_l: usize, b_r: usize) -> usize {
    let l = a_l.max(b_l);
    let r = a_r.min(b_r);
    if r > l { r - l } else { 0 }
}

fn select_diverse(nodes: &[Node], mut cand: Vec<BeamItemU32>, beam: usize) -> Vec<BeamItemU32> {
    cand.sort_unstable_by_key(|x| x.cost);
    if cand.len() <= beam { return cand; }

    let mut out: Vec<BeamItemU32> = Vec::with_capacity(beam);
    let mut out_ranges: Vec<(usize, usize)> = Vec::with_capacity(beam);

    for it in cand.iter() {
        if out.len() >= beam { break; }
        let n = &nodes[it.node];
        let (l, r) = (n.l, n.r);
        let len = r - l;

        let mut ok = true;
        for &(ol, or) in &out_ranges {
            let olen = or - ol;
            let ov = overlap_len(l, r, ol, or);
            if ov * 2 >= len.min(olen) { ok = false; break; }
        }
        if ok {
            out.push(*it);
            out_ranges.push((l, r));
        }
    }

    if out.len() < beam {
        for it in cand.iter() {
            if out.len() >= beam { break; }
            if !out.iter().any(|x| x.node == it.node) {
                out.push(*it);
            }
        }
    }

    out
}

#[inline(always)]
fn leaf_contains(tree_pos_of: &[usize], nodes: &[Node], node_id: usize, cid: usize) -> bool {
    let pos = tree_pos_of[cid];
    let n = &nodes[node_id];
    pos >= n.l && pos < n.r
}

// Baseline routing: cost += distance to pivot key
fn route_beam_baseline(
    nodes: &[Node],
    ids: &[usize],
    keys: &[MultiKey],
    root: usize,
    query: &MultiKey,
    beam: usize,
    max_depth: usize,
) -> Vec<usize> {
    let mut cur: Vec<BeamItemU32> = vec![BeamItemU32 { node: root, cost: 0 }];

    for _ in 0..max_depth {
        if cur.iter().all(|it| is_leaf(&nodes[it.node])) { break; }

        let mut next: Vec<BeamItemU32> = Vec::with_capacity(cur.len() * 3);
        for it in &cur {
            let n = &nodes[it.node];
            if is_leaf(n) { next.push(*it); continue; }
            for ci in 0..3 {
                if let Some(ch) = n.child[ci] {
                    let piv_id = n.piv[ci];
                    let d = dist_key(query, &keys[piv_id]) as i32;
                    next.push(BeamItemU32 { node: ch, cost: it.cost + d });
                }
            }
        }
        cur = select_diverse(nodes, next, beam);
    }

    cur.into_iter().map(|x| x.node).collect()
}

fn route_beam_metric_learned(
    nodes: &[Node],
    keys: &[MultiKey],
    root: usize,
    query: &MultiKey,
    metric: &MetricRouter,
    beam: usize,
    max_depth: usize,
    learn_levels: usize,
) -> Vec<usize> {
    const SHIFT: i32 = 4; // 4=>/16, 5=>/32 (weaker)
    let step_penalty: i32 = 2;

    let mut cur: Vec<BeamItemU32> = vec![BeamItemU32 { node: root, cost: 0 }];

    for depth in 0..max_depth {
        if cur.iter().all(|it| is_leaf(&nodes[it.node])) {
            break;
        }

        let use_learned = depth < learn_levels;

        let mut next: Vec<BeamItemU32> = Vec::with_capacity(cur.len() * 3);
        for it in &cur {
            let n = &nodes[it.node];
            if is_leaf(n) {
                next.push(*it);
                continue;
            }

            if use_learned {
                // compute learned costs once
                let mut lcost = [0i32; 3];
                for ci in 0..3 {
                    let piv_id = n.piv[ci];
                    let dvec = comp_dists(query, &keys[piv_id]);
                    lcost[ci] = metric.cost(depth, &dvec);
                }
                let min_l = lcost[0].min(lcost[1]).min(lcost[2]);

                // push children once
                for ci in 0..3 {
                    if let Some(ch) = n.child[ci] {
                        let piv_id = n.piv[ci];
                        let base = dist_key(query, &keys[piv_id]) as i32;

                        // one-sided penalty (never decreases cost)
                        let penalty = ((lcost[ci] - min_l).max(0)) >> SHIFT;

                        let d = base + penalty;
                        next.push(BeamItemU32 {
                            node: ch,
                            cost: it.cost + d + step_penalty,
                        });
                    }
                }
            } else {
                // baseline push (once)
                for ci in 0..3 {
                    if let Some(ch) = n.child[ci] {
                        let piv_id = n.piv[ci];
                        let base = dist_key(query, &keys[piv_id]) as i32;
                        next.push(BeamItemU32 {
                            node: ch,
                            cost: it.cost + base + step_penalty,
                        });
                    }
                }
            }
        }

        cur = select_diverse(nodes, next, beam);
    }

    cur.into_iter().map(|x| x.node).collect()
}

// Learned routing: cost -= router_score(child) (higher score -> lower cost)
fn route_beam_learned(
    nodes: &[Node],
    ids: &[usize],
    keys: &[MultiKey],
    root: usize,
    model: &RouterModel,
    query: &MultiKey,
    feat_idx: &[u8; FEATS_PER_QUERY],
    feat_sgn: &[i8; FEATS_PER_QUERY],
    beam: usize,
    max_depth: usize,
    learn_levels: usize,   // NEW
) -> Vec<usize> {
    let step_penalty: i32 = 2;
    let router_shift: i32 = 4; // gap >> 4  (try 4 or 5)

    let mut cur: Vec<BeamItemU32> = vec![BeamItemU32 { node: root, cost: 0 }];

    for depth in 0..max_depth {
        if cur.iter().all(|it| is_leaf(&nodes[it.node])) { break; }

        let use_router = depth < learn_levels;

        let mut next: Vec<BeamItemU32> = Vec::with_capacity(cur.len() * 3);
        for it in &cur {
            let n = &nodes[it.node];
            if is_leaf(n) { next.push(*it); continue; }

            // Only compute router scores in the top levels
            let (smax, s) = if use_router {
                let s = model.scores(it.node, feat_idx, feat_sgn); // [3]
                (s[0].max(s[1]).max(s[2]), s)
            } else {
                (0, [0i32; 3])
            };

            for ci in 0..3 {
                if let Some(ch) = n.child[ci] {
                    let piv_id = n.piv[ci];
                    let d = dist_key(query, &keys[piv_id]) as i32;

                    // Baseline backbone
                    let mut new_cost = it.cost + d + step_penalty;

                    // Router only as a small penalty (top levels only)
                    if use_router {
                        let gap = (smax - s[ci]).max(0);
                        let router_penalty = gap >> router_shift; // divide by 16 if shift=4
                        new_cost += router_penalty;
                    }

                    next.push(BeamItemU32 { node: ch, cost: new_cost });
                }
            }
        }
        cur = select_diverse(nodes, next, beam);
    }

    cur.into_iter().map(|x| x.node).collect()
}

fn build_boundary_keys_and_examples(
    path: &str,
    max_chunks: usize,
    chunk_tokens: usize,
    context_tokens: usize,
    max_examples: usize,
    token_stride: usize,
    seed: u64,
    k: u64,
) -> (Vec<MultiKey>, Vec<u8>, Vec<usize>) {
    let file = File::open(path).expect("failed to open input file");
    let reader = BufReader::new(file);

    let mut osc = GcdThreeCycleOsc::new(seed ^ 0xC0FFEE, k);

    // Context counters: [phase][lane][bit]
    let mut ctx_c: [[[i32; 64]; LANES]; PHASES] = [[[0; 64]; LANES]; PHASES];
    let mut ctx_queue: VecDeque<(u64, u32)> = VecDeque::with_capacity(context_tokens.max(1));
    let mut slot_counter: u64 = 0;

    let mut keys: Vec<MultiKey> = Vec::with_capacity(max_chunks);
    let mut phases: Vec<u8> = Vec::with_capacity(max_chunks);
    let mut example_ids: Vec<usize> = Vec::with_capacity(max_examples);

    let mut token_buf: Vec<u8> = Vec::with_capacity(64);

    let mut global_tok: u64 = 0;
    let mut cur_chunk_tok: usize = 0;
    let mut warmed = false;

    let sample_every = (max_chunks / max_examples.max(1)).max(1);

    for line_res in reader.lines() {
        let line = line_res.expect("read line failed");
        let bytes = line.as_bytes();
        let mut i = 0usize;

        while i < bytes.len() {
            while i < bytes.len() && bytes[i].is_ascii_whitespace() { i += 1; }
            if i >= bytes.len() { break; }

            token_buf.clear();

            if is_word_byte(bytes[i]) {
                while i < bytes.len() && is_word_byte(bytes[i]) {
                    token_buf.push(bytes[i]);
                    i += 1;
                }
            } else {
                token_buf.push(bytes[i]);
                i += 1;
            }

            let base_h = token_hash_from_bytes(seed, &token_buf);

            let do_update = (global_tok as usize) % token_stride == 0;
            global_tok += 1;

            if do_update {
                let slot = (slot_counter % (context_tokens.max(1) as u64)) as u32;
                slot_counter += 1;

                // ADD case (with position salt)
                for p in 0..PHASES {
                    let ph = phase_variant(base_h, p as u8);
                    for l in 0..LANES {
                        let hv = lane_variant(ph ^ pos_salt(slot, p as u8, l), l);
                        add_simhash(&mut ctx_c[p][l], hv, 1);
                    }
                }
                ctx_queue.push_back((base_h, slot));

                // REMOVE case (with the same slot)
                while ctx_queue.len() > context_tokens.max(1) {
                    if let Some((old, old_slot)) = ctx_queue.pop_front() {
                        for p in 0..PHASES {
                            let ph = phase_variant(old, p as u8);
                            for l in 0..LANES {
                                let hv = lane_variant(ph ^ pos_salt(old_slot, p as u8, l), l);
                                add_simhash(&mut ctx_c[p][l], hv, -1);
                            }
                        }
                    }
                }

                if ctx_queue.len() >= context_tokens.max(1) {
                    warmed = true;
                }
            }

            cur_chunk_tok += 1;

            // snapshot key at START of chunk boundary
            if cur_chunk_tok == 1 && warmed && keys.len() < max_chunks {
                let mut h = [[0u64; LANES]; PHASES];
                for p in 0..PHASES {
                    for l in 0..LANES {
                        h[p][l] = counters_to_u64(&ctx_c[p][l]);
                    }
                }
                let phase0 = osc.step_and_phase();
                let cid = keys.len();
                keys.push(MultiKey { h });
                phases.push(phase0);

                if cid % sample_every == 0 && example_ids.len() < max_examples {
                    example_ids.push(cid);
                }
            }

            if cur_chunk_tok >= chunk_tokens {
                cur_chunk_tok = 0;
                if keys.len() >= max_chunks {
                    return (keys, phases, example_ids);
                }
            }
        }
    }

    (keys, phases, example_ids)
}

// -------------------- forest tree container --------------------

struct Tree {
    nodes: Vec<Node>,
    ids: Vec<usize>,      // permutation of key IDs (positions)
    pos_of: Vec<usize>,   // inverse map: key ID -> position in ids
    root: usize,
    metric: MetricRouter,
}

fn build_one_tree(keys: &[MultiKey], leaf_size: usize, seed: u64, train_depth: usize) -> Tree {
    let n = keys.len();
    let mut ids: Vec<usize> = (0..n).collect();
    let mut scratch: Vec<usize> = vec![0usize; n];
    let mut nodes: Vec<Node> = Vec::with_capacity(n * 2);
    let mut rng = seed ^ 0xBAD5EEDu64;

    let root = build_tree(&mut nodes, &mut ids, &mut scratch, keys, 0, n, leaf_size, &mut rng);

    let mut pos_of = vec![0usize; n];
    for (pos, &cid) in ids.iter().enumerate() {
        pos_of[cid] = pos;
    }

    let metric = MetricRouter::new(train_depth);
    Tree { nodes, ids, pos_of, root, metric }
}

fn build_forest(keys: &[MultiKey], leaf_size: usize, trees: usize, seed: u64, train_depth: usize) -> Vec<Tree> {
    let mut forest: Vec<Tree> = Vec::with_capacity(trees);
    for t in 0..trees {
        let s = seed ^ ((t as u64).wrapping_mul(0x9E3779B97F4A7C15)) ^ 0xC0FFEEu64;
        forest.push(build_one_tree(keys, leaf_size, s, train_depth));
    }
    forest
}

// -------------------- router training --------------------

#[inline(always)]
fn child_label_for_pos(nodes: &[Node], node_id: usize, pos: usize) -> Option<usize> {
    let n = &nodes[node_id];
    if is_leaf(n) { return None; }
    for ci in 0..3 {
        if let Some(ch) = n.child[ci] {
            let cn = &nodes[ch];
            if pos >= cn.l && pos < cn.r {
                return Some(ci);
            }
        }
    }
    None
}

fn train_tree_router_metric(
    tree: &mut Tree,
    keys: &[MultiKey],
    train_ids: &[usize],
    train_depth: usize,
    epochs: usize,
) {
    let mut total: u64 = 0;
    let mut correct: u64 = 0;
    let mut updates: u64 = 0;

    // Any constant seed is fine; per-epoch/per-id mixing makes it deterministic but varied
    let base_seed: u64 = 0x1F2A_BADC_0DEu64;

    for ep in 0..epochs {
        for &cid in train_ids {
            let pos = tree.pos_of[cid];
            let q0 = keys[cid];

            // --- v7: train on the same corruption family we evaluate on ---
            // Clean
            let q_clean = q0;

            // Noise (matches your noisy proxy distribution)
            let q_noisy = noisy_key(
                &q0,
                base_seed ^ (cid as u64) ^ ((ep as u64) << 32) ^ 0xC0FFEE,
                NOISE_MASK_BITS,
            );

            // Structured masking / "less evidence" (simulates shorter / partial queries)
            // mode 1 => ~50% keep; mode 2 => ~25% keep if you want it harsher
            let q_mask = masked_key(
                &q0,
                base_seed ^ (cid as u64) ^ ((ep as u64) << 32) ^ 0xD00D,
                1,
            );

            let variants = [q_clean, q_noisy, q_mask];

            for q in &variants {
                let mut node = tree.root;

                for depth in 0..train_depth {
                    let lbl = match child_label_for_pos(&tree.nodes, node, pos) {
                        Some(x) => x,
                        None => break,
                    };

                    let n = &tree.nodes[node];
                    if is_leaf(n) { break; }

                    // compute per-child component distance vectors (to that child's pivot)
                    let mut dvec = [[0i16; DCOMP]; 3];
                    let mut cost = [0i32; 3];

                    for ci in 0..3 {
                        let piv_id = n.piv[ci];
                        dvec[ci] = comp_dists(q, &keys[piv_id]);
                        cost[ci] = tree.metric.cost(depth, &dvec[ci]);
                    }

                    // predict = argmin cost
                    let mut best = 0usize;
                    if cost[1] < cost[best] { best = 1; }
                    if cost[2] < cost[best] { best = 2; }

                    // second best (for margin/rival)
                    let mut second = if best == 0 { 1 } else { 0 };
                    for c in 0..3 {
                        if c != best && cost[c] < cost[second] {
                            second = c;
                        }
                    }

                    total += 1;
                    if best == lbl { correct += 1; }

                    // hinge: want cost(lbl) + margin <= cost(rival)
                    let rival = if best == lbl { second } else { best };
                    if cost[lbl] + METRIC_MARGIN > cost[rival] {
                        tree.metric.update(depth, &dvec[lbl], &dvec[rival]);
                        updates += 1;
                    }

                    // teacher-forced descend
                    if let Some(ch) = n.child[lbl] {
                        node = ch;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    let acc = (correct as f64) / (total as f64).max(1.0);
    let upd = (updates as f64) / (total as f64).max(1.0);
    println!(
        "  metric router train: steps={} acc={:.2}% updates/step={:.2}%",
        total, 100.0 * acc, 100.0 * upd
    );
}

// -------------------- evaluation (baseline vs learned) --------------------

fn recall_forest(
    forest: &[Tree],
    keys: &[MultiKey],
    example_ids: &[usize],
    beam: usize,
    probes: usize,
    learned: bool,
    noisy: bool,
) -> f64 {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(12).max(1);
    let chunk = (example_ids.len() + threads - 1) / threads;

    let found_total: u64 = std::thread::scope(|scope| {
        let mut handles = Vec::new();

        for tid in 0..threads {
            let start = tid * chunk;
            if start >= example_ids.len() { break; }
            let end = ((tid + 1) * chunk).min(example_ids.len());
            let slice = &example_ids[start..end];

            let forest_ref = forest;
            let keys_ref = keys;

            handles.push(scope.spawn(move || -> u64 {
                let mut found = 0u64;
                let mut feat_idx = [0u8; FEATS_PER_QUERY];
                let mut feat_sgn = [0i8; FEATS_PER_QUERY];

                for &cid in slice {
                    let base = &keys_ref[cid];
                    let q_base = if noisy {
                        noisy_key(base, 0xBADC0DE ^ (cid as u64), NOISE_MASK_BITS)
                    } else {
                        *base
                    };

                    let mut ok = false;

                    for shift in 0..probes.min(PHASES).max(1) {
                        let q = rotate_phases(&q_base, shift);
                        if learned {
                            // features depend on probe rotation
                            RouterModel::extract_features(&q, 0x1F2A ^ (shift as u64), &mut feat_idx, &mut feat_sgn);
                        }

                        for tr in forest_ref {
                            let learn_levels = 4; // start here

                            let beam_nodes = if learned {
                                route_beam_metric_learned(
                                    &tr.nodes, keys_ref, tr.root, &q, &tr.metric,
                                    beam, 64, learn_levels
                                )
                            } else {
                                route_beam_baseline(&tr.nodes, &tr.ids, keys_ref, tr.root, &q, beam, 64)
                            };

                            for nid in beam_nodes {
                                if leaf_contains(&tr.pos_of, &tr.nodes, nid, cid) {
                                    ok = true;
                                    break;
                                }
                            }
                            if ok { break; }
                        }
                        if ok { break; }
                    }

                    if ok { found += 1; }
                }
                found
            }));
        }

        let mut sum = 0u64;
        for h in handles {
            sum += h.join().unwrap();
        }
        sum
    });

    (found_total as f64) / (example_ids.len() as f64).max(1.0)
}

// -------------------- main --------------------

fn main() {
    // ./main <path> <max_chunks> <chunk_tokens> <context_tokens> <examples>
    //        <beam> <token_stride> <leaf_size> <trees> <probes>
    //        <train_examples> <train_depth> <epochs>
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <path> [max_chunks] [chunk_tokens] [context_tokens] [examples] [beam] [token_stride] [leaf_size] [trees] [probes] [train_examples] [train_depth] [epochs]",
            args[0]
        );
        std::process::exit(1);
    }

    let path = &args[1];
    let max_chunks: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(300_000);
    let chunk_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    let context_tokens: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(256);
    let examples_n: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let beam: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(32).max(1);
    let token_stride: usize = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(1).max(1);
    let leaf_size: usize = args.get(8).and_then(|s| s.parse().ok()).unwrap_or(64).max(1);
    let trees: usize = args.get(9).and_then(|s| s.parse().ok()).unwrap_or(8).max(1);
    let probes: usize = args.get(10).and_then(|s| s.parse().ok()).unwrap_or(3).max(1);

    let train_examples: usize = args.get(11).and_then(|s| s.parse().ok()).unwrap_or(examples_n).max(1);
    let train_depth: usize = args.get(12).and_then(|s| s.parse().ok()).unwrap_or(12).max(1);
    let epochs: usize = args.get(13).and_then(|s| s.parse().ok()).unwrap_or(1).max(1);

    let seed = 0x1F2Au64;
    let k = 1u64;

    println!("--- Pyramid Router v5 (Forest + Learned Routers) ---");
    println!("path = {}", path);
    println!("max_chunks = {}", max_chunks);
    println!("chunk_tokens = {}", chunk_tokens);
    println!("context_tokens = {}", context_tokens);
    println!("examples = {}", examples_n);
    println!("beam = {}", beam);
    println!("token_stride = {}", token_stride);
    println!("leaf_size = {}", leaf_size);
    println!("trees = {}", trees);
    println!("probes = {}", probes);
    println!("train_examples = {}", train_examples);
    println!("train_depth = {}", train_depth);
    println!("epochs = {}", epochs);
    println!("LANES = {} (=> {} bits/phase, {} total bits)", LANES, 64 * LANES, 64 * LANES * PHASES);
    println!("Router: FEAT_DIM={} FEATS_PER_QUERY={} MARGIN={}", FEAT_DIM, FEATS_PER_QUERY, ROUTER_MARGIN);

    // 1) Build boundary keys
    let t0 = Instant::now();
    let (keys, _phases, example_ids) = build_boundary_keys_and_examples(
        path,
        max_chunks,
        chunk_tokens,
        context_tokens,
        examples_n,
        token_stride,
        seed,
        k,
    );
    let dt_build = t0.elapsed();
    println!("\nBuilt boundary keys:");
    println!("  keys indexed = {}", keys.len());
    println!("  examples collected = {}", example_ids.len());
    println!("  time = {:?}\n", dt_build);

    // Prepare training ids (subset of example_ids)
    let train_n = train_examples.min(example_ids.len());
    let train_ids = &example_ids[..train_n];

    // 2) Build forest
    let t1 = Instant::now();
    let mut forest = build_forest(&keys, leaf_size, trees, seed, train_depth);
    let dt_forest = t1.elapsed();
    let node_sum: usize = forest.iter().map(|t| t.nodes.len()).sum();
    println!("Forest built:");
    println!("  trees = {}", forest.len());
    println!("  total nodes = {}", node_sum);
    println!("  time = {:?}\n", dt_forest);

    // 3) Baseline recall (similarity routing)
    let t2 = Instant::now();
    let r_base = recall_forest(&forest, &keys, &example_ids, beam, probes, false, false);
    let dt_base = t2.elapsed();
    println!("Baseline routing (similarity) recall@beam = {:.2}%  |  time = {:?}", 100.0 * r_base, dt_base);

    // 4) Train routers (parallel across trees)
    println!("\nTraining routers (teacher-forced path supervision)...");
    let t3 = Instant::now();
    std::thread::scope(|scope| {
        let keys_ref = &keys;
        for (ti, tr) in forest.iter_mut().enumerate() {
            let seed_t = seed ^ (ti as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ 0x51515151;
            scope.spawn(move || {
                println!(" Tree {}:", ti);
                train_tree_router_metric(tr, keys_ref, train_ids, train_depth, epochs);
            });
        }
    });
    let dt_train = t3.elapsed();
    println!("Router training time: {:?}\n", dt_train);

    // 5) Learned recall (exact keys)
    let t4 = Instant::now();
    let r_learn = recall_forest(&forest, &keys, &example_ids, beam, probes, true, false);
    let dt_learn = t4.elapsed();
    println!("Learned routing recall@beam = {:.2}%  |  time = {:?}", 100.0 * r_learn, dt_learn);

    // 6) Noisy-query recall (generalization proxy)
    let t5 = Instant::now();
    let r_noise_base = recall_forest(&forest, &keys, &example_ids, beam, probes, false, true);
    let dt_noise_base = t5.elapsed();
    let t6 = Instant::now();
    let r_noise_learn = recall_forest(&forest, &keys, &example_ids, beam, probes, true, true);
    let dt_noise_learn = t6.elapsed();

    println!("\nNoisy-query (proxy) recall:");
    println!("  baseline = {:.2}% | time = {:?}", 100.0 * r_noise_base, dt_noise_base);
    println!("  learned  = {:.2}% | time = {:?}", 100.0 * r_noise_learn, dt_noise_learn);

    println!("\nIf learned routing beats baseline on noisy queries, that’s the first real sign we’re generalizing beyond exact-key lookup.");
    println!("Next (v6): teach routers using *real* queries (sub-contexts, paraphrase-like masks), not just boundary self-keys.");
}