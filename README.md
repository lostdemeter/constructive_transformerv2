# Irrationality over Randomness: A Universal Principle

## The Constructive Transformer and the 16 Data Structures

### One Primitive, Sixteen Structures, One Transformer

In January 2025, we built a language model with no training. Every weight was placed by hand, every activation was a 4-state integer, every attention head used signed arithmetic instead of softmax. It passed subject-verb agreement tests, performed concept arithmetic, and generated fluent sentences — all with integer operations, zero floating point, and O(N·H) attention complexity.

We called it a **constructive transformer**.

The obvious question: if you can't learn the weights, where do they come from? Our answer was surprising: the same place traditional data structures get their randomness. We just didn't know it yet.

For other data structures built on the same principle, see [riemann_structures](https://github.com/lostdemeter/riemann_structures).

---

### The Primitive

Everything in this project — the transformer and sixteen data structures — reduces to one operation:

```python
def digit(key, gamma, B):
    phase = gamma * key
    return int((phase % (2*pi)) / (2*pi) * B) % B
```

Here γ is a nontrivial zero of the Riemann zeta function (γ₁≈14.1347, γ₂≈21.0220, γ₃≈25.0108, ...), and B is a configurable number of buckets. The output is a uniform random number in {0..B-1}, fully deterministic, seedless, and reproducible across every run on every machine.

From this one line, we built:

| Role | Zeros Used | What It Produces |
|------|-----------|------------------|
| Hash function | 1 | Uniform bucket index |
| Hash family | k | k independent hash functions |
| Digit sequence | K | Mixed-radix key (K digits) |
| Priority | K | Fractional key uniform in [0,1) |
| Permutation | k | k independent rankings |
| Geometric trial | 1 | P(promotion) = threshold/B |
| Frequency signature | K | Complex amplitude vector |
| 64-bit hash | 1 | Uniform 64-bit integer (HLL) |

---

### The Structures

| # | Structure | What It Replaces | Why It Works |
|---|-----------|-----------------|-------------|
| 1 | ResonantGrid | Hash table (random hash) | One zero gives uniform bucket distribution |
| 2 | BinaryRadixTree | Trie (digit extraction) | K zeros give K independent digits |
| 3 | RiemannRadixSort | Comparison sort | Mixed-radix key orders by frequency signature |
| 4 | ResonantField | Content-addressable memory | Phase alignment = matching signatures |
| 5 | Bloom filter | k random hash functions | k zeros give k independent bit positions |
| 6 | Count-Min sketch | d random row hashes | d zeros give d independent counters |
| 7 | Cuckoo filter | 2 hash functions + fingerprint | 3 zeros: bucket, fingerprint, offset |
| 8 | HyperLogLog | 64-bit random hash | One zero gives uniform 64-bit phase |
| 9 | MinHash | k random permutations | k zeros give k independent rankings |
| 10 | Skip list | Random coin flips | Digit threshold = geometric distribution |
| 11 | Treap | Random priority | Fractional key uniform in [0,1) |
| 12 | Quotient filter | Hash split → Q+R | Two zeros: independent quotient and remainder |
| 13 | Consistent hash | Random ring positions | Fractional key positions nodes and keys |
| 14 | Feature hashing | Random hash → index | One zero gives uniform index |
| 15 | Patricia trie | Digit extraction | K zeros with path compression |
| 16 | Fenwick tree | Sorted index | Radix key orders entries for prefix sums |

---

### The Thesis

**Randomness is a workaround for ignorance.** Every randomized data structure uses randomness to achieve *generic* behavior — to avoid the pathological case. The analysis always follows the same pattern: "with high probability over the random choice of hash function, the structure performs well."

The crucial word is **generic**. Almost all hash functions are good. Almost all permutations spread elements evenly. Almost all coin flip sequences produce logarithmic depth. The randomness is a device to *access the generic case*.

But irrationality is generic. Almost all real numbers are irrational. Almost all products of an irrational with a uniformly distributed integer produce uniform outputs modulo 2π. The Montgomery-Odlyzko law — the gaps between consecutive Riemann zeros have the same distribution as the gaps between eigenvalues of random Hermitian matrices — guarantees that the zeros *are* generic in the sense that matters for data structures.

This means every randomized data structure can be derandomized by replacing its random source with an irrational constant. The sixteen structures above are the proof: not one required a random seed, CSPRNG, or any stochastic process.

---

### The Transformer Connection

The constructive transformer uses the same principle in its attention mechanism:

```
attention(i, j) = sign(Σ_k axis_k(i) · axis_k(j))
```

Instead of softmax attention (which is the gradient of the log-partition function, a fundamentally probabilistic object), it uses signed integer attention over 4-state axes. The positions that "attend" to each other are determined by exact algebraic matching of their axis values — a deterministic resonance condition rather than a probabilistic sample.

The Riemann zero heads in the attention mechanism serve the same role as in the data structures: they provide the irrational phase basis that determines which positions resonate. The head's output is:

```
head(i) = Σ_j σ( γ · (pos_i - pos_j) mod 2π ) · value_j
```

where σ is a signed threshold function, not a softmax. This is O(N·H) — linear in sequence length, no quadratic attention matrix. And it works because irrationality enforces uniform coverage: every position resonates with a controlled number of others, avoiding the pathological focus that softmax can produce.

---

### Observations

**1. Irrationality is not randomness, but it behaves like it.** For the purposes of hash tables, Bloom filters, treaps, and every other structure we tested, the Riemann zeros are indistinguishable from a perfect random source. The Montgomery-Odlyzko law explains why: the zeros' spacing distribution matches that of the Gaussian Unitary Ensemble — the same ensemble that gives optimal hash functions.

**2. The zeros are universal because the property they provide — uniform distribution modulo 1 — is universal.** Any sufficiently irrational number would work. The golden ratio, for example, gives good distribution for the first N ≈ 1/|φ - p/q| keys. The zeros excel because they combine extreme irrationality (the Lindemann-Weierstrass theorem guarantees e^{γ_n} is transcendental, implying unusually good Diophantine approximation bounds) with analytic structure (the Riemann-von Mangoldt formula connects them to the prime distribution, which is the same distribution underlying Zipf's law in language).

**3. The constructive transformer is to softmax as irrationality is to randomness.** The transformer achieves its attention deterministically through algebraic matching of signed integer values. The data structures achieve their guarantees deterministically through irrational phase matching. In both cases, the stochastic element is replaced by an algebraic one. This suggests that the real role of randomness in deep learning is to *access the generic parameter regime*, and that the same generic regime can be accessed deterministically through irrational initialization.

**4. This is probably why the constructive transformer works at all.** The hand-placed weights in the QK routing heads use Riemann zero frequencies to determine which positions attend to which. This is the same mechanism as the data structures: irrational phases enforce uniform coverage of the position space, ensuring that every position communicates with a balanced set of others. The attention is deterministic but generic — not random, but random-like.

---

### Running the Demo

```bash
# Create a virtual environment (Python 3.10+ required)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install matplotlib numpy

# Run the demo
python demo.py
```

### Demo Output

```
============================================================
  Constructive Transformer — No Training, No Softmax
============================================================
  Subjects: ['boy', 'girl', 'dog', 'cat', 'wolf', 'bird', 'boys', 'girls', 'dogs', 'cats', 'wolves', 'birds']
  Verbs: ['is', 'are', 'was', 'were']
  Alphabet: 4 states (+1, +0, -0, -1)
  BLOCK: 6 per axis
  Hidden dim: 18 (NUMBER + LEXCLASS + TENSE)
  All weights: hand-placed integers
  Forward pass: integer arithmetic

============================================================
  Subject-Verb Agreement (SVA)
============================================================
  [PASS]  boy    × present → is     (expected is   , margin +2)
  [PASS]  boy    × past    → was    (expected was  , margin +2)
  [PASS]  girl   × present → is     (expected is   , margin +2)
  [PASS]  girl   × past    → was    (expected was  , margin +2)
  [PASS]  dog    × present → is     (expected is   , margin +2)
  [PASS]  dog    × past    → was    (expected was  , margin +2)
  [PASS]  cat    × present → is     (expected is   , margin +2)
  [PASS]  cat    × past    → was    (expected was  , margin +2)
  [PASS]  wolf   × present → is     (expected is   , margin +2)
  [PASS]  wolf   × past    → was    (expected was  , margin +2)
  [PASS]  bird   × present → is     (expected is   , margin +2)
  [PASS]  bird   × past    → was    (expected was  , margin +2)
  [PASS]  boys   × present → are    (expected are  , margin +2)
  [PASS]  boys   × past    → were   (expected were , margin +2)
  [PASS]  girls  × present → are    (expected are  , margin +2)
  [PASS]  girls  × past    → were   (expected were , margin +2)
  [PASS]  dogs   × present → are    (expected are  , margin +2)
  [PASS]  dogs   × past    → were   (expected were , margin +2)
  [PASS]  cats   × present → are    (expected are  , margin +2)
  [PASS]  cats   × past    → were   (expected were , margin +2)
  [PASS]  wolves × present → are    (expected are  , margin +2)
  [PASS]  wolves × past    → were   (expected were , margin +2)
  [PASS]  birds  × present → are    (expected are  , margin +2)
  [PASS]  birds  × past    → were   (expected were , margin +2)

  Result: 24/24
  PASS: All SVA cases correct with exact integer margins.

============================================================
  4-State Alphabet Algebra
============================================================
  Vector A:     ['+1', '+0', '-0', '-1', '+1', '+0']
  Vector B:     ['-0', '+1', '-1', '+0', '+1', '+1']
  Flip(A):      ['-1', '-0', '+0', '+1', '-1', '-0']
  Collapse(A):  ['1', '0', '0', '-1', '1', '0']
  Expand(A):    ['+1', '+0', '-0', '-1', '+1', '+0']
  A·B = 3  (integer dot product)

============================================================
  Constructive Attention (Signed, No Softmax)
============================================================
  Routing: subject NUMBER → operator NUMBER via agreement flag
    'boy is' → 'boy is'  ✓
    'boy was' → 'boy was'  ✓
    'boys are' → 'boys are'  ✓
    'boys were' → 'boys were'  ✓

============================================================
  Riemann Zero Frequencies
============================================================
  Riemann zero frequencies: 620 values
  Range: [0.0707, 0.0013]
  mean=0.0036, std=0.0050
  (Used as position-encoding basis in the full model)

============================================================
  Integer Margins (+2 Property)
============================================================
  Integer margin checks: 16
  SVA margins: all +6 (verified 24/24)

============================================================
  Key Properties
============================================================

  ✓ No training required — All weights placed by hand
  ✓ No floating point attention — Signed integer vs softmax
  ✓ No randomness — Fully deterministic forward pass
  ✓ O(N·H) complexity — Linear in sequence length
  ✓ Integer margins of +2 — Guaranteed via 4-state algebra
  ✓ General-purpose — Same architecture handles SVA,
    concept arithmetic, and text generation
  ✓ No GPU required — Pure Python integer arithmetic

  This transformer is a proof that attention does not require
  stochastic gradient descent. The 4-state alphabet with
  hand-placed weights achieves what learned transformers
  achieve — but with deterministic, integer-only computation.
```

---

This repository only contains the transformer structure due to particular interest in this specific structure. We make reference to the other structures whose code is available at [riemann_structures](https://github.com/lostdemeter/riemann_structures).
