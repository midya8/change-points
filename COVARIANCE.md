# Covariance across design indices

`pair_covariance` reports covariance of the latent segment means at supplied
indices. It averages over every admissible partition under the same Gaussian
model as `infer`. It does not replace that distribution with the MAP partition
or multiply marginal segment probabilities.

```sh
python -m segmentlab covariance examples/sequence.json --pairs '[[0,0],[0,3],[3,3]]'
python examples/covariance_audit.py > /tmp/covariance-audit.json
```

The Python entry point is `segmentlab.pair_covariance(data, pairs, **settings)`.
Pairs must be a list of 1..64 unique `[i,j]` lists with integer
`0 <= i <= j < len(data)`. The diagonal is allowed. Query order is preserved.
All original model bounds and minimum-length/count constraints apply.

## Two sources of covariance

Let `M_i` be the random segment mean at index `i`, and `P` a partition.
Conditional on `P`, different segment means are independent. Two indices in
the same segment share one Gaussian mean, so their conditional covariance is
that segment's posterior variance. Averaging over `P` also adds covariance
between the conditional means:

```text
Cov(M_i,M_j | data)
  = E[Cov(M_i,M_j | P,data) | data]
    + Cov(E[M_i | P,data], E[M_j | P,data] | data).
```

The second term can be negative. A positive probability of sharing a segment
therefore does not imply positive total covariance. `same_segment_probability`
is a separate output, not an estimate of covariance or a claim of independence.

The fixed four-point audit uses `[-1,-.8,.9,1.2]`, minimum length 1 and at most
3 segments, with the ordinary default model parameters. For indices `[1,2]`,
the same-segment probability is about `0.0629123`, but latent covariance is
about `-0.0460892`. All ten index pairs are retained. The central pair was
selected after inspecting that table; this is a counterexample under the
supplied model, not evidence about typical data or real-world accuracy.

## Centered moment recurrence

For each query, a prefix/count state carries a log mass and normalized moments
`E[A]`, `E[B]`, `E[AB]`. `A` and `B` are centered latent means contributed by
intervals covering the two marked indices; unvisited indices contribute zero.
The anchors are the existing pointwise posterior means from `infer`.

Appending an interval contributes centered conditional means `x,y`. The new
cross moment is `old_cross + old_A*y + old_B*x + x*y + v`, where `v` is the
segment posterior variance if both indices are in the appended interval and
zero otherwise. Alternatives are mixed after subtracting their largest log
mass. The final count mixture includes the original Bernoulli cut-prior terms.
Subtracting `E[A]*E[B]` gives covariance without subtracting two large raw
second moments. No partition enumeration is used in the implementation.

After the original full inference, each pair costs `O(N^2*K)` work and
`O(N*K)` state, where `K` is the effective segment cap. Queries are evaluated
one at a time. This is an operation bound, not a timing claim.

## Replicate observations and limits

`replicate_covariance` describes one fresh observation at each distinct design
index, sharing the posterior partition and segment means but with independent
observation noise there. Off-diagonal covariance equals latent covariance.
On the diagonal, it adds the supplied noise variance. A diagonal query is the
variance of one replicate, not covariance between two independently repeated
observations at the same index.

These are same-design, model-conditional moments using all observed data.
They are not forecasts, causal effects, a joint predictive distribution,
independence tests or calibrated real-data guarantees. Zero covariance does
not establish independence. Floating-point normalization and underflow still
apply; centered recurrences do not make the underlying scores symbolic or
repair an inference that fails the existing numerical guards. No clipping of
negative off-diagonal covariance is used.

## Checks

Eight tests cover all pairs in 20 seeded tiny models against complete partition
enumeration. The reference uses independent conjugate mean/variance formulas
and 90-digit Decimal normalization of represented float partition log scores.
It does not claim those float scores are exact real Gaussian likelihoods.
Other checks cover the forced-one-segment closed form, diagonal agreement with
`infer`, the negative-covariance counterexample, offsets near both input limits,
sign reflection, 81 covariance quadratic forms, maximum query/input bounds,
strict input handling, order/isolation, the command line and the complete audit.

Three deliberate errors are rejected: removing shared-mean variance, removing
between-partition dependence, and adding observation noise off the diagonal.
The complete 112-test suite and installed-wheel checks retain the original
tests, reports and both figure checks. The original demo and all five existing
feature audits remain byte-for-byte unchanged.
