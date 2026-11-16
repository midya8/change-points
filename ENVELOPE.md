# Supplied joint latent-path envelopes

This evaluates one supplied rectangular event, not a procedure that chooses a
band for target coverage. The model, data bounds and conditioned partition prior
are unchanged. Every latent value at an index is the mean of its covering
segment, not a new observation. Fresh observation noise is excluded.

## API and command

```python
from segmentlab import latent_envelope
report = latent_envelope([0., 0.], [-1., .1], [-.1, 1.],
                         min_length=1, max_segments=2)
```

`lower` and `upper` must be length-N lists of finite real numbers in [-1e6,1e6].
Booleans are rejected. Each lower bound must be at most its corresponding upper
bound; equality is allowed and gives a zero-measure event at that index. All
original model validation still applies, with API defaults unchanged. Neither
input lists nor RNG state are changed. CLI payloads retain the same strict seven
model fields; bounds are separate required JSON arguments:

```sh
python -m segmentlab envelope examples/sequence.json --lower '[-2,-2,-2,-2,-2,-2]' --upper '[2,2,2,2,2,2]'
python examples/envelope_audit.py > /tmp/envelope-audit.json
```

## Sum over partitions, product over distinct segment means

For a segment [a,b), take L=max(lower[a:b]) and U=min(upper[a:b]). Its
conditional event mass q(a,b) is the posterior Gaussian mean probability of
[L,U], or zero when L>=U. A segment mean is shared at every covered index.
Multiplying one Gaussian probability per index would count the same random
variable several times.

The usual likelihood forward recurrence F(k,b) sums over starts a. A second
recurrence G(k,b) multiplies each transition by q(a,b):

```
F(k,b) = sum_a F(k-1,a) * likelihood(a,b)
G(k,b) = sum_a G(k-1,a) * likelihood(a,b) * q(a,b)
F(0,0) = G(0,0) = 1
```

Only admissible minimum-length transitions are allowed. Values are accumulated
in log space. With the same complete-partition Bernoulli weight w(k) used by
`infer`, the result is sum_k w(k)G(k,N) / sum_k w(k)F(k,N). The prior
admissibility normalizer cancels in this posterior ratio. Multiplying prior
weights at every transition instead would silently change the prior.

The report retains every segment intersection and its Gaussian mass, all count
contributions P(event,K=k), and P(K=k|event). The conditional count law can differ
from the original partition posterior. It does not claim to return every other
conditional partition marginal or a new fitted model.

`pointwise_probabilities` use the original covering-interval marginals. Their
product is only a comparison. The report also supplies the ordinary union-bound
lower bound and minimum-marginal upper bound for the joint event. Neither bound
requires independence. Last-bit floating error can slightly perturb these
comparisons; they are not formal arithmetic certificates.

## Fixed examples, all retained

- Four zeros, default Gaussian settings, min_length=1 and max_segments=1:
  [-.5,.5] at every index has joint probability .955243141782119. Multiplying
  marginals gives .8326370144683303. One shared Gaussian variable is involved.
- Two zeros with index intervals [-1,-.1] and [.1,1], min_length=1 and
  max_segments=2: joint probability .003123780167675003. Only the two-segment
  partition can satisfy the event, so P(K=2|event)=1. The product of marginals
  is .14895596841923653 and is not its joint law.
- The same disjoint intervals with max_segments=1 have zero joint probability,
  even though both individual index events have positive marginal probability.
- The complete audit retains all three common half-widths .5,1,2 for data
  [-.3,.8,-.1,1], min_length=1,max_segments=3. Joint probabilities are
  .6764719841969786, .9665551230607787 and .9991309476746694. These settings
  are development examples selected to expose dependence, not held-out data.

## Zero events and numerical limits

An integer recurrence separately counts admissible partitions whose every
segment intersection has strictly positive width. All Gaussian means have
positive variance. Thus a zero positive-volume count establishes a zero event
under the stated model, without relying on CDF arithmetic. This is not a claim
about observations or actual event detection.

Same-tail `erfc` differences avoid subtracting two CDFs close to one, and
straddling intervals use `erf`. These are still floating-point calculations:
very narrow or extreme intervals may round to zero, and final exponentiation
may underflow. A positive-volume event with represented probability zero is
flagged separately. `log_joint_probability` is null if no positive numerator
was represented; it is not proof that the true log probability is minus
infinity. If the numerator survives but final exponentiation underflows, the
finite log probability and conditional count law remain available. Conditional
count probabilities are null when the numerator itself is unrepresented.
Even a finite log result can omit tiny rounded segment masses. No arbitrary
precision tail-accuracy guarantee is made.

Bounds are evaluated against the posterior conditional on all the data and the
supplied Gaussian assumptions. This is not a forecast, replicate-observation
band, frequentist simultaneous coverage claim or protection against a wrong
noise model. There is no target-coverage band search or hyperparameter fit.

## Checks and costs

Ten tests add 32 seeded tiny models with all admissible gap bitsets enumerated.
The separate oracle scores each complete partition by sequential conjugate
predictive densities, calculates its segment means sequentially and integrates
Gaussian densities with adaptive Simpson quadrature. It uses no production
interval score, CDF helper or forward state. The quadrature clips standard
normal integration at +/-12 for these moderate test fixtures only; omitted
mass is negligible at their asserted tolerance, not a production tail method.
Tests compare joint/count probabilities and positive-volume counts, shared-mean
and conflicting-bound closed cases, time reversal/sign/offset symmetries,
nesting, union bounds, numerical zeros, strict inputs, full CLI reports, and
N=120 limits. A representable far-tail test catches ordinary CDF cancellation.
The original 112 tests and full reports remain unchanged.

After original inference, the extra forward recurrences cost O(N^2*K) and store
O(N*K) states. This simple implementation scans bounds inside every interval
and scans intervals for pointwise comparisons: total extra time O(N^3+N^2*K),
space O(N^2+N*K). It enumerates no complete partitions in production and makes
no wall-time or scalability claim. The Gaussian model and partition framework
are those already attributed in DESIGN.md; this is a direct event-weighted
recurrence, not a new statistical theorem or a paper reproduction.
