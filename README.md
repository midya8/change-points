# Change points

This implementation infers partitions of a supplied sequence under a Bayesian Gaussian
changepoint model. Inspect the partition law, draw complete partitions and
choose boundaries under an explicit loss. A confident posterior can still be
wrong when the supplied noise model is wrong.

Python 3.10+, no runtime dependencies, local data only, no paid API or GPU.

The same isolated outlier is assigned a high-probability two-cut segment under a low supplied noise scale. That is model confidence, not evidence that an event occurred.

![Noise-assumption failure and known shifts](assets/partition-audit.svg)

The repository is now `change-points`. The Python import and CLI names stay
`segmentlab` for compatibility with existing scripts.

## Run it

From the repository root:

```sh
python -m unittest discover -s tests -v
python tools/check_installed.py
python -m segmentlab --help
python -m segmentlab infer examples/sequence.json
python -m segmentlab sample examples/sequence.json --draws 4 --seed 23
python -m segmentlab joint examples/sequence.json --pairs '[[1,3],[2,4]]'
python -m segmentlab information examples/sequence.json
python -m segmentlab decide examples/sequence.json --false-cut-cost 1 --missed-cut-cost 1
python -m segmentlab predictive examples/sequence.json --queries '[[2,0.0],[3,1.0]]'
python -m segmentlab envelope examples/sequence.json --lower '[-2,-2,-2,-2,-2,-2]' --upper '[2,2,2,2,2,2]'
python -m segmentlab covariance examples/sequence.json --pairs '[[0,0],[0,3],[3,3]]'
```

Every model command reads exactly these seven fields, with no extra keys:

```json
{
  "data": [
    -1.1,
    -0.9,
    -1.0,
    1.0,
    0.9,
    1.1
  ],
  "noise_std": 0.2,
  "prior_mean": 0.0,
  "prior_std": 3.0,
  "cut_probability": 0.1,
  "min_length": 2,
  "max_segments": 3
}
```

`demo` takes no file. Output is JSON; invalid inputs exit 2 with an error on stderr. Python APIs accept model defaults, but the CLI intentionally requires a complete model payload. `python -m segmentlab COMMAND --help` describes each command's options, schema and bounds.

| Command / API | What it returns | What it does not mean |
|---|---|---|
| `infer` / `infer` | Conditioned-prior evidence; count, interval and boundary marginals; complete MAP; latent-mean moments | Marginal peaks are not a joint partition; latent variance excludes future observation noise |
| `sample` / `sample_posterior` | Seeded complete partitions and conditional Gaussian segment means | No MCMC or credible-band/calibration claim; zero visits do not imply zero mass |
| `joint` / `joint_cuts` | Joint inclusion and both conditional directions for requested cut pairs | Other cuts remain allowed; pairwise summaries do not determine the full law |
| `information` / `partition_information` | Full partition/count entropy, conditioned-prior KL, binary-cut entropy gap | Lower entropy or higher KL does not prove a correct model |
| `envelope` / `latent_envelope` | Joint probability a supplied latent-path envelope contains every index; segment-count law conditional on that event | Not multiplied marginals, replicate bands, forecasts or real-data coverage |
| `covariance` / `pair_covariance` | Latent-mean and same-design replicate covariance across index pairs | Not same-segment probability, independence tests or forecasts |
| `predictive` / `posterior_predictive` | Same-design replicate mixture density, CDF/survival and 5/50/95% quantiles | Not forecasts, leave-one-out scores, anomaly p-values or simultaneous bands |
| `decide` / `boundary_decision` | Admissible minimum expected false/missed boundary loss; MAP/threshold comparisons | A supplied-loss optimum is not observed real-world accuracy |

All APIs are exported from `segmentlab`. For formulas, exact defaults and output semantics, see [the technical reference](DESIGN.md#explicit-model-and-prior). The reference preserves every fixture, implementation audit, bound and limitation.

## What is being inferred?

Each segment has an independent mean drawn from `Normal(prior_mean,prior_std²)`. Observations in that segment are conditionally independent `Normal(mean,noise_std²)` with **known supplied noise**. Independent Bernoulli interior-gap cut weights are **conditioned on minimum segment length and a segment-count cap**. The prior normalizer is computed and included in data evidence. Nothing here learns the noise, prior or cut probability.

All inference is offline: later observations inform earlier cuts and means. Intervals are half-open `[start,end)`; cut j means a boundary before zero-based observation j. No slopes, correlated errors or robust outlier component are included.

## Inspect the failure cases

- **Noise assumption:** the same 24 observations (zeros except 4 at index 12) yield MAP cuts `[12,13]` with probability.960218 under sigma .15, versus no cuts with probability.648223 under sigma 2. Both use the same mean prior and partition constraints. Different sigmas define different models, not a quality ranking.
- **Joint law:** two individually possible cuts can be jointly forbidden by a count cap. For six zeros and cap 2, cuts 2/4 each have mass .011658220257; their exact joint is 0, while multiplying marginals invents mass .000135914100.
- **Decision objective:** a selected seven-point fixture has MAP `[3]`, but no cuts minimize equal-cost expected boundary loss (.680000591 versus MAP's .869071786). MAP instead minimizes whole-partition 0/1 loss. These are posterior expectations, not measured mistakes.
- **Uncertainty:** the four-point fixture's posterior entropy 1.029013408 exceeds prior entropy .836308115, yet KL is positive .166962288. KL is not prior entropy minus posterior entropy.

These are selected development examples, not held-out performance, event detection, causal attribution or real-data calibration. [The reference](DESIGN.md) states every input, setting, selection provenance and numerical qualification.

## Replicate uncertainty is a mixture

`predictive` integrates over covering segments and their Gaussian mean posteriors, then adds fresh observation noise. At an existing index, latent variance and replicate variance are different. A Gaussian with matching mean/variance need not match the mixture density: in a selected five-point outlier fixture, its log density at 2 is 1.54 nats higher than the exact mixture. [Inputs, two-prior comparison and numerical quantile audit](DESIGN.md#same-design-replicate-predictive-mixtures). Fit uses all observations, including the queried index. These are not forecasts, leave-one-out scores, anomaly p-values or coverage claims.

## Covariance is not the chance of sharing a segment

`covariance` retains uncertainty over complete partitions and their Gaussian segment means. In a fixed four-point shift, indices 1 and 2 have a 0.0629123 chance of sharing a segment, yet their latent covariance is -0.0460892. Partition uncertainty can move their conditional means in opposite directions. The audit keeps all ten index pairs and a forced-one-segment comparison; the central pair was selected after inspecting the table. Independent replicate noise adds variance only on the diagonal. [Recurrence, validation and limits](COVARIANCE.md).

## A whole-path envelope is one joint event

`envelope` evaluates supplied lower/upper bounds for every latent index. A
segment has one shared mean, so its allowed set is the intersection of all
index intervals it covers. The recurrence multiplies independent Gaussian
segment probabilities within each partition, then mixes over partitions.
It does not multiply pointwise marginal probabilities.

For four zeros forced into one segment, bounds [-.5,.5] at every index have
joint posterior probability .955243142, while the product of the four
marginals is .832637014. With two zeros, disjoint index intervals [-1,-.1]
and [.1,1] require a cut: the event has probability .003123780 and its
conditional segment count is two with probability one. Forcing one segment
makes that same event have probability zero. These are fixed model-conditional
examples, not measured coverage. [Recurrence, numerical zeros and full retained
comparisons](ENVELOPE.md).

## Complete paths versus marginal averages

![Joint posterior draw audit](assets/joint-draw-audit.svg)

The first three generated latent paths are shown without selecting for fit. Count/cut frequencies and latent variances use **all 10,000 draws**. All observations, counts, gaps and variance points are retained, with numeric SVG metadata. K4 has positive exact mass .000115427 but zero sampled visits. The figure is not a credible band. [Sampling method and finite-RNG limits](DESIGN.md#joint-posterior-draws-not-thresholded-marginal-cuts).

## Reproduce the audits

```sh
python -m segmentlab demo > /tmp/segment-demo.json
python examples/render_demo.py /tmp/segment-demo.json --check
python examples/sample_audit.py > /tmp/sample-audit.json
python examples/render_samples.py /tmp/sample-audit.json --check
python examples/joint_audit.py > /tmp/joint-audit.json
python examples/information_audit.py > /tmp/information-audit.json
python examples/decision_audit.py > /tmp/decision-audit.json
python examples/predictive_audit.py > /tmp/predictive-audit.json
python examples/covariance_audit.py > /tmp/covariance-audit.json
python examples/envelope_audit.py > /tmp/envelope-audit.json
```

**122 tests.** Independent tiny exhaustive partition references check evidence, marginals, MAP, variance, sampler path probabilities, joint queries, entropy/KL and decision risk. Other tests cover sequential conjugate evidence, offsets/extremes, strict inputs, RNG/input isolation, work counts, predictive mixture densities/tails/quantiles, centered pair covariance against 20 seeded exhaustive models, CLI contracts and SVG geometry/regeneration. CI runs Python 3.10..3.13 plus both figure checks. [Detailed validation and limits](DESIGN.md#validation-and-limits).

Inputs are bounded: 1..120 observations and prior mean in [-1e6,1e6], noise/prior std in [1e-4,1e4], p in [1e-6,1-1e-6], integer minimum length 1..N and requested cap 1..12. Exact means finite-partition sums **in floating point**, not symbolic arithmetic: negligible mass can underflow, near ties can move and numerical guards are not formal certificates. Full marginal inference costs `O(N²*K²)`; feature-specific costs and output bounds are in the reference. No streaming alerts, forecasts, learned hyperparameters, credible-band coverage or model-misspecification guarantee.

## Read further

- [Model, prior and conjugate evidence](DESIGN.md#explicit-model-and-prior)
- [Rolling interval work audit](DESIGN.md#rolling-interval-statistics-without-posterior-changes)
- [Direct joint sampling](DESIGN.md#joint-posterior-draws-not-thresholded-marginal-cuts)
- [Exact joint cut queries](DESIGN.md#exact-joint-cut-inclusion-queries)
- [Full partition entropy and KL](DESIGN.md#full-partition-entropy-and-information-in-nats)
- [Constrained boundary-loss decisions](DESIGN.md#constrained-bayes-decisions-under-boundary-loss)

Independent implementation inspired by [Fearnhead, Exact and Efficient Bayesian Inference for Multiple Changepoint Problems](https://people.csail.mit.edu/siracusa/papers/fearnhead06.pdf). This uses its own explicitly conditioned prior and capped recurrences, not copied code, paper experiments, pruning or a reproduction of the paper's sampling algorithm. [Attribution details](DESIGN.md#sources-and-attribution). MIT licensed.
- [Same-design replicate predictive mixtures](DESIGN.md#same-design-replicate-predictive-mixtures)

Pair covariance has its own [reference and limits](COVARIANCE.md). The package check runs all 122 tests from the installed wheel, verifies eleven module origins, preserves the original seven-report/two-entrypoint checks and reproduces the complete demo, all seven feature audits and both figures outside the source checkout.

The envelope tests add a separate exhaustive gap-bitset oracle with sequential
conjugate likelihoods and adaptive Simpson Gaussian integration, rather than
using the production interval scores, CDF calls or recurrence. The installed
check also compares the full envelope CLI report through both entrypoints.
