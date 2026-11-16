"""Joint rectangular events for the latent path, not multiplied marginals."""
import math
from .core import NEG, infer, lse, real


def _normal_interval(lower, upper, mean, variance):
    """Same-tail erfc subtraction avoids subtracting two CDFs near one.

    This is still floating point: narrow or extreme tail intervals may round
    to zero. A represented zero is not proof of an impossible Gaussian event.
    """
    if lower >= upper:
        return 0.
    scale = math.sqrt(2*variance)
    a, b = (lower-mean)/scale, (upper-mean)/scale
    if a >= 0:
        return max(0., .5*(math.erfc(a)-math.erfc(b)))
    if b <= 0:
        return max(0., .5*(math.erfc(-b)-math.erfc(-a)))
    return .5*(math.erf(b)-math.erf(a))


def latent_envelope(data, lower, upper, **settings):
    """Probability every latent mean lies in its supplied index interval.

    Bounds are finite length-N lists in [-1e6,1e6], lower[i]<=upper[i]. A
    segment has ONE shared mean: its permissible set is the intersection of
    all its index intervals. Independent segment means can be multiplied only
    after conditioning on a complete partition. Observation noise is excluded.
    """
    if not isinstance(data, list) or not 1 <= len(data) <= 120:
        raise ValueError('data requires1..120 observations')
    n = len(data)
    bounds = []
    for values, name in [(lower, 'lower'), (upper, 'upper')]:
        if not isinstance(values, list) or len(values) != n:
            raise ValueError(name+' requires one bound per observation')
        bounds.append([real(x, name+' bound', -1e6, 1e6) for x in values])
    lower, upper = bounds
    if any(a > b for a, b in zip(lower, upper)):
        raise ValueError('lower bounds must not exceed upper bounds')
    result = infer(data, **settings)
    model = result['settings']
    minimum, cap = model['min_length'], model['max_segments_effective']
    intervals = []
    table = {}
    for segment in result['intervals']:
        a, b = segment['start'], segment['end']
        lo, hi = max(lower[a:b]), min(upper[a:b])
        mass = _normal_interval(lo, hi, segment['posterior_mean'], segment['posterior_variance'])
        row = {'start': a, 'end': b, 'intersection_lower': lo,
               'intersection_upper': hi, 'positive_width': lo < hi,
               'conditional_event_probability': mass,
               'positive_width_rounded_to_zero': lo < hi and mass == 0.}
        intervals.append(row)
        table[a,b] = (segment['log_marginal_likelihood'], mass, lo < hi)
    plain, event, possible = {(0,0): 0.}, {(0,0): 0.}, {(0,0): 1}
    transitions = 0
    for k in range(1, cap+1):
        for end in range(k*minimum, n+1):
            plain_terms, event_terms, count = [], [], 0
            for start in range((k-1)*minimum, end-minimum+1):
                score, mass, width = table[start,end]
                previous = plain.get((k-1,start), NEG)
                if previous != NEG:
                    plain_terms.append(previous+score)
                    transitions += 1
                prior_event = event.get((k-1,start), NEG)
                if mass > 0 and prior_event != NEG:
                    event_terms.append(prior_event+score+math.log(mass))
                if width:
                    count += possible.get((k-1,start), 0)
            plain[k,end] = lse(plain_terms)
            event[k,end] = lse(event_terms)
            possible[k,end] = count
    p = model['cut_probability']
    prior = {k: (k-1)*math.log(p)+(n-k)*math.log1p(-p) for k in range(1,cap+1)}
    denominator = lse(plain[k,n]+prior[k] for k in prior)
    numerator = lse(event[k,n]+prior[k] for k in prior)
    delta = numerator-denominator if numerator != NEG else None
    if delta is not None and delta > 1e-8:
        raise ValueError('envelope probability numerically unresolved; rescale inputs')
    log_probability = min(0., delta) if delta is not None else None
    probability = math.exp(log_probability) if log_probability is not None else 0.
    counts = []
    for k in prior:
        joint = math.exp(event[k,n]+prior[k]-denominator) if event[k,n] != NEG else 0.
        conditional = math.exp(event[k,n]+prior[k]-numerator) if numerator != NEG else None
        counts.append({'segments': k, 'event_and_count_probability': joint,
                       'count_probability_given_event': conditional,
                       'positive_volume_partition_count': possible[k,n]})
    marginals = []
    for index in range(n):
        terms = [s['posterior_probability']*_normal_interval(lower[index], upper[index], s['posterior_mean'], s['posterior_variance'])
                 for s in result['intervals'] if s['start'] <= index < s['end']]
        marginals.append(min(1., max(0., math.fsum(terms))))
    feasible = sum(row['positive_volume_partition_count'] for row in counts)
    return {'inference': result, 'lower': lower, 'upper': upper,
            'joint_probability': probability, 'log_joint_probability': log_probability,
            'positive_volume_partition_count': feasible,
            'numerical_zero_with_positive_volume': probability == 0. and feasible > 0,
            'segment_intersections': intervals, 'segment_count_given_event': counts,
            'pointwise_probabilities': marginals,
            'pointwise_product_comparison': math.prod(marginals),
            'union_bound_lower': max(0., 1-math.fsum(1-x for x in marginals)),
            'marginal_upper_bound': min(marginals),
            'query_work': {'reachable_segment_transitions': transitions,
                           'bound': 'O(N^3 + N^2*K) time and O(N^2+N*K) space after original inference; interval intersections and pointwise comparisons use simple interval scans; no partition enumeration'},
            'semantics': 'joint posterior event lower[i]<=latent_mean[i]<=upper[i] at every observed design index, conditional on all supplied data and fixed model settings; a segment has one shared Gaussian mean constrained by the intersection of its index bounds; products are over segments only within a partition, then mixed over the full partition law; excludes fresh observation noise, forecasts and real-data coverage guarantees; supplied envelope is evaluated, not fitted to a target coverage; pointwise product is only a comparison, not an independence assumption; finite-partition sums in floating point, erfc subtraction and exp may underflow or round, null log probability means no represented positive numerator, not necessarily geometric impossibility; conditional count probabilities are null only when that numerator is unrepresented; positive-volume counts ignore likelihood and use strict interval width, distinguishing model-zero events from numerical zeros'}
