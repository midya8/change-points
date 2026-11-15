"""Pairwise latent-mean covariance, with partition uncertainty retained."""
import math
from .core import NEG, infer, integer


def _mix(terms):
    """Disjoint alternatives carrying centered first and cross moments."""
    if not terms:
        return (NEG, 0., 0., 0.)
    maximum = max(t[0] for t in terms)
    weights = [math.exp(t[0]-maximum) for t in terms]
    total = math.fsum(weights)
    return (maximum+math.log(total), *(
        math.fsum(w*t[j] for w, t in zip(weights, terms))/total
        for j in range(1, 4)))


def _pair(result, first, second):
    n = len(result['data'])
    model = result['settings']
    minimum = model['min_length']
    cap = model['max_segments_effective']
    anchors = [result['pointwise'][i]['latent_mean_posterior_mean']
               for i in (first, second)]
    scores = {(s['start'], s['end']): s for s in result['intervals']}
    states = {(0, 0): (0., 0., 0., 0.)}
    for k in range(1, cap+1):
        for end in range(k*minimum, n+1):
            terms = []
            for start in range((k-1)*minimum, end-minimum+1):
                log_mass, a, b, cross = states.get((k-1, start), (NEG, 0., 0., 0.))
                if log_mass == NEG:
                    continue
                segment = scores[start, end]
                has_a = start <= first < end
                has_b = start <= second < end
                x = segment['posterior_mean']-anchors[0] if has_a else 0.
                y = segment['posterior_mean']-anchors[1] if has_b else 0.
                # Given a partition, distinct segment means are independent.
                # Two indices in the same segment share its one random mean.
                within = segment['posterior_variance'] if has_a and has_b else 0.
                terms.append((log_mass+segment['log_marginal_likelihood'],
                              a+x, b+y, math.fsum((cross, a*y, b*x, x*y, within))))
            states[k, end] = _mix(terms)
    p = model['cut_probability']
    terms = []
    for k in range(1, cap+1):
        log_mass, a, b, cross = states[k, n]
        if log_mass != NEG:
            terms.append((log_mass+(k-1)*math.log(p)+(n-k)*math.log1p(-p), a, b, cross))
    log_mass, a, b, cross = _mix(terms)
    covariance = cross-a*b
    same_segment = math.fsum(s['posterior_probability'] for s in result['intervals']
                             if s['start'] <= first <= second < s['end'])
    return {'first_index': first, 'second_index': second,
            'latent_mean_covariance': covariance,
            'replicate_covariance': covariance+(model['noise_std']**2 if first == second else 0.),
            'same_segment_probability': same_segment,
            'centered_mean_residuals': [a, b],
            'log_joint_mass': log_mass}


def pair_covariance(data, pairs, **settings):
    """Evaluate 1..64 unique ordered index pairs, including the diagonal.

    Each pair is [i,j] with 0<=i<=j<N. Replicate covariance means one fresh
    observation at each distinct design index, with independent noise there.
    The diagonal is the variance of one replicate, not two repeated draws.
    """
    if not isinstance(data, list) or not 1 <= len(data) <= 120:
        raise ValueError('data requires1..120 observations')
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= 64:
        raise ValueError('pairs requires1..64 unique[i,j] pairs')
    validated = []
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError('pair requires[i,j]')
        a = integer(pair[0], 'first index', 0, len(data)-1)
        b = integer(pair[1], 'second index', 0, len(data)-1)
        if a > b:
            raise ValueError('pair requires first index <= second index')
        if (a, b) in validated:
            raise ValueError('duplicate pair')
        validated.append((a, b))
    result = infer(data, **settings)
    return {'inference': result,
            'pairs': [_pair(result, a, b) for a, b in validated],
            'semantics': 'posterior latent-mean covariance over the full admissible partition law, not only the MAP partition; same-segment mean uncertainty and between-partition mean dependence both retained; replicate covariance adds independent observation-noise variance on the diagonal only; same-design replicates conditional on all observed data, not forecasts, causal effects, independence tests or calibrated real-data claims; covariance and same-segment probability are different quantities; centered normalized moment recurrences in floating point, not symbolic exactness; pair queries do not specify a joint distribution',
            'query_work': {'pair_recurrences': len(validated),
                           'bound': 'O(pair_count * N^2 * effective_segment_cap), after the original full inference; no partition enumeration'}}
