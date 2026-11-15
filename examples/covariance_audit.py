"""Fixed pair-covariance examples; no sampling or accuracy claim."""
import itertools
import json
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from segmentlab import pair_covariance


def audit():
    data = [-1., -.8, .9, 1.2]
    pairs = [list(p) for p in itertools.combinations_with_replacement(range(4), 2)]
    result = pair_covariance(data, pairs, min_length=1, max_segments=3)
    row = next(r for r in result['pairs'] if (r['first_index'],r['second_index']) == (1,2))
    assert row['latent_mean_covariance'] < -.04
    assert .06 < row['same_segment_probability'] < .07
    shared = pair_covariance(data, [[0,0],[0,3]], max_segments=1)
    for item in shared['pairs']:
        assert abs(item['latent_mean_covariance']-1/(1/9+4/.25)) < 1e-14
    return {'partition_uncertainty': result, 'forced_one_segment': shared,
            'provenance': 'Fixed four-point shift, inspected during development. The central pair was selected after viewing all ten index pairs. This is a model-conditional counterexample, not held-out evidence or a typical-data claim.',
            'interpretation': 'A positive chance of sharing a segment does not force positive posterior covariance: uncertainty over the partition can move the two conditional means in opposite directions. Independent replicate noise changes diagonal variance only.'}


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2, allow_nan=False))
