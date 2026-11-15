import copy
from decimal import Decimal, localcontext
import itertools
import json
import math
import random
import subprocess
import sys
import unittest
from segmentlab import infer, pair_covariance
from test_core import enumerate_partitions


def reference(data, first, second, settings):
    """Enumerate complete partitions, then mix independent conjugate moments.

    Decimal normalizes the represented float log scores at 90 digits. It does
    not turn those input scores into exact real Gaussian likelihoods.
    """
    parts = enumerate_partitions(data, sigma=settings.get('noise_std', .5),
                                 mu=settings.get('prior_mean', 0.),
                                 tau=settings.get('prior_std', 3.),
                                 p=settings.get('cut_probability', .08),
                                 minimum=settings.get('min_length', 2),
                                 cap=settings.get('max_segments', 8))
    with localcontext() as context:
        context.prec = 90
        D = Decimal.from_float
        sigma2 = D(float(settings.get('noise_std', .5)))**2
        tau2 = D(float(settings.get('prior_std', 3.)))**2
        mu = D(float(settings.get('prior_mean', 0.)))
        maximum = max(p[3] for p in parts)
        weights = [(D(p[3])-D(maximum)).exp() for p in parts]
        total = sum(weights)
        rows = []
        for weight, (_, segments, _, _) in zip(weights, parts):
            a = next(s for s in segments if s[0] <= first < s[1])
            b = next(s for s in segments if s[0] <= second < s[1])
            def moments(segment):
                start, end = segment
                variance = 1/(1/tau2+Decimal(end-start)/sigma2)
                mean = variance*(mu/tau2+sum(D(float(x)) for x in data[start:end])/sigma2)
                return mean, variance
            ma, va = moments(a)
            mb, _ = moments(b)
            rows.append((weight/total, ma, mb, va if a == b else Decimal(0), a == b))
        ea = sum(w*a for w,a,b,v,s in rows)
        eb = sum(w*b for w,a,b,v,s in rows)
        covariance = sum(w*(v+(a-ea)*(b-eb)) for w,a,b,v,s in rows)
        same = sum(w for w,a,b,v,s in rows if s)
        return float(covariance), float(same)


class CovarianceTests(unittest.TestCase):
    def test_all_pairs_against_full_partition_reference(self):
        rng = random.Random(7319)
        for trial in range(20):
            n = rng.randrange(1, 8)
            data = [rng.uniform(-2, 2) for _ in range(n)]
            settings = {'noise_std': rng.uniform(.3, 1.4), 'prior_std': rng.uniform(.4, 3.),
                        'prior_mean': rng.uniform(-1, 1), 'cut_probability': rng.uniform(.02, .8),
                        'min_length': rng.randrange(1, n+1), 'max_segments': rng.randrange(1, 6)}
            pairs = [list(p) for p in itertools.combinations_with_replacement(range(n), 2)]
            result = pair_covariance(data, pairs, **settings)
            for row, (a,b) in zip(result['pairs'], pairs):
                expected, same = reference(data, a, b, settings)
                self.assertAlmostEqual(row['latent_mean_covariance'], expected, 11)
                self.assertAlmostEqual(row['same_segment_probability'], same, 11)
                self.assertAlmostEqual(row['replicate_covariance'], expected+(settings['noise_std']**2 if a == b else 0.), 11)

    def test_shared_mean_and_noise_diagonal_closed_form(self):
        result = pair_covariance([.2,.4,.3], [[0,0],[0,2],[2,2]], max_segments=1)
        variance = 1/(1/9+3/.25)
        for row in result['pairs']:
            self.assertAlmostEqual(row['latent_mean_covariance'], variance, 15)
            self.assertEqual(row['same_segment_probability'], 1.)
        self.assertAlmostEqual(result['pairs'][0]['replicate_covariance'], variance+.25, 15)
        self.assertEqual(result['pairs'][1]['replicate_covariance'], result['pairs'][1]['latent_mean_covariance'])

    def test_diagonal_and_inference_unchanged(self):
        data = [-1.,-.8,.9,1.2]
        settings = {'min_length':1,'max_segments':3}
        result = pair_covariance(data, [[i,i] for i in range(4)], **settings)
        self.assertEqual(result['inference'], infer(data, **settings))
        for row, point in zip(result['pairs'], result['inference']['pointwise']):
            self.assertAlmostEqual(row['latent_mean_covariance'], point['latent_mean_posterior_variance'], 13)
            self.assertAlmostEqual(row['same_segment_probability'], 1., 13)

    def test_partition_dependence_can_make_covariance_negative(self):
        result = pair_covariance([-1.,-.8,.9,1.2], [[1,2]], min_length=1, max_segments=3)
        row = result['pairs'][0]
        self.assertGreater(row['same_segment_probability'], .06)
        self.assertLess(row['latent_mean_covariance'], -.04)
        # Keeping only shared-segment variance would be strictly positive.
        shared = sum(s['posterior_probability']*s['posterior_variance']
                     for s in result['inference']['intervals'] if s['start'] <= 1 <= 2 < s['end'])
        self.assertGreater(shared, 0.)

    def test_centering_offsets_sign_and_psd_quadratic_forms(self):
        data = [-1.,-.8,.9,1.2]
        pairs = [list(p) for p in itertools.combinations_with_replacement(range(4), 2)]
        base = pair_covariance(data, pairs, min_length=1, max_segments=3)
        for offset, sign in [(1e6-2,1),(-1e6+2,1),(0,-1)]:
            shifted = pair_covariance([offset+sign*x for x in data], pairs, prior_mean=offset, min_length=1,max_segments=3)
            for a,b in zip(base['pairs'],shifted['pairs']):
                self.assertAlmostEqual(a['latent_mean_covariance'],b['latent_mean_covariance'],8)
        matrix = [[0.]*4 for _ in range(4)]
        for row in base['pairs']:
            i,j = row['first_index'],row['second_index']
            matrix[i][j] = matrix[j][i] = row['latent_mean_covariance']
        for vector in itertools.product((-1,0,1), repeat=4):
            self.assertGreaterEqual(math.fsum(vector[i]*matrix[i][j]*vector[j] for i in range(4) for j in range(4)), -1e-12)

    def test_validation_order_and_isolation(self):
        data = [0.,1.,2.];pairs = [[2,2],[0,1],[0,0]]
        before = copy.deepcopy((data,pairs))
        result = pair_covariance(data,pairs,min_length=1)
        self.assertEqual((data,pairs), before)
        self.assertEqual([[r['first_index'],r['second_index']] for r in result['pairs']],pairs)
        for invalid in [None,{},[],[[0,0]]*65,[[0,0],[0,0]],[[1,0]],[[True,1]],[[0,1.]],[(0,1)],[[-1,1]],[[0,3]],[[0]],[[0,1,2]]]:
            with self.assertRaises(ValueError):pair_covariance(data,invalid,min_length=1)
        with self.assertRaises(ValueError):pair_covariance([],[[0,0]])
        with self.assertRaises(ValueError):pair_covariance(data,[[0,0]],noise_std=0)

    def test_bounds_extremes_and_cli(self):
        pairs = [[i,i] for i in range(64)]
        result = pair_covariance([0.]*120,pairs,min_length=60,max_segments=12)
        self.assertEqual(len(result['pairs']),64)
        json.dumps(result,allow_nan=False)
        for data in [[-1e6,1e6],[1e6,1e6]]:
            result = pair_covariance(data,[[0,0],[0,1]],min_length=1,noise_std=1e-4,prior_std=1e4)
            json.dumps(result,allow_nan=False)
        args = [sys.executable,'-m','segmentlab','covariance','examples/sequence.json','--pairs','[[0,0],[0,3]]']
        p = subprocess.run(args,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(len(json.loads(p.stdout)['pairs']),2)
        for value in ['bad','[[1,0]]','[[true,1]]']:
            p = subprocess.run(args[:-1]+[value],capture_output=True,text=True)
            self.assertEqual(p.returncode,2)
            self.assertNotIn('Traceback',p.stderr)

    def test_complete_retained_audit(self):
        p = subprocess.run([sys.executable,'examples/covariance_audit.py'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        result = json.loads(p.stdout)
        self.assertEqual(len(result['partition_uncertainty']['pairs']),10)
        self.assertEqual(len(result['forced_one_segment']['pairs']),2)
