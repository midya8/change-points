import copy
import itertools
import json
import math
import random
import subprocess
import sys
import unittest
from segmentlab import infer, latent_envelope


def quadrature(a, b, mean, variance):
    """Independent adaptive Simpson integration of the standard normal density."""
    a, b = max(-12., (a-mean)/math.sqrt(variance)), min(12., (b-mean)/math.sqrt(variance))
    if a >= b:
        return 0.
    def f(x): return math.exp(-x*x/2)/math.sqrt(2*math.pi)
    def recurse(a,b,fa,fm,fb,whole,tol,depth):
        m=(a+b)/2;lm=(a+m)/2;rm=(m+b)/2
        fl,fr=f(lm),f(rm)
        left=(m-a)*(fa+4*fl+fm)/6;right=(b-m)*(fm+4*fr+fb)/6
        delta=left+right-whole
        if depth==0 or abs(delta)<=15*tol:
            return left+right+delta/15
        return recurse(a,m,fa,fl,fm,left,tol/2,depth-1)+recurse(m,b,fm,fr,fb,right,tol/2,depth-1)
    fa,fm,fb=f(a),f((a+b)/2),f(b)
    whole=(b-a)*(fa+4*fm+fb)/6
    return recurse(a,b,fa,fm,fb,whole,min(2e-13,abs(whole)*1e-10),24)


def reference(data, lower, upper, **settings):
    """Enumerate gap bitsets; sequential conjugate predictive likelihood.

    Does not use production interval scores, Gaussian CDFs or DP states.
    """
    sigma2=settings.get('noise_std',.5)**2;tau2=settings.get('prior_std',3.)**2
    mu=settings.get('prior_mean',0.);p=settings.get('cut_probability',.08)
    minimum=settings.get('min_length',2);cap=settings.get('max_segments',8)
    rows=[]
    for bits in itertools.product((0,1),repeat=len(data)-1):
        edges=[0]+[i+1 for i,b in enumerate(bits) if b]+[len(data)]
        segments=list(zip(edges,edges[1:]));k=len(segments)
        if k>cap or any(b-a<minimum for a,b in segments): continue
        logmass=(k-1)*math.log(p)+(len(data)-k)*math.log1p(-p)
        event=1.;positive=True
        for a,b in segments:
            mean=mu;v=tau2
            for y in data[a:b]:
                predictive=v+sigma2
                logmass+=-.5*(math.log(2*math.pi*predictive)+(y-mean)**2/predictive)
                gain=v/predictive;mean+=gain*(y-mean);v*=1-gain
            lo,hi=max(lower[a:b]),min(upper[a:b])
            positive &= lo<hi
            event*=quadrature(lo,hi,mean,v)
        rows.append((k,logmass,event,positive))
    peak=max(r[1] for r in rows);den=math.fsum(math.exp(r[1]-peak) for r in rows)
    weighted=[(k,math.exp(logmass-peak)/den,event,positive) for k,logmass,event,positive in rows]
    joint=math.fsum(w*e for k,w,e,pos in weighted)
    counts={k: math.fsum(w*e for kk,w,e,pos in weighted if kk==k) for k in range(1,min(cap,len(data)//minimum)+1)}
    return joint,counts,sum(pos for k,w,e,pos in weighted)


class EnvelopeTests(unittest.TestCase):
    def test_seeded_exhaustive_partitions_and_independent_quadrature(self):
        rng=random.Random(7492)
        for trial in range(32):
            n=rng.randrange(1,8);data=[rng.uniform(-1.5,1.5) for _ in range(n)]
            lower=[rng.uniform(-2,.2) for _ in data];upper=[x+rng.uniform(.1,3) for x in lower]
            settings=dict(noise_std=rng.uniform(.3,1.4),prior_std=rng.uniform(.4,2),prior_mean=rng.uniform(-.5,.5),cut_probability=rng.uniform(.05,.8),min_length=rng.randrange(1,n+1),max_segments=rng.randrange(1,7))
            expected,counts,feasible=reference(data,lower,upper,**settings)
            result=latent_envelope(data,lower,upper,**settings)
            self.assertAlmostEqual(result['joint_probability'],expected,11)
            self.assertEqual(result['positive_volume_partition_count'],feasible)
            for row in result['segment_count_given_event']:
                k=row['segments'];self.assertAlmostEqual(row['event_and_count_probability'],counts[k],11)
                if expected>0:self.assertAlmostEqual(row['count_probability_given_event'],counts[k]/expected,10)
            self.assertGreaterEqual(result['joint_probability']+1e-11,result['union_bound_lower'])
            self.assertLessEqual(result['joint_probability'],result['marginal_upper_bound']+1e-11)

    def test_shared_one_segment_not_indexwise_product(self):
        result=latent_envelope([0.]*4,[-.5]*4,[.5]*4,min_length=1,max_segments=1)
        expected=quadrature(-.5,.5,0.,1/(1/9+4/.25))
        self.assertAlmostEqual(result['joint_probability'],expected,12)
        self.assertAlmostEqual(result['pointwise_product_comparison'],expected**4,12)
        self.assertGreater(result['joint_probability']-result['pointwise_product_comparison'],.1)

    def test_conflicting_intervals_change_conditional_partition_law(self):
        result=latent_envelope([0.,0.],[-1.,.1],[-.1,1.],min_length=1,max_segments=2)
        self.assertEqual(result['positive_volume_partition_count'],1)
        self.assertEqual(result['segment_count_given_event'][0]['count_probability_given_event'],0.)
        self.assertEqual(result['segment_count_given_event'][1]['count_probability_given_event'],1.)
        impossible=latent_envelope([0.,0.],[-1.,.1],[-.1,1.],min_length=1,max_segments=1)
        self.assertEqual(impossible['joint_probability'],0.)
        self.assertEqual(impossible['positive_volume_partition_count'],0)
        self.assertFalse(impossible['numerical_zero_with_positive_volume'])
        self.assertIsNone(impossible['log_joint_probability'])
        self.assertIsNone(impossible['segment_count_given_event'][0]['count_probability_given_event'])

    def test_zero_width_and_numerical_zero_are_distinct(self):
        r=latent_envelope([0.],[0.],[0.],min_length=1)
        self.assertEqual(r['positive_volume_partition_count'],0)
        r=latent_envelope([0.],[999.],[1000.],min_length=1)
        self.assertEqual(r['joint_probability'],0.)
        self.assertEqual(r['positive_volume_partition_count'],1)
        self.assertTrue(r['numerical_zero_with_positive_volume'])
        self.assertTrue(r['segment_intersections'][0]['positive_width_rounded_to_zero'])

    def test_same_tail_subtraction_preserves_representable_tail(self):
        r=latent_envelope([0.],[4.5],[4.7],min_length=1)
        # Ordinary CDF subtraction rounds both endpoints to one here.
        variance=1/(1/9+4)
        from statistics import NormalDist
        normal=NormalDist(0,math.sqrt(variance))
        self.assertEqual(normal.cdf(4.7)-normal.cdf(4.5),0.)
        self.assertGreater(r['joint_probability'],0.)
        self.assertAlmostEqual(r['joint_probability']/quadrature(4.5,4.7,0.,variance),1.,5)

    def test_permutation_free_time_reversal_sign_offset(self):
        y=[-.3,.8,-.1,1.];lo=[-1.,-.2,-.5,.2];hi=[.2,1.1,.9,1.5]
        base=latent_envelope(y,lo,hi,min_length=1,max_segments=3)
        reverse=latent_envelope(y[::-1],lo[::-1],hi[::-1],min_length=1,max_segments=3)
        sign=latent_envelope([-x for x in y],[-x for x in hi],[-x for x in lo],min_length=1,max_segments=3)
        shift=latent_envelope([x+7 for x in y],[x+7 for x in lo],[x+7 for x in hi],prior_mean=7,min_length=1,max_segments=3)
        for r in [reverse,sign,shift]:self.assertAlmostEqual(r['joint_probability'],base['joint_probability'],12)

    def test_nesting_wide_event_inference_and_mass_sums(self):
        y=[-.3,.8,-.1,1.];settings=dict(min_length=1,max_segments=3)
        narrow=latent_envelope(y,[-.2]*4,[.2]*4,**settings)
        wide=latent_envelope(y,[-1e6]*4,[1e6]*4,**settings)
        self.assertLess(narrow['joint_probability'],wide['joint_probability'])
        self.assertAlmostEqual(wide['joint_probability'],1.,14)
        self.assertEqual(wide['inference'],infer(y,**settings))
        self.assertAlmostEqual(sum(x['event_and_count_probability'] for x in narrow['segment_count_given_event']),narrow['joint_probability'],14)
        self.assertAlmostEqual(sum(x['count_probability_given_event'] for x in narrow['segment_count_given_event']),1.,14)

    def test_strict_inputs_and_no_mutation(self):
        y=[0.,1.];lo=[-1.,0.];hi=[1.,2.];before=copy.deepcopy((y,lo,hi))
        rng_before=random.getstate()
        r=latent_envelope(y,lo,hi,min_length=1)
        self.assertEqual(random.getstate(),rng_before)
        self.assertEqual((y,lo,hi),before)
        r['lower'][0]=8;self.assertEqual(lo,before[1])
        for bad in [None,{},[],[0.],[True,0.],[0.,float('nan')],[0.,float('inf')],(0.,0.),[0.,1e6+1]]:
            with self.assertRaises(ValueError):latent_envelope(y,bad,hi,min_length=1)
            with self.assertRaises(ValueError):latent_envelope(y,lo,bad,min_length=1)
        with self.assertRaises(ValueError):latent_envelope(y,[2.,0.],hi,min_length=1)
        with self.assertRaises(ValueError):latent_envelope([],[],[])
        with self.assertRaises(ValueError):latent_envelope(y,lo,hi,noise_std=0)

    def test_maximum_bound_and_extreme_json(self):
        r=latent_envelope([0.]*120,[-1.]*120,[1.]*120,min_length=60,max_segments=12)
        self.assertEqual(r['positive_volume_partition_count'],2)
        self.assertEqual(len(r['segment_intersections']),1891)
        json.dumps(r,allow_nan=False)
        for y in [[-1e6,1e6],[1e6,1e6]]:
            r=latent_envelope(y,[-1e6]*2,[1e6]*2,min_length=1,noise_std=1e-4,prior_std=1e4)
            json.dumps(r,allow_nan=False)

    def test_cli_and_retained_audit(self):
        args=[sys.executable,'-m','segmentlab','envelope','examples/sequence.json','--lower','[-2,-2,-2,-2,-2,-2]','--upper','[2,2,2,2,2,2]']
        p=subprocess.run(args,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertIn('joint_probability',json.loads(p.stdout))
        for bad in ['bad','[]','[true]']:
            p=subprocess.run(args[:-1]+[bad],capture_output=True,text=True)
            self.assertEqual(p.returncode,2);self.assertNotIn('Traceback',p.stderr)
        p=subprocess.run([sys.executable,'examples/envelope_audit.py'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        report=json.loads(p.stdout)
        self.assertEqual(len(report['all_three_widths']),3)
        self.assertGreater(report['forced_shared_mean']['joint_probability'],report['forced_shared_mean']['pointwise_product_comparison'])
