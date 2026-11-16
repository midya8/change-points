"""Retain every supplied width and the forced-shared-mean comparison."""
import json
from pathlib import Path
import sys
# Prefer an installed package; use the source root only for direct source runs.
try:
    import segmentlab
except ModuleNotFoundError:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from segmentlab import latent_envelope

settings = dict(min_length=1,max_segments=3)
print(json.dumps({
    'all_three_widths': [dict(width=w,report=latent_envelope([-.3,.8,-.1,1.],[-w]*4,[w]*4,**settings)) for w in (.5,1.,2.)],
    'forced_shared_mean': latent_envelope([0.]*4,[-.5]*4,[.5]*4,min_length=1,max_segments=1),
    'conflicting_bounds': latent_envelope([0.,0.],[-1.,.1],[-.1,1.],min_length=1,max_segments=2),
    'forced_one_segment_conflict': latent_envelope([0.,0.],[-1.,.1],[-.1,1.],min_length=1,max_segments=1),
},indent=2,allow_nan=False))
