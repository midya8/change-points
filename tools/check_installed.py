"""Build a wheel and replay all CLI families outside the source checkout."""

import json, os, subprocess, sys, tempfile, venv
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env.pop("PYTHONPATH", None)


def call(args, cwd):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True)


with tempfile.TemporaryDirectory() as d:
    t = Path(d)
    w = t / "w"
    w.mkdir()
    call(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(w), str(root)], t
    )
    venv.create(t / "env", with_pip=True)
    py = str(t / "env/bin/python")
    call([py, "-m", "pip", "install", "--no-deps", str(next(w.glob("*.whl")))], t)
    out = t / "outside"
    out.mkdir()
    code = "import importlib,pathlib; names=['segmentlab','segmentlab.cli','segmentlab.core','segmentlab.decision','segmentlab.demo','segmentlab.information','segmentlab.joint','segmentlab.predictive','segmentlab.sampling']; assert all('site-packages' in str(pathlib.Path(importlib.import_module(n).__file__).resolve()) for n in names);print(len(names))"
    assert call([py, "-c", code], out).strip() == "9"
    jobs = [
        ["demo"],
        ["infer", "examples/sequence.json"],
        ["sample", "examples/sequence.json", "--draws", "4", "--seed", "23"],
        ["joint", "examples/sequence.json", "--pairs", "[[1,3],[2,4]]"],
        ["information", "examples/sequence.json"],
        [
            "decide",
            "examples/sequence.json",
            "--false-cut-cost",
            "1",
            "--missed-cut-cost",
            "1",
        ],
        ["predictive", "examples/sequence.json", "--queries", "[[2,0.0],[3,1.0]]"],
    ]
    for job in jobs:
        expected = call([py, "-m", "segmentlab"] + job, root)
        args = [str(root / a) if a.startswith("examples/") else a for a in job]
        actual = call([py, "-m", "segmentlab"] + args, out)
        assert actual == expected, job
        console = call([str(t / "env/bin/segmentlab")] + args, out)
        assert console == expected, job
    print(
        json.dumps(
            dict(module_origins=9, full_reports=7, entrypoints=2, outside_checkout=True)
        )
    )
    # Keep the original nine-origin and seven-report/two-entrypoint checks.
    # The new report and the entire test suite also run from the installed wheel.
    import shutil
    shutil.copytree(root / 'examples', out / 'examples')
    shutil.copytree(root / 'assets', out / 'assets')
    job = ['covariance', 'examples/sequence.json', '--pairs', '[[0,0],[0,3]]']
    expected = call([py, '-m', 'segmentlab'] + job, root)
    absolute_job = [str(root / a) if a.startswith('examples/') else a for a in job]
    assert call([py, '-m', 'segmentlab'] + absolute_job, out) == expected
    assert call([str(t / 'env/bin/segmentlab')] + absolute_job, out) == expected
    suite_code = """
import importlib, importlib.util, pathlib, sys, unittest
names = ['segmentlab','segmentlab.cli','segmentlab.core','segmentlab.decision',
         'segmentlab.demo','segmentlab.information','segmentlab.joint',
         'segmentlab.predictive','segmentlab.sampling','segmentlab.covariance']
for name in names: importlib.import_module(name)
def check():
    assert all('site-packages' in importlib.util.find_spec(name).origin for name in names)
check()
sys.path.insert(0, str(pathlib.Path.cwd()))
suite = unittest.defaultTestLoader.discover(sys.argv[1])
assert unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
check()
print('Full installed test suite; 10 origins before and after')
"""
    print(call([py, '-I', '-c', suite_code, str(root / 'tests')], out))
    for name in ['sample', 'joint', 'information', 'decision', 'predictive', 'covariance']:
        source = call([sys.executable, str(root / 'examples' / (name + '_audit.py'))], root)
        installed = call([py, '-I', str(out / 'examples' / (name + '_audit.py'))], out)
        assert installed == source, name
    source_demo = call([sys.executable, '-m', 'segmentlab', 'demo'], root)
    installed_demo = call([py, '-I', '-m', 'segmentlab', 'demo'], out)
    assert installed_demo == source_demo
    (out / 'demo.json').write_text(installed_demo)
    sample_report = call([py, '-I', str(out / 'examples' / 'sample_audit.py')], out)
    (out / 'sample.json').write_text(sample_report)
    call([py, '-I', str(out / 'examples' / 'render_demo.py'), str(out / 'demo.json'), '--check'], out)
    call([py, '-I', str(out / 'examples' / 'render_samples.py'), str(out / 'sample.json'), '--check'], out)
    print('Complete demo and six feature audit reports, both figures reproduced from the installed wheel')
