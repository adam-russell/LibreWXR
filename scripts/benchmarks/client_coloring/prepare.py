# SPDX-License-Identifier: AGPL-3.0-or-later
"""Create a self-contained benchmark bundle from this working tree, without Git writes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args()
    repo=HERE.parents[2];out=args.output.resolve()
    if out.exists():raise SystemExit('Choose a new output directory')
    out.mkdir(parents=True);runtime=out/'runtime';runtime.mkdir()
    for name in ('guards.py','run-benchmark.py'):shutil.copy2(HERE/name,out/name)
    for file in (HERE/'runtime').iterdir():
        if file.is_file():shutil.copy2(file,runtime/file.name)
    shutil.copytree(repo/'src/librewxr',runtime/'source/librewxr',ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.so','*.dylib'))
    shutil.copy2(repo/'LICENSE',runtime/'source/LICENSE')
    manifest=dict(dependency_image='sha256:a92564fd683bdaa10459996edad0ea8bf6acb7d82eefd3a1c74f64ab7ddbd4a4',
                  snapshot_sha256='129c077c30b3fcb6e7280ef0ce67e70b3f5ba8f778c1ac47226562fbfd677d5a',
                  native_sha256='15c17fbfadf839231ad92ff634c07dc72ebad4f20aef1e7d1b18d3bf63b248ac',
                  baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                  rounds=5,deadline_seconds=1200,
                  cases=[dict(name=name,lat=lat,lon=lon,z=z) for name,lat,lon,z in [('Halifax',44.5,-64.4,7),('Genoa',44.4,8.9,9),('Osaka',34.69,135.5,7),('Taipei',25.03,121.56,9)]],
                  scope='Current colored PNG scheme 2 vs palette-independent zlib levels 1 and 6. 512px, smooth+snow, four locations, five saved real frames cycled to six steps. Nine serial fresh workers in three balanced cycles. Separate presentation-only, coordinate-warm full render, coordinate-cold full render, adjacent-tile coordinate-cold render, and reference-client costs; each phase five repetitions. No HTTP throughput, completed-tile cache-hit, GPU, battery or native-app performance claims. Exact sampled planes and all-palette reference RGBA checked before timing; offline browser client verifies its own RGBA hash before canvas. Synthetic edge/mixed-snow supplements disclosed separately. Palette-switch demonstration is a single geometry and uncompressed palette JSON overhead is reported separately. Existing pinned dependency image and native sampler, source overlaid from this working tree, immutable weather mounts. Services and settings unchanged. 2 CPUs/1.5GiB/no swap/network; 1GiB host reserve; 20-minute global deadline.')
    manifest['runtime_hashes']={str(f.relative_to(runtime)):sha(f) for f in sorted(runtime.rglob('*')) if f.is_file()}
    (runtime/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    checks={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}
    (out/'checksums.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(out)
if __name__=='__main__':main()
