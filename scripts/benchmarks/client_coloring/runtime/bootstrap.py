# SPDX-License-Identifier: AGPL-3.0-or-later
"""Load the pinned CPython extension before importing the snapshot sampler."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def load_native():
    directory=os.environ.get('TRIAL_NATIVE_DIR')
    if not directory:
        # Local verification uses the already built native extension, copied
        # only into the scratch source snapshot, never packaged for Linux.
        from librewxr.tiles import _sampling_native
        return hashlib.sha256(Path(_sampling_native.__file__).read_bytes()).hexdigest()
    root=Path(directory);record=json.loads((root/'native.json').read_text())
    source=Path(__file__).with_name('source')/'librewxr/tiles/_sampling_native.c'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==record['source_sha256'],'Native source differs'
    assert Path(record['file']).name==record['file']
    path=root/record['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],'Native binary differs'
    import librewxr.tiles
    name='librewxr.tiles._sampling_native'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    setattr(librewxr.tiles,'_sampling_native',module)
    return record['sha256']


NATIVE_SHA256=load_native()
