# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the optional CPython sampler without NumPy headers or runtime tools."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig
import tempfile

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return
        required = os.environ.get("LIBREWXR_REQUIRE_NATIVE") == "1"
        disabled = os.environ.get("LIBREWXR_DISABLE_NATIVE") == "1"
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        if version == "editable" and suffix:
            # A failed rebuild must not silently leave an older native kernel
            # active in the editable checkout. Unlinking preserves live mappings.
            (Path(self.root) / ("src/librewxr/tiles/_sampling_native" + suffix)).unlink(missing_ok=True)
        if disabled and required:
            raise RuntimeError("Native sampling cannot be both required and disabled")
        if disabled:
            return
        self._native_build = tempfile.TemporaryDirectory(prefix="sampling-build-")
        try:
            if sys.platform not in ("linux", "darwin"):
                raise OSError("Native sampling currently builds on Linux and macOS")
            compiler = shlex.split(os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc")
            if not suffix:
                raise OSError("Python extension suffix is unavailable")
            name = "_sampling_native" + suffix
            binary = Path(self._native_build.name) / name
            source = Path(self.root) / "src/librewxr/tiles/_sampling_native.c"
            flags = ["-bundle", "-undefined", "dynamic_lookup"] if sys.platform == "darwin" else ["-shared"]
            command = compiler + flags + [
                "-O3", "-std=c11", "-fPIC", "-ffp-contract=off", "-fno-fast-math",
            ]
            for directory in {sysconfig.get_path("include"), sysconfig.get_path("platinclude")}:
                if directory:
                    command += ["-I", directory]
            subprocess.run(command + [str(source), "-o", str(binary)], check=True, timeout=120)
            if version == "editable":
                # Editable imports resolve to src/, so the extension belongs
                # beside the Python module rather than only in site-packages.
                import shutil
                target = source.with_name(name)
                shutil.copyfile(binary, target)
            else:
                build_data["force_include"][str(binary)] = "librewxr/tiles/" + name
            build_data["pure_python"] = False
            build_data["infer_tag"] = True
        except (OSError, subprocess.SubprocessError) as error:
            self._native_build.cleanup()
            if required:
                raise RuntimeError("Required native sampler failed to build") from error
            self.app.display_warning(f"Native sampler unavailable; using NumPy: {error}")

    def finalize(self, version, build_data, artifact_path):
        temporary = getattr(self, "_native_build", None)
        if temporary is not None:
            temporary.cleanup()
