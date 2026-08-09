# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Sphinx extension that generates the docs reference tables (doc/_ext/ttsim_registry.py)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

_EXT_PATH = Path(__file__).resolve().parents[1] / 'doc' / '_ext' / 'ttsim_registry.py'


def _load_ext():
    spec = importlib.util.spec_from_file_location('ttsim_registry_ext', _EXT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
def test_generated_tables_depend_on_the_extension_file():
    # A change to the table-building logic must regenerate the tables on an incremental build,
    # even when the registry and shim inputs are unchanged.
    ext = _load_ext()
    env = Mock()
    directive = SimpleNamespace(state=SimpleNamespace(document=SimpleNamespace(settings=SimpleNamespace(env=env))))

    ext._GeneratedDirective._note_deps(directive, [Path('/inputs/op.py')])

    noted = {Path(c.args[0]).resolve() for c in env.note_dependency.call_args_list}
    assert _EXT_PATH in noted
    assert Path('/inputs/op.py').resolve() in noted
