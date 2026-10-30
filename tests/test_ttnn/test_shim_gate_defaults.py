# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Every opt-in shim gate defaults OFF, and is queryable through its accessor.

These gates change LUT keys or op counts for *every* workload built in the
process, so a gate that silently defaulted ON would corrupt whichever workload
was built next.  Also pins the set_/`_enabled` pair for each, which is the
documented API.
"""
import pytest

import ttsim.front.ttnn as ttnn
from ttsim.utils.shim_gates import (
    bcast_hw_shape_propagation_enabled,
    conv_input_reshard_enabled,
    conv_math_fidelity_in_key_enabled,
    halo_output_dtype_promotion_enabled,
    memory_config_propagation_enabled,
    move_on_reallocate_halo_output_enabled,
    set_bcast_hw_shape_propagation,
    set_conv_input_reshard,
    set_conv_math_fidelity_in_key,
    set_halo_output_dtype_promotion,
    set_memory_config_propagation,
    set_move_on_reallocate_halo_output,
    set_tilize_before_matmul,
    set_to_memory_config_reshard,
    tilize_before_matmul_enabled,
    to_memory_config_reshard_enabled,
)

# (setter, getter, the Device attribute the getter looks for)
GATES = [
    (set_memory_config_propagation,   memory_config_propagation_enabled,   'propagate_memory_config'),
    (set_tilize_before_matmul,        tilize_before_matmul_enabled,        'tilize_before_matmul'),
    (set_bcast_hw_shape_propagation,  bcast_hw_shape_propagation_enabled,  'propagate_bcast_hw_shape'),
    (set_conv_math_fidelity_in_key,   conv_math_fidelity_in_key_enabled,   'conv_math_fidelity_in_key'),
    (set_halo_output_dtype_promotion, halo_output_dtype_promotion_enabled, 'halo_output_dtype_promotion'),
    (set_conv_input_reshard,          conv_input_reshard_enabled,          'conv_input_reshard'),
    (set_to_memory_config_reshard,    to_memory_config_reshard_enabled,    'to_memory_config_reshard'),
    (set_move_on_reallocate_halo_output, move_on_reallocate_halo_output_enabled,
     'move_on_reallocate_halo_output'),
]


@pytest.fixture(autouse=True)
def _restore_process_wide_defaults():
    """Save and restore the process-wide value of every gate around each test.

    Two tests below move process-wide state (``set_<gate>(True)`` with no
    device).  They used to restore it with a hardcoded ``setter(False)``, which
    is only correct when the gate started off -- so on a run where any gate
    started ON, a passing test still DESTROYED that state for everything
    collected after it.  This captures the observed value instead, which is the
    pattern ``test_memory_config_propagation.py`` already uses for its one gate;
    generalised here because this file is parametrised over all of them.

    What this fixes and what it does NOT.  Each module default is seeded from
    its ``TTSIM_*`` variable at import (``ttsim/utils/shim_gates.py``), and
    ``doc/shim_gates.md`` documents setting one to try a gate across a whole
    run.  Under e.g. ``TTSIM_PROPAGATE_MEMORY_CONFIG=1 pytest`` this fixture
    stops the file from CLEARING that variable's effect for later tests, and the
    two tests that move process-wide state now seed themselves instead of
    requiring the gate to arrive off.

    ``test_gate_defaults_off`` and ``test_setter_returns_previous`` still fail
    under such a run, and should: they assert the SHIPPED default is off, which
    an env override deliberately contradicts.  Seeding them would turn them into
    tests of the setter and delete the only coverage of the shipped value.  So a
    failure there means "an env var is set", not "the gates are broken" -- worth
    knowing before chasing it.

    Restoring every gate rather than only the parametrised one is deliberate: it
    costs 8 reads and makes the guarantee "this file leaves no process-wide
    state behind", which does not need revisiting when a gate is added.
    """
    saved = [(setter, getter()) for setter, getter, _ in GATES]
    yield
    for setter, previous in saved:
        setter(previous)


def _dev():
    return ttnn.open_device(l1_small_size=24576, device_id=0)


@pytest.mark.unit
@pytest.mark.parametrize("setter,getter,attr", GATES, ids=lambda f: getattr(f, "__name__", ""))
def test_gate_defaults_off(setter, getter, attr):
    assert getter(_dev()) is False


@pytest.mark.unit
@pytest.mark.parametrize("setter,getter,attr", GATES, ids=lambda f: getattr(f, "__name__", ""))
def test_gate_is_device_scoped(setter, getter, attr):
    a, b = _dev(), _dev()
    setter(True, a)
    try:
        assert getter(a) is True, "setter did not take on its own device"
        assert getter(b) is False, "gate leaked to another device"
    finally:
        setter(False, a)


@pytest.mark.unit
@pytest.mark.parametrize("setter,getter,attr", GATES, ids=lambda f: getattr(f, "__name__", ""))
def test_setter_returns_previous(setter, getter, attr):
    d = _dev()
    assert setter(True, d) is False
    assert setter(False, d) is True
    assert getter(d) is False


# ---------------------------------------------------------------------------
# the two resolution layers: process-wide default, then per-device override
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("setter,getter,attr", GATES, ids=lambda f: getattr(f, "__name__", ""))
def test_fresh_device_carries_no_override(setter, getter, attr):
    """Device.__init__ must not materialise the gate onto the instance.

    Every gate resolves as ``getattr(device, attr, <module default>)``. If the
    constructor always set the attribute the fallback arm would be dead code and
    the process-wide layer unreachable, so the absence is the contract.
    """
    assert not hasattr(_dev(), attr)


@pytest.mark.unit
@pytest.mark.parametrize("setter,getter,attr", GATES, ids=lambda f: getattr(f, "__name__", ""))
def test_process_wide_default_reaches_a_later_device(setter, getter, attr):
    """``set_<gate>(True)`` with no device must apply to devices opened after it.

    Regression: Device.__init__ used to snapshot the ``TTSIM_*`` env var onto
    every instance, so a device created after a device-less setter call came back
    pinned to the env's False and silently ignored the call.
    """
    # Seed deterministically rather than requiring the gate to already be off.
    # ``assert setter(True) is False`` alone is a PRECONDITION on inherited
    # state: under ``TTSIM_<gate>=1`` it fails here, and the message points at
    # the gate machinery rather than at the environment.  The autouse fixture
    # puts the real previous value back.
    setter(False)
    assert setter(True) is False, 'seeding the process-wide default did not take'
    try:
        assert getter() is True
        assert getter(_dev()) is True, "device opened after the call ignored it"
    finally:
        setter(False)
    assert getter(_dev()) is False


@pytest.mark.unit
@pytest.mark.parametrize("setter,getter,attr", GATES, ids=lambda f: getattr(f, "__name__", ""))
def test_device_override_beats_process_wide_default(setter, getter, attr):
    """Both directions: a per-device value wins over the process-wide one."""
    on, off = _dev(), _dev()
    setter(False)  # seed; see the note in the test above
    assert setter(True) is False, 'seeding the process-wide default did not take'
    try:
        setter(False, off)
        assert getter(off) is False, "device override did not beat the global"
        assert getter(on) is True, "unset device did not follow the global"
    finally:
        setter(False)
    setter(True, on)
    try:
        assert getter(on) is True
        assert getter(_dev()) is False
    finally:
        setter(False, on)


# ---------------------------------------------------------------------------
# layering: ttsim.ops must not depend on ttsim.front to read a gate
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_gate_state_module_is_a_leaf():
    """``ttsim.utils.shim_gates`` must not import ttsim.ops or ttsim.front.

    It is the shared floor both packages read the gates from.  The moment it
    imports either one, it stops being able to serve the other.
    """
    import ast
    import pathlib
    src = pathlib.Path('ttsim/utils/shim_gates.py').read_text()
    deps: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            deps.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            deps.add(node.module)
    offenders = {d for d in deps if d.startswith(('ttsim.ops', 'ttsim.front', 'ttsim.graph', 'ttsim.back'))}
    assert not offenders, f'shim_gates must stay a leaf; it imports {sorted(offenders)}'


@pytest.mark.unit
def test_bcast_gate_read_does_not_drag_in_the_ttnn_front_end():
    """``bidir_bcast`` reads its gate without ttsim.ops importing ttsim.front.

    Regression: the read used to be a deferred
    ``from ttsim.utils.shim_gates import bcast_hw_shape_propagation_enabled``
    inside the function.  Deferring moved the dependency rather than removing
    it — every elementwise binary in every pure-ONNX workload pulled in the
    whole ttnn front end, closing a real cycle back into
    ``ttsim.ops.desc.helpers`` itself.
    """
    import ast
    import pathlib
    src = pathlib.Path('ttsim/ops/desc/helpers.py').read_text()
    bad = []
    for node in ast.walk(ast.parse(src)):
        mod = None
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
        elif isinstance(node, ast.Import):
            mod = next((a.name for a in node.names if a.name.startswith('ttsim.front')), None)
        if mod and mod.startswith('ttsim.front'):
            bad.append((getattr(node, 'lineno', '?'), mod))
    assert not bad, f'helpers.py imports ttsim.front at {bad} — deferred counts too'


@pytest.mark.unit
def test_op_py_imports_only_the_accessors_it_calls():
    """op.py must not re-export setters it never uses.

    The gates moved to ttsim/utils/shim_gates.py to break a back-edge, and the
    first cut re-exported all sixteen names from op.py so callers would not have
    to change.  Ten of those were pure pass-through -- op.py READS gates, it
    never sets them -- which left an import block describing its callers' needs
    rather than its own, kept alive by a ``# noqa: F401``.  Setters now come
    from shim_gates directly.
    """
    import ast
    import pathlib
    src = pathlib.Path('ttsim/front/ttnn/op.py').read_text()
    imported = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module == 'ttsim.utils.shim_gates':
            imported += [a.name for a in node.names]
    setters = [n for n in imported if n.startswith('set_')]
    assert not setters, f'op.py re-exports setters it does not call: {setters}'
    # Everything it does import, it must actually reference in code.
    loaded = {n.id for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    unused = [n for n in imported if n not in loaded]
    assert not unused, f'op.py imports gate names it never calls: {unused}'


@pytest.mark.unit
def test_setters_and_getters_come_from_one_module():
    """A matched pair must not arrive from two different modules."""
    import ttsim.utils.shim_gates as gates
    for setter, getter, _ in GATES:
        assert getattr(gates, setter.__name__) is setter
        assert getattr(gates, getter.__name__) is getter
