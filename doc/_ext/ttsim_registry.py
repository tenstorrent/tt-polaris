# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Sphinx build-time directives that render Polaris reference tables from the live code.

Three directives are registered:

* ``simops-table``   -- every registered SimOp op-type and its metadata (from the ops
  descriptor registry).
* ``ttnn-simop-map`` -- the TTNN shim function -> SimOp op-type mapping (public factory-bound
  shims extracted statically via ``ast``, private helpers excluded; hand-written
  wrapper/composite shims are noted rather than fabricated, since their emission is
  argument-dependent).
* ``onnx-simop-map`` -- the ONNX op-type -> SimOp mapping (identity): the ``ai.onnx`` op-types
  with an implemented (callable) shape function, with registered-but-unimplemented stubs
  called out separately.

Design constraints (see the Sphinx docs plan):

* This module lives under ``doc/_ext`` and is imported ONLY by ``sphinx-build`` -- it is
  never imported by ``ttsim``/``polaris.py``, so it adds no runtime import cost.
* For the TTNN map it parses ``op.py`` source with ``ast`` -- it does NOT
  ``import ttsim.front.ttnn`` (avoids pulling the shim/ttnn import weight).
* For the SimOp/ONNX tables it imports only the ops descriptor registry, which any
  Polaris run already initializes.

Tables are built as docutils nodes directly (not by emitting RST/Markdown text), so they
render as real ``<table>`` elements regardless of the surrounding parser (these pages are
MyST/Markdown).
"""

from __future__ import annotations

import ast
from pathlib import Path

# docutils ships no type stubs and `types-docutils` is not worth a dev dependency for two
# imports — narrow, rule-specific ignores keep the rest of this file type-checked.
from docutils import nodes  # type: ignore[import-untyped]
from docutils.parsers.rst import Directive  # type: ignore[import-untyped]

# doc/_ext/ttsim_registry.py -> repo root is two parents up from this file's dir.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TTNN_OP_PY = _REPO_ROOT / 'ttsim' / 'front' / 'ttnn' / 'op.py'
_TTNN_INIT_PY = _REPO_ROOT / 'ttsim' / 'front' / 'ttnn' / '__init__.py'
_DESC_DIR = _REPO_ROOT / 'ttsim' / 'ops' / 'desc'  # source of the SimOp / ONNX registry


def _overridden_op_names() -> set:
    """Names that ``ttsim/front/ttnn/__init__.py`` re-imports AFTER ``from .op import *``.
    These shadow the star-imported op.py bindings, so for such a name the op.py factory
    mapping is NOT what the public shim actually emits (e.g. ``permute``/``reshape`` are
    re-exported from ``ttnn_shim``). They must be excluded from the factory table."""
    tree = ast.parse(_TTNN_INIT_PY.read_text())
    overridden: set = set()
    star_seen = False
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if any(a.name == '*' for a in node.names):
                if (node.module or '').split('.')[-1] == 'op':
                    star_seen = True
            elif star_seen:
                for a in node.names:
                    overridden.add(a.asname or a.name)
    return overridden

# Representative public wrapper/composite shims (hand-written `def`s in op.py, NOT factory
# assignments) whose emitted SimOp(s) depend on their arguments. Non-exhaustive BY DESIGN:
# op.py is the authoritative set. This list exists only so the factory table below is not
# mistaken for the whole shim API — it deliberately carries no per-op emission strings
# (those go stale; e.g. rms_norm now delegates to a single LayerNormalization).
_TTNN_WRAPPER_EXAMPLES = [
    'linear', 'conv2d', 'conv_transpose2d', 'max_pool2d', 'rms_norm', 'moe', 'silu',
    'concat', 'manual_seed', 'sampling',
]


# --- docutils node builders (parser-agnostic) ---------------------------------------

def _cell(content) -> nodes.entry:
    """One table cell. ``content`` is a str (plain), a ('code', str) tuple (monospace),
    or a list mixing str and ('code', str) fragments."""
    entry = nodes.entry()
    para = nodes.paragraph()
    frags = content if isinstance(content, list) else [content]
    for frag in frags:
        if isinstance(frag, tuple) and frag[0] == 'code':
            para += nodes.literal(text=frag[1])
        else:
            para += nodes.Text(frag if isinstance(frag, str) else str(frag))
    entry += para
    return entry


def _table(caption: str, headers: list[str], rows: list[list]) -> nodes.table:
    table = nodes.table()
    table['classes'].append('colwidths-auto')
    if caption:
        table += nodes.title(text=caption)
    tgroup = nodes.tgroup(cols=len(headers))
    table += tgroup
    for _ in headers:
        tgroup += nodes.colspec(colwidth=1)
    thead = nodes.thead()
    tgroup += thead
    hrow = nodes.row()
    for h in headers:
        hrow += _cell(h)
    thead += hrow
    tbody = nodes.tbody()
    tgroup += tbody
    for row in rows:
        r = nodes.row()
        for cell in row:
            r += _cell(cell)
        tbody += r
    return table


def _para(text: str) -> nodes.paragraph:
    p = nodes.paragraph()
    p += nodes.Text(text)
    return p


class _GeneratedDirective(Directive):
    """Base: subclasses implement ``_build_nodes`` returning a list of docutils nodes."""

    has_content = False

    def _build_nodes(self) -> list[nodes.Node]:  # pragma: no cover - overridden
        raise NotImplementedError

    def run(self) -> list[nodes.Node]:
        try:
            return self._build_nodes()
        except Exception as exc:
            # Fail the build — do NOT emit a silent warning node. A warning admonition does
            # not raise a Sphinx warning, so `sphinx-build -W` would still pass with the table
            # missing. Raising self.error() surfaces a docutils error so the gate enforces
            # that the table actually generated.
            raise self.error(f'{type(self).__name__}: could not generate table: {exc!r}')

    def _note_deps(self, paths) -> None:
        # Register the source files read at build time as page dependencies, so an
        # incremental build regenerates the table when they change (Sphinx cannot otherwise
        # infer that this .md depends on the registry / op.py). Keeps tables from going stale.
        env = self.state.document.settings.env
        for p in paths:
            env.note_dependency(str(p))


def _load_registry() -> dict:
    from ttsim.ops.desc import initialize_op_desc
    from ttsim.ops.desc.registry import get_opdesc_registry

    initialize_op_desc()
    return get_opdesc_registry()._registry


def _shape_fn_cell(val: object):
    if callable(val):
        # Not every callable carries __name__ — functools.partial, lru_cache wrappers and
        # callable class instances do not. Every shape fn registered today is a plain
        # function, but falling back to the type name keeps a future one from crashing the
        # docs build.
        return ('code', getattr(val, '__name__', None) or type(val).__name__)
    if isinstance(val, str):
        return [('code', val), ' (declared, unimplemented)']
    return str(val)


class SimOpsTableDirective(_GeneratedDirective):
    """Render all registered SimOp op-types with their metadata."""

    def _build_nodes(self) -> list[nodes.Node]:
        self._note_deps(sorted(_DESC_DIR.glob('*.py')))
        reg = _load_registry()
        headers = ['Op type', 'Group', 'Domain', 'Inputs', 'Outputs', 'Opset', 'Shape fn', 'Attrs']
        rows = []
        for name in sorted(reg):
            d = reg[name]
            rows.append([
                ('code', name),
                d.get('group', ''),
                d.get('domain', ''),
                f"{d['min_input']}–{d['max_input']}",
                f"{d['min_output']}–{d['max_output']}",
                str(d.get('version', '')),
                _shape_fn_cell(d.get('shape_inf_func')),
                'yes' if d.get('has_attr') else 'no',
            ])
        intro = _para(f'{len(rows)} registered op-types (generated at build time from the '
                      'ops descriptor registry).')
        return [intro, _table('Registered SimOp op-types', headers, rows)]


class TTNNSimOpMapDirective(_GeneratedDirective):
    """Render the TTNN shim -> SimOp op-type mapping."""

    def _build_nodes(self) -> list[nodes.Node]:
        self._note_deps([_TTNN_OP_PY, _TTNN_INIT_PY])
        overridden = _overridden_op_names()
        tree = ast.parse(_TTNN_OP_PY.read_text())
        raw = []  # (name, optype, kind) for every public factory assignment in op.py
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                fname = getattr(node.value.func, 'id', None)
                if fname in ('single_output_immediate_op', 'multiple_output_immediate_op') and node.value.args:
                    a0 = node.value.args[0]
                    if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                        kind = 'single' if fname.startswith('single') else 'multi'
                        for t in node.targets:
                            # Public only — skip private helper assignments (e.g. _concat_impl,
                            # _move, _conv2d_raw) that are not part of the exported shim API.
                            if isinstance(t, ast.Name) and not t.id.startswith('_'):
                                raw.append((t.id, a0.value, kind))
        # Exclude names that __init__.py re-exports (from ttnn_shim) after `from .op import *`
        # — their op.py factory binding is NOT the public emission, so listing it is wrong.
        factory = sorted((n, o, k) for (n, o, k) in raw if n not in overridden)
        shadowed = sorted({n for (n, o, k) in raw if n in overridden})
        rows = [[('code', shim), ('code', optype), kind] for shim, optype, kind in factory]
        intro = _para(f'{len(rows)} public factory-bound shims (extracted statically from '
                      'ttsim/front/ttnn/op.py; private helpers and __init__-overridden names excluded).')
        table = _table('TTNN shim to SimOp op-type', ['TTNN shim', 'SimOp op-type', 'Outputs'], rows)

        # The factory table omits the shim's hand-written wrapper/composite functions, whose
        # emitted SimOp(s) are argument-dependent. Name a representative few and point at the
        # source as authoritative — do NOT fabricate per-op emission strings (they go stale).
        note = nodes.paragraph()
        note += nodes.Text('The factory table above does not capture the shim’s hand-written '
                           'wrapper/composite functions (e.g. ')
        for i, w in enumerate(_TTNN_WRAPPER_EXAMPLES):
            if i:
                note += nodes.Text(', ')
            note += nodes.literal(text=w)
        note += nodes.Text('), whose emitted SimOp(s) depend on their arguments. See ')
        note += nodes.literal(text='ttsim/front/ttnn/op.py')
        note += nodes.Text(' for the complete, authoritative set.')
        result = [intro, table, note]
        if shadowed:
            snote = nodes.paragraph()
            snote += nodes.Text('Note: ')
            snote += nodes.literal(text='ttsim/front/ttnn/__init__.py')
            snote += nodes.Text(' re-exports ')
            for i, n in enumerate(shadowed):
                if i:
                    snote += nodes.Text(', ')
                snote += nodes.literal(text=n)
            snote += nodes.Text(' from ')
            snote += nodes.literal(text='ttnn_shim')
            snote += nodes.Text(' after ')
            snote += nodes.literal(text='from .op import *')
            snote += nodes.Text(', so the public shim’s emission differs from the op.py factory '
                                'binding of the same name — these are excluded from the table above.')
            result.append(snote)
        return result


class ONNXSimOpMapDirective(_GeneratedDirective):
    """Render the ONNX op-type -> SimOp mapping (identity) and supported ai.onnx op set."""

    def _build_nodes(self) -> list[nodes.Node]:
        self._note_deps(sorted(_DESC_DIR.glob('*.py')))
        reg = _load_registry()
        onnx_ops = sorted(n for n, d in reg.items() if d.get('domain') == 'ai.onnx')
        # "Registered" != "executable": some ai.onnx entries carry a *string* shape-fn stub,
        # which SimOp.get_perf_counts would call and raise on. Only callable shape-fns are
        # genuinely supported; the stubs are listed separately as not-yet-implemented.
        callable_ops = [n for n in onnx_ops if callable(reg[n].get('shape_inf_func'))]
        stub_ops = [n for n in onnx_ops if isinstance(reg[n].get('shape_inf_func'), str)]
        headers = ['ONNX op-type', 'SimOp op-type', 'Inputs', 'Outputs', 'Shape fn']
        rows = []
        for name in callable_ops:
            d = reg[name]
            rows.append([
                ('code', name),
                [('code', name), ' (identity)'],
                f"{d['min_input']}–{d['max_input']}",
                f"{d['min_output']}–{d['max_output']}",
                _shape_fn_cell(d.get('shape_inf_func')),
            ])
        intro = _para('ONNX nodes map to SimOps by identity (the node’s op_type is the SimOp op-type), '
                      'sharing the same shape-inference functions. The table lists the '
                      f'{len(rows)} ai.onnx op-types with an implemented (callable) shape function.')
        result = [intro, _table('Supported ONNX op-types', headers, rows)]
        if stub_ops:
            note = nodes.paragraph()
            note += nodes.Text(f'A further {len(stub_ops)} ai.onnx op-types are registered but NOT yet '
                               'implemented (string shape-fn stub — would raise at runtime if used): ')
            for i, n in enumerate(stub_ops):
                if i:
                    note += nodes.Text(', ')
                note += nodes.literal(text=n)
            note += nodes.Text('.')
            result.append(note)
        return result


def setup(app):
    app.add_directive('simops-table', SimOpsTableDirective)
    app.add_directive('ttnn-simop-map', TTNNSimOpMapDirective)
    app.add_directive('onnx-simop-map', ONNXSimOpMapDirective)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
