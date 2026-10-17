# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Sphinx configuration for the Polaris (ttsim) documentation site.

Phase 1: MyST ingests the existing Markdown in ``doc/`` in place, plus a custom
build-time extension (``ttsim_registry``) that generates the SimOp / TTNN / ONNX
reference tables from the live code. API autodoc and publishing come in later phases.
"""

import os
import re
import sys
from pathlib import Path

_DOC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DOC_DIR.parent

# Make the custom extension importable, and the repo importable so the extension can
# read the ops descriptor registry at build time.
sys.path.insert(0, str(_DOC_DIR / '_ext'))
sys.path.insert(0, str(_REPO_ROOT))

project = 'Polaris'
copyright = '2025, Tenstorrent AI ULC'
author = 'Tenstorrent AI ULC'

extensions = [
    'myst_parser',
    'ttsim_registry',
]

# MyST: allow the ```{directive}``` fences the reference pages use, plus a few niceties.
myst_enable_extensions = [
    'colon_fence',
    'deflist',
]
myst_heading_anchors = 3

source_suffix = {'.md': 'markdown'}
root_doc = 'index'

# The existing markdown (INTRODUCTION.md, user_guide.md) links to source-code dirs
# (../ttsim/..., ../config/), the top-level README, and docs excluded from this Phase-1
# build. Those targets are real but sit outside the Sphinx tree, so MyST cannot resolve
# them. They are rewritten to GitHub URLs at source-read (_rewrite_repo_links below) so the
# published links WORK; no warning is emitted and none is suppressed. `-W` therefore still
# fails on a genuinely broken link, including a typo under those same directories.

# Peripheral docs excluded from the Phase-1 site (avoids orphan warnings under -W).
# These are wired into the toctree in later phases.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    'README.md',                              # GitHub folder-landing page, not a site page
    'README_dynamic_badges.md',
    'README_github_actions_architecture.md',
    'SPEC_ops_perf_three_csv_merge.md',
    'SPEC_ops_perf_trace_replay_merge.md',
    'tools/**',                               # tool/CI READMEs — folded in a later phase
]

html_theme = 'sphinx_rtd_theme'
html_title = 'Polaris Documentation'

# Belt-and-suspenders: some CI shells set cwd elsewhere; ensure repo import works.
os.environ.setdefault('PYTHONPATH', str(_REPO_ROOT))


# Blob/tree base for links that point at repository content rather than doc pages.
# Pin this to a tag when the site is published for a release; `main` tracks HEAD.
_GITHUB_BLOB = 'https://github.com/tenstorrent/polaris/blob/main'
_GITHUB_TREE = 'https://github.com/tenstorrent/polaris/tree/main'

# Markdown inline-link targets, e.g. the "../ttsim/ops/op.py" in "[op.py](../ttsim/ops/op.py)".
# Anchors/queries are excluded so a fragment link to a doc page is never touched.
_MD_LINK_RE = re.compile(r'\]\((?P<target>[^)\s#?]+)\)')


def _rewrite_repo_links(app, docname, source):
    """Point links at repository content to GitHub, so they RESOLVE instead of being silenced.

    ``INTRODUCTION.md`` and ``user_guide.md`` link to source dirs (``../ttsim/ops/op.py``),
    the top-level README, and docs excluded from this phase. Those targets are real, but they
    are outside the Sphinx tree, so MyST cannot resolve them and emits ``myst.xref_missing``.
    Earlier revisions of this file silenced that warning — first with an allowlist, then with
    an on-disk existence test. Both left the built HTML carrying
    ``<a href="#../ttsim/front/ttnn/">``: a same-page fragment link that navigates nowhere.
    The warning was the messenger; the dead link was the defect.

    Rewriting the target at ``source-read`` (before MyST parses it) makes the published link
    work, needs no warning filter at all, and drops this file's dependence on the wording of
    Sphinx log messages.

    The typo gate is preserved *by construction*: only a target that resolves to a real path
    is rewritten. ``../ttsim/tpyo.py`` resolves to nothing, is left untouched, and still
    raises ``myst.xref_missing`` — so ``sphinx-build -W`` fails, exactly as before.

    Note the asymmetry: a rewritten link is absolute and therefore tracks ``_GITHUB_BLOB``'s
    ref, while the on-disk markdown keeps its relative form and stays browsable on GitHub.
    """
    doc_dir = (Path(app.srcdir) / docname).parent

    def _repo_path(target):
        """Absolute path inside the repo for a link target, or None if it isn't one."""
        if target.startswith(('http://', 'https://', 'mailto:', 'ftp:', '/')):
            return None
        candidate = Path(os.path.normpath(str(doc_dir / target)))
        try:
            rel = candidate.relative_to(_REPO_ROOT)
        except ValueError:
            return None                      # escapes the repo — leave it alone
        if candidate.exists():
            return rel, candidate.is_dir()
        md = Path(f'{candidate}.md')          # pages are linked without their extension
        return (rel.with_suffix('.md'), False) if md.exists() else None

    def _sub(match):
        target = match.group('target')
        resolved = _repo_path(target)
        if resolved is None:
            return match.group(0)             # not repo content (or a typo) — untouched
        rel, is_dir = resolved
        # A doc page inside the Sphinx tree must keep its relative link so MyST resolves it
        # internally; only content OUTSIDE the tree needs to go to GitHub.
        if not is_dir and rel.suffix == '.md' and str(rel).startswith(f'{_DOC_DIR.name}/'):
            in_tree = Path(app.srcdir) / rel.relative_to(_DOC_DIR.name)
            if in_tree.exists() and not _is_excluded(rel.relative_to(_DOC_DIR.name)):
                return match.group(0)
        base = _GITHUB_TREE if is_dir else _GITHUB_BLOB
        return f']({base}/{rel})'

    source[0] = _MD_LINK_RE.sub(_sub, source[0])


def _is_excluded(rel_to_srcdir):
    """True if a doc-tree-relative path is kept out of this phase's build."""
    from fnmatch import fnmatch

    text = str(rel_to_srcdir)
    return any(fnmatch(text, pattern) for pattern in exclude_patterns)


def setup(app):
    app.connect('source-read', _rewrite_repo_links)
