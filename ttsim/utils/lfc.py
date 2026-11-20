# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

CACHE_AGE_SECONDS = 604800  # 1 week
LFC_URL_SUFFIX = '/simulators-ai-perf/'
# host[:port] of an LFC_SERVER_URLS entry: a DNS name or a bracketed IPv6 literal.
LFC_HOSTPORT_RE = re.compile(r'(?:[A-Za-z0-9.-]+|\[[0-9A-Fa-f:.]+\])(?::[0-9]+)?')

def _validate_server_path(server_path: str) -> None:
    """
    Validate server path to prevent path traversal attacks.

    Args:
        server_path: Path after 'lfc://' prefix

    Raises:
        ValueError: If path contains security risks (absolute paths, .., or backslashes)
    """
    if not server_path:
        raise ValueError("Server path cannot be empty")
    
    # Reject absolute paths
    if server_path.startswith('/'):
        raise ValueError(f"Absolute paths not allowed in LFC paths: {server_path}")
    
    # Reject Windows absolute paths (e.g., C:/, \\server\share)
    if len(server_path) >= 2 and server_path[1] == ':':
        raise ValueError(f"Windows drive letters not allowed in LFC paths: {server_path}")
    
    if server_path.startswith('\\\\'):
        raise ValueError(f"UNC paths not allowed in LFC paths: {server_path}")
    
    # Reject backslashes (potential path separator confusion)
    if '\\' in server_path:
        raise ValueError(f"Backslashes not allowed in LFC paths: {server_path}")
    
    # Reject path traversal attempts
    path_parts = server_path.split('/')
    for part in path_parts:
        if part == '..':
            raise ValueError(f"Path traversal (..) not allowed in LFC paths: {server_path}")
    
    # Additional safety: normalize and verify the resolved path stays within __ext
    try:
        # Construct the full local path and resolve it
        local_path = Path('__ext') / server_path
        resolved = local_path.resolve()
        expected_base = Path('__ext').resolve()
        
        # Check that resolved path is under __ext
        try:
            resolved.relative_to(expected_base)
        except ValueError:
            raise ValueError(f"Path would escape __ext directory: {server_path}")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Invalid path format: {server_path}") from e

def resolve_lfc_path(lfc_path: str) -> str:
    """
    Resolve an LFC path to a local file path, downloading if necessary.

    Args:
        lfc_path: Path starting with 'lfc://', e.g., 'lfc://hlm-lut/whb0_n150_lut.yaml'

    Returns:
        Local file path relative to workspace, e.g., '__ext/hlm-lut/whb0_n150_lut.yaml'

    Raises:
        ValueError: If path does not start with 'lfc://' or contains invalid/unsafe components
        RuntimeError: If download fails and no local file exists
    """
    if not lfc_path.startswith('lfc://'):
        raise ValueError(f"Path must start with 'lfc://': {lfc_path}")

    # Translate to local path
    server_path = lfc_path[6:]  # Remove 'lfc://'
    
    # Validate server path for security (prevent path traversal)
    _validate_server_path(server_path)
    
    local_path = f"__ext/{server_path}"

    local_file = Path(local_path)
    if local_file.exists():
        mtime = os.path.getmtime(local_path)
        if time.time() - mtime < CACHE_AGE_SECONDS:
            return local_path  # Fresh, use it

    # Need to download
    download_lfc_file(server_path, local_path)
    return local_path

def _get_lfc_base_urls() -> list[str]:
    """Return the ordered list of LFC base URLs to try, each ending in
    ``/simulators-ai-perf/``.

    The env var ``LFC_SERVER_URLS`` (comma-separated host URLs) is required —
    there are no built-in defaults.  Whitespace is stripped, trailing slashes
    are removed, and empty entries are dropped; if nothing usable remains,
    ``RuntimeError`` is raised.  Same normalization as
    ``tools/ci/lfc_downloader.sh``.

    Setup: dev users obtain the URL values from the internal team documentation
    (Slack pinned message / internal wiki) and export them in their shell rc or
    a local ``.env`` file.  CI runs set the value via a repository secret wired
    into ``.github/actions/lfcdownload/action.yml``.

    Behavioral difference from the shell script: the shell script probes each
    candidate host up front and commits to the first one that passes; this
    function returns all candidates and lets :func:`download_lfc_file` try them
    in order, falling back to the next URL only on a per-file HTTP or network
    error.  The end result is the same for a healthy primary URL; the difference
    is observable only when the primary host responds to the probe but then fails
    on the actual file request.
    """
    env = os.getenv('LFC_SERVER_URLS')
    if not env:
        raise RuntimeError(
            'LFC_SERVER_URLS is not set. This environment variable is required '
            'to use LFC downloads. See README.md ("LFC Server Configuration") '
            'for setup instructions.'
        )
    candidates = [u.strip().rstrip('/') for u in env.split(',') if u.strip()]
    if not candidates:
        raise RuntimeError('LFC_SERVER_URLS is set but contains no usable URLs')
    # Reject, never print: an entry with other characters would reach urllib,
    # whose error text escapes or normalizes the host beyond redaction.
    for index, candidate in enumerate(candidates, start=1):
        if not all('!' <= ch <= '~' for ch in candidate):
            raise RuntimeError(
                f'LFC_SERVER_URLS entry {index} of {len(candidates)} contains '
                'characters outside printable ASCII'
            )
        # A host limited to this grammar is printed exactly as configured,
        # apart from case, which _redact_lfc_urls covers.
        if not LFC_HOSTPORT_RE.fullmatch(_lfc_hostport(candidate)):
            raise RuntimeError(f'LFC_SERVER_URLS entry {index} of {len(candidates)} has an invalid host')
    return [c + LFC_URL_SUFFIX for c in candidates]


def _lfc_hostport(url: str) -> str:
    """Return ``host[:port]`` of ``url`` without user-info, using string operations only.

    urlsplit() is not used: it raises on a malformed entry, with the host in its message.
    """
    # The authority ends at the first '/', '?' or '#': an '@' after it is not user-info.
    return re.split(r'[/?#]', url.split('://', 1)[-1], maxsplit=1)[0].rpartition('@')[2]


def _redact_lfc_urls(text: str, base_urls: list[str]) -> str:
    """Replace every LFC base URL and host in ``text`` with a neutral label.

    Server URLs are configuration, not output.  Any text that can carry one
    (an exception message from urllib) passes through here before it is
    printed or raised.  ``tools/ci/lfc_downloader.sh`` applies the same rule.
    """
    for index, base_url in enumerate(base_urls, start=1):
        label = f'<LFC server {index} of {len(base_urls)}>'
        root = base_url.removesuffix(LFC_URL_SUFFIX)
        hostport = _lfc_hostport(root)
        if hostport.startswith('['):
            host = hostport[1:].split(']', 1)[0]  # IPv6 literal
        else:
            host = hostport.split(':', 1)[0]
        for value in (root, hostport, host, host.lower()):
            if value:
                text = text.replace(value, label)
    return text


def _attempt_download(url: str, local_file: Path, local_path: str) -> None:
    """Single download attempt for one URL.

    Atomic temp-file write + rename.  Raises ``urllib.error.HTTPError`` on any
    HTTP error status — including 401, since ``urllib.request.urlopen`` raises
    before returning a response object — or other exceptions on failure.  The
    caller (:func:`download_lfc_file`) catches ``HTTPError``, normalizes 401 to
    ``RuntimeError("Authentication required...")`` so auth errors are not
    retried, and decides retry / next-URL behavior for other errors.
    """
    tmp_path = None
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            with tempfile.NamedTemporaryFile(
                mode='wb',
                dir=local_file.parent,
                prefix=f'.{local_file.name}.',
                suffix='.tmp',
                delete=False,
            ) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(response.read())
            os.replace(tmp_path, local_path)
    except BaseException:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
        raise


def download_lfc_file(server_path: str, local_path: str):
    """
    Download a file from LFC server to local path.

    Tries each base URL in ``_get_lfc_base_urls()`` in order.  For each URL,
    makes up to 3 attempts before moving on.  First URL whose attempt succeeds
    wins.  If every URL exhausts its retries: in dev mode, an existing local
    file is used as a stale fallback (preferable to a hard failure when VPN is
    temporarily down); in CI (``GITHUB_ACTIONS=true``), the function raises
    ``RuntimeError`` immediately — a stale cached file in CI silently produces
    wrong results, so failing fast is the safer choice.

    Args:
        server_path: Path relative to simulators-ai-perf, e.g., 'hlm-lut/whb0_n150_lut.yaml'
        local_path: Local path, e.g., '__ext/hlm-lut/whb0_n150_lut.yaml'
    """
    base_urls = _get_lfc_base_urls()
    is_ci = os.getenv('GITHUB_ACTIONS') == 'true'

    local_file = Path(local_path)
    local_file.parent.mkdir(parents=True, exist_ok=True)

    # Use a lock file to prevent concurrent downloads
    lock_path = f"{local_path}.lock"
    lock_file = None
    try:
        if HAS_FCNTL:
            lock_file = open(lock_path, 'w')
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        # Check again if file exists and is fresh (another process may have downloaded it)
        if local_file.exists():
            mtime = os.path.getmtime(local_path)
            if time.time() - mtime < CACHE_AGE_SECONDS:
                return  # Fresh file exists, no need to download

        last_err: Optional[BaseException] = None
        for base_url in base_urls:
            url = base_url + server_path
            for attempt in range(3):
                try:
                    _attempt_download(url, local_file, local_path)
                    return  # Success — first URL/attempt that works wins
                except urllib.error.HTTPError as e:
                    if e.code == 401:
                        # All LFC hosts share the same auth domain, so a 401 on
                        # one host will repeat on every other host — bail early
                        # rather than burning retries on a credential problem.
                        # ``from None``: the HTTPError reason text comes from the server.
                        raise RuntimeError(f"Authentication required for LFC path {server_path}") from None
                    last_err = e
                    # Retry within this URL unless it's the last attempt
                except Exception as e:
                    last_err = e

        # All URLs exhausted.
        # In dev, fall back to an existing local file (may be stale, but far
        # preferable to a hard failure when VPN is temporarily down).
        # In CI, skip the fallback and fail fast — a stale cached file in CI
        # silently produces wrong results, which is worse than a clear failure.
        err_desc = f" ({_redact_lfc_urls(str(last_err), base_urls)})" if last_err else ""
        if local_file.exists() and not is_ci:
            print(
                f"Download failed across {len(base_urls)} LFC URL(s){err_desc}; "
                f"using existing local file: {local_path}",
                file=sys.stderr,
            )
            return
        if not is_ci:
            print("Direct access failed. Ensure Tailscale VPN is connected.", file=sys.stderr)
        http_detail = (
            f" (HTTP {last_err.code})"
            if isinstance(last_err, urllib.error.HTTPError)
            else err_desc
        )
        # ``from None``: the redacted urllib detail is already in the message.
        raise RuntimeError(
            f"Download failed across {len(base_urls)} LFC URL(s){http_detail} "
            f"and no local file exists: {local_path}"
        ) from None
    finally:
        # Release lock and clean up lock file
        if lock_file:
            try:
                if HAS_FCNTL:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                os.unlink(lock_path)
            except Exception:
                pass  # Best effort cleanup