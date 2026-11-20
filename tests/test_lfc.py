# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import time
import traceback
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ttsim.utils.lfc import (
    CACHE_AGE_SECONDS,
    _get_lfc_base_urls,
    _redact_lfc_urls,
    resolve_lfc_path,
)

# Sentinel LFC hosts for the redaction tests. No output may contain them.
SENTINEL_HOST_A = 'lfc-sentinel-a.invalid'
SENTINEL_HOST_B = 'lfc-sentinel-b.invalid'
SENTINEL_URLS = f' http://{SENTINEL_HOST_A}:8080/ ,http://{SENTINEL_HOST_B}'


@pytest.fixture(autouse=True)
def _set_lfc_env(monkeypatch):
    """Default LFC_SERVER_URLS for all tests so download paths execute against a
    fake URL instead of failing the "must be set" check.  Tests that exercise
    unset/empty behavior explicitly delenv inside the test body — that takes
    precedence over this fixture for the duration of those tests."""
    monkeypatch.setenv('LFC_SERVER_URLS', 'http://test.example')


@pytest.mark.unit
def test_resolve_lfc_path_invalid_prefix():
    with pytest.raises(ValueError, match="Path must start with 'lfc://'"):
        resolve_lfc_path("invalid/path.yaml")

@pytest.mark.unit
def test_resolve_lfc_path_cached():
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            local_path = "__ext/test/file.yaml"
            Path(local_path).parent.mkdir(parents=True)
            Path(local_path).touch()
            # Set mtime to now
            os.utime(local_path, (time.time(), time.time()))

            result = resolve_lfc_path("lfc://test/file.yaml")
            assert result == local_path
    finally:
        os.chdir(original_cwd)

@pytest.mark.unit
def test_resolve_lfc_path_stale_download_success():
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            local_path = "__ext/test/file.yaml"
            Path(local_path).parent.mkdir(parents=True)
            Path(local_path).touch()
            # Set mtime to old
            old_time = time.time() - CACHE_AGE_SECONDS - 1
            os.utime(local_path, (old_time, old_time))
            with patch('urllib.request.urlopen') as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = b"content"
                mock_response.status = 200
                mock_urlopen.return_value.__enter__.return_value = mock_response
                result = resolve_lfc_path("lfc://test/file.yaml")
                assert result == local_path
                mock_urlopen.assert_called_once()
                # Check content was written
                with open(local_path, 'rb') as f:
                    assert f.read() == b"content"
    finally:
        os.chdir(original_cwd)

@pytest.mark.unit
def test_resolve_lfc_path_download_fail_fallback(monkeypatch):
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            local_path = "__ext/test/file.yaml"
            Path(local_path).parent.mkdir(parents=True)
            Path(local_path).write_text("old content")
            # Set mtime to be stale (older than CACHE_AGE_SECONDS)
            old_time = time.time() - CACHE_AGE_SECONDS - 1
            os.utime(local_path, (old_time, old_time))
            with patch('urllib.request.urlopen', side_effect=Exception("Network error")) as mock_urlopen:
                result = resolve_lfc_path("lfc://test/file.yaml")
                assert result == local_path
                # Verify download was attempted
                mock_urlopen.assert_called()
                # Content should remain old (preserved after failed download)
                assert Path(local_path).read_text() == "old content"
    finally:
        os.chdir(original_cwd)

@pytest.mark.unit
def test_resolve_lfc_path_download_fail_no_local():
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            with patch('urllib.request.urlopen', side_effect=Exception("Network error")):
                with pytest.raises(RuntimeError, match=r"Download failed.*no local file exists"):
                    resolve_lfc_path("lfc://test/file.yaml")
    finally:
        os.chdir(original_cwd)

# Security validation tests

@pytest.mark.unit
def test_resolve_lfc_path_rejects_empty_path():
    with pytest.raises(ValueError, match="Server path cannot be empty"):
        resolve_lfc_path("lfc://")


@pytest.mark.unit
def test_resolve_lfc_path_rejects_absolute_path():
    with pytest.raises(ValueError, match="Absolute paths not allowed"):
        resolve_lfc_path("lfc:///etc/passwd")


@pytest.mark.unit
def test_resolve_lfc_path_rejects_path_traversal():
    with pytest.raises(ValueError, match="Path traversal.*not allowed"):
        resolve_lfc_path("lfc://../../../etc/passwd")


@pytest.mark.unit
def test_resolve_lfc_path_rejects_path_traversal_midpath():
    with pytest.raises(ValueError, match="Path traversal.*not allowed"):
        resolve_lfc_path("lfc://test/../../../etc/passwd")


@pytest.mark.unit
def test_resolve_lfc_path_rejects_backslashes():
    with pytest.raises(ValueError, match="Backslashes not allowed"):
        resolve_lfc_path("lfc://test\\file.yaml")


@pytest.mark.unit
def test_resolve_lfc_path_rejects_windows_drive_letter():
    with pytest.raises(ValueError, match="Windows drive letters not allowed"):
        resolve_lfc_path("lfc://C:/Windows/System32/file.txt")


@pytest.mark.unit
def test_resolve_lfc_path_rejects_unc_path():
    with pytest.raises(ValueError, match="UNC paths not allowed"):
        resolve_lfc_path("lfc://\\\\server\\share\\file.txt")


@pytest.mark.unit
def test_resolve_lfc_path_allows_valid_nested_path():
    """Test that valid nested paths are allowed"""
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            local_path = "__ext/deep/nested/path/file.yaml"
            Path(local_path).parent.mkdir(parents=True)
            Path(local_path).touch()
            os.utime(local_path, (time.time(), time.time()))
            
            result = resolve_lfc_path("lfc://deep/nested/path/file.yaml")
            assert result == local_path
    finally:
        os.chdir(original_cwd)


@pytest.mark.unit
def test_resolve_lfc_path_http_404_with_local_fallback(monkeypatch):
    """Test that 404 errors fall back to stale local cache in dev mode"""
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            local_path = "__ext/test/file.yaml"
            Path(local_path).parent.mkdir(parents=True)
            Path(local_path).write_text("cached content")
            # Set mtime to be stale
            old_time = time.time() - CACHE_AGE_SECONDS - 1
            os.utime(local_path, (old_time, old_time))
            
            http_error = urllib.error.HTTPError(
                url="http://test.com/file.yaml",
                code=404,
                msg="Not Found",
                hdrs={},
                fp=None
            )
            with patch('urllib.request.urlopen', side_effect=http_error) as mock_urlopen:
                result = resolve_lfc_path("lfc://test/file.yaml")
                assert result == local_path
                # Verify download was attempted
                mock_urlopen.assert_called()
                # Content should be preserved
                assert Path(local_path).read_text() == "cached content"
    finally:
        os.chdir(original_cwd)


@pytest.mark.unit
def test_resolve_lfc_path_http_500_with_local_fallback(monkeypatch):
    """Test that 500 errors fall back to stale local cache in dev mode"""
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            local_path = "__ext/test/file.yaml"
            Path(local_path).parent.mkdir(parents=True)
            Path(local_path).write_text("cached content")
            # Set mtime to be stale
            old_time = time.time() - CACHE_AGE_SECONDS - 1
            os.utime(local_path, (old_time, old_time))
            
            http_error = urllib.error.HTTPError(
                url="http://test.com/file.yaml",
                code=500,
                msg="Internal Server Error",
                hdrs={},
                fp=None
            )
            with patch('urllib.request.urlopen', side_effect=http_error) as mock_urlopen:
                result = resolve_lfc_path("lfc://test/file.yaml")
                assert result == local_path
                # Verify download was attempted
                mock_urlopen.assert_called()
                # Content should be preserved
                assert Path(local_path).read_text() == "cached content"
    finally:
        os.chdir(original_cwd)


@pytest.mark.unit
def test_resolve_lfc_path_http_404_without_local_fallback():
    """Test that 404 errors raise when no local cache exists"""
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            
            http_error = urllib.error.HTTPError(
                url="http://test.com/file.yaml",
                code=404,
                msg="Not Found",
                hdrs={},
                fp=None
            )
            with patch('urllib.request.urlopen', side_effect=http_error):
                with pytest.raises(RuntimeError, match=r"Download failed.*no local file exists"):
                    resolve_lfc_path("lfc://test/file.yaml")
    finally:
        os.chdir(original_cwd)


@pytest.mark.unit
def test_resolve_lfc_path_http_401_no_fallback():
    """Test that 401 errors are raised immediately without fallback"""
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            local_path = "__ext/test/file.yaml"
            Path(local_path).parent.mkdir(parents=True)
            Path(local_path).write_text("cached content")
            # Set mtime to be stale
            old_time = time.time() - CACHE_AGE_SECONDS - 1
            os.utime(local_path, (old_time, old_time))
            
            http_error = urllib.error.HTTPError(
                url="http://test.com/file.yaml",
                code=401,
                msg="Unauthorized",
                hdrs={},
                fp=None
            )
            with patch('urllib.request.urlopen', side_effect=http_error):
                # 401 should raise immediately, not fall back to cache
                with pytest.raises(RuntimeError, match="Authentication required"):
                    resolve_lfc_path("lfc://test/file.yaml")
    finally:
        os.chdir(original_cwd)


# _get_lfc_base_urls() — env var resolution


@pytest.mark.unit
def test_get_lfc_base_urls_unset_in_dev_raises(monkeypatch):
    """Dev mode (no GITHUB_ACTIONS, no LFC_SERVER_URLS) → RuntimeError."""
    monkeypatch.delenv('LFC_SERVER_URLS', raising=False)
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    with pytest.raises(RuntimeError, match='LFC_SERVER_URLS is not set'):
        _get_lfc_base_urls()


@pytest.mark.unit
def test_get_lfc_base_urls_unset_in_ci_raises(monkeypatch):
    """CI mode (GITHUB_ACTIONS=true, no LFC_SERVER_URLS) → RuntimeError."""
    monkeypatch.delenv('LFC_SERVER_URLS', raising=False)
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    with pytest.raises(RuntimeError, match='LFC_SERVER_URLS is not set'):
        _get_lfc_base_urls()


@pytest.mark.unit
def test_get_lfc_base_urls_dev(monkeypatch):
    """LFC_SERVER_URLS parsed in dev mode; trims whitespace; strips trailing slashes."""
    monkeypatch.setenv(
        'LFC_SERVER_URLS',
        ' http://a.example/ , http://b.example, , http://c.example/ ',
    )
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    urls = _get_lfc_base_urls()
    assert urls == [
        'http://a.example/simulators-ai-perf/',
        'http://b.example/simulators-ai-perf/',
        'http://c.example/simulators-ai-perf/',
    ]


@pytest.mark.unit
def test_get_lfc_base_urls_ci(monkeypatch):
    """LFC_SERVER_URLS parsed identically in CI mode."""
    monkeypatch.setenv('LFC_SERVER_URLS', 'http://x.example,http://y.example')
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    urls = _get_lfc_base_urls()
    assert urls == [
        'http://x.example/simulators-ai-perf/',
        'http://y.example/simulators-ai-perf/',
    ]


@pytest.mark.unit
def test_get_lfc_base_urls_env_empty_after_parsing_raises(monkeypatch):
    """LFC_SERVER_URLS that yields zero usable URLs raises RuntimeError."""
    monkeypatch.setenv('LFC_SERVER_URLS', ',,   ,  ,')
    with pytest.raises(RuntimeError, match="no usable URLs"):
        _get_lfc_base_urls()


# Multi-URL fallback download path


@pytest.mark.unit
def test_download_first_url_fails_second_succeeds(monkeypatch):
    """When the first URL raises and the second succeeds, the file lands and
    the second URL is the one actually used."""
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            monkeypatch.setenv(
                'LFC_SERVER_URLS',
                'http://first.invalid,http://second.invalid',
            )
            monkeypatch.delenv('GITHUB_ACTIONS', raising=False)

            seen_urls: list[str] = []

            def fake_urlopen(url, timeout):
                seen_urls.append(url)
                if 'first.invalid' in url:
                    raise Exception('first URL down')
                # Second URL succeeds
                mock_response = MagicMock()
                mock_response.read.return_value = b"v2-content"
                mock_response.status = 200
                cm = MagicMock()
                cm.__enter__.return_value = mock_response
                return cm

            with patch('urllib.request.urlopen', side_effect=fake_urlopen):
                result = resolve_lfc_path("lfc://multi/file.yaml")
                assert result == "__ext/multi/file.yaml"

            # First URL retried 3 times, then second URL tried once (and succeeded)
            first_attempts = sum(1 for u in seen_urls if 'first.invalid' in u)
            second_attempts = sum(1 for u in seen_urls if 'second.invalid' in u)
            assert first_attempts == 3, f"expected 3 retries on first URL, got {first_attempts}"
            assert second_attempts == 1, f"expected 1 attempt on second URL, got {second_attempts}"
            # Content was written from the second URL
            assert Path("__ext/multi/file.yaml").read_text() == "v2-content"
    finally:
        os.chdir(original_cwd)

# No LFC base URL or host may appear in any message.


def _assert_no_sentinel_host(text: str) -> None:
    for host in (SENTINEL_HOST_A, SENTINEL_HOST_B):
        assert host not in text, f'LFC host found in output: {text!r}'


@pytest.mark.unit
def test_redact_lfc_urls_replaces_every_form():
    """Base URL, host:port and bare host are each replaced by a label."""
    base_urls = [
        f'http://{SENTINEL_HOST_A}:8080/simulators-ai-perf/',
        f'http://{SENTINEL_HOST_B}/simulators-ai-perf/',
    ]
    text = (
        f'GET http://{SENTINEL_HOST_A}:8080/simulators-ai-perf/x; '
        f'connect {SENTINEL_HOST_A}:8080; resolve {SENTINEL_HOST_A}; '
        f'resolve {SENTINEL_HOST_B}'
    )
    redacted = _redact_lfc_urls(text, base_urls)
    _assert_no_sentinel_host(redacted)
    assert '<LFC server 1 of 2>' in redacted
    assert '<LFC server 2 of 2>' in redacted


@pytest.mark.unit
def test_redact_lfc_urls_handles_user_info():
    """A base URL with user-info still has its bare host redacted."""
    base_urls = [f'http://user:pass@{SENTINEL_HOST_A}:8080/simulators-ai-perf/']
    text = f'resolve {SENTINEL_HOST_A}; connect {SENTINEL_HOST_A}:8080'
    _assert_no_sentinel_host(_redact_lfc_urls(text, base_urls))


@pytest.mark.unit
def test_redact_lfc_urls_handles_ipv6_literal():
    """An IPv6 literal is redacted bare, not only in its [bracketed] form."""
    base_urls = ['http://[2001:db8::558]:8080/simulators-ai-perf/']
    redacted = _redact_lfc_urls('connect 2001:db8::558 port 8080', base_urls)
    assert '2001:db8::558' not in redacted


# Entries that urllib or wget would escape or normalize: full-width colon,
# control byte, non-ASCII host, inner space.
MALFORMED_URLS = [
    f'http://{SENTINEL_HOST_A}：8080',
    f'http://{SENTINEL_HOST_A}\x01:8080',
    f'http://{SENTINEL_HOST_A}ë:8080',
    f'http://{SENTINEL_HOST_A} x:8080',
    f'http://{SENTINEL_HOST_A}\nextra:8080',
]


@pytest.mark.unit
@pytest.mark.parametrize('ci', [True, False], ids=['ci', 'dev-with-stale-file'])
@pytest.mark.parametrize(
    'urls', MALFORMED_URLS, ids=['fullwidth-colon', 'control-byte', 'non-ascii', 'space', 'newline']
)
def test_lfc_malformed_url_rejected_without_host(monkeypatch, tmp_path, capsys, urls, ci):
    """An entry outside printable ASCII is rejected by position, before any request.

    Dev mode does not fall back to a stale local file for it: the setting is wrong.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('LFC_SERVER_URLS', f'http://ok.invalid,{urls}')
    if ci:
        monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    else:
        monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
        local_path = Path('__ext/test/file.yaml')
        local_path.parent.mkdir(parents=True)
        local_path.write_text('cached content')
        old_time = time.time() - CACHE_AGE_SECONDS - 1
        os.utime(local_path, (old_time, old_time))
    with patch('urllib.request.urlopen') as mock_urlopen:
        with pytest.raises(RuntimeError, match='entry 2 of 2 contains characters outside printable ASCII') as excinfo:
            resolve_lfc_path('lfc://test/file.yaml')
        mock_urlopen.assert_not_called()
    captured = capsys.readouterr()
    _assert_no_sentinel_host(''.join(traceback.format_exception(excinfo.value)) + captured.out + captured.err)


@pytest.mark.unit
@pytest.mark.parametrize('ci', [True, False], ids=['ci', 'dev-with-stale-file'])
@pytest.mark.parametrize(
    'host_suffix',
    ['%01', '%20x', '%01?@cover.invalid', '%01#@cover.invalid'],
    ids=['pct-control-byte', 'pct-space', 'at-in-query', 'at-in-fragment'],
)
def test_lfc_invalid_host_rejected_without_host(monkeypatch, tmp_path, capsys, host_suffix, ci):
    """A host outside the DNS-name / IPv6-literal grammar is rejected by position.

    urllib would decode %-escapes and print the decoded host, which redaction cannot match.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('LFC_SERVER_URLS', f'http://ok.invalid,http://{SENTINEL_HOST_A}{host_suffix}:8080')
    if ci:
        monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    else:
        monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
        local_path = Path('__ext/test/file.yaml')
        local_path.parent.mkdir(parents=True)
        local_path.write_text('cached content')
        old_time = time.time() - CACHE_AGE_SECONDS - 1
        os.utime(local_path, (old_time, old_time))
    with patch('urllib.request.urlopen') as mock_urlopen:
        with pytest.raises(RuntimeError, match='entry 2 of 2 has an invalid host') as excinfo:
            resolve_lfc_path('lfc://test/file.yaml')
        mock_urlopen.assert_not_called()
    captured = capsys.readouterr()
    _assert_no_sentinel_host(''.join(traceback.format_exception(excinfo.value)) + captured.out + captured.err)


@pytest.mark.unit
def test_get_lfc_base_urls_accepts_valid_host_forms(monkeypatch):
    """User-info (even %-encoded), ports, and bracketed IPv6 literals stay valid."""
    monkeypatch.setenv(
        'LFC_SERVER_URLS',
        'http://user%40corp@lfc.example:8080,https://[2001:db8::1]:443/,http://LFC.Example',
    )
    assert len(_get_lfc_base_urls()) == 3


@pytest.mark.unit
@pytest.mark.parametrize('url', MALFORMED_URLS)
def test_redact_lfc_urls_never_raises(url):
    """The redaction helper parses with string operations, so no entry makes it raise."""
    _assert_no_sentinel_host(_redact_lfc_urls(f'error at {url}', [url + '/simulators-ai-perf/']))


@pytest.mark.unit
@pytest.mark.parametrize('rest', ['?c=u@other.invalid', '#u@other.invalid'], ids=['query', 'fragment'])
def test_redact_lfc_urls_ignores_at_after_authority(rest):
    """An '@' in a query or fragment is not user-info; the real host is redacted."""
    redacted = _redact_lfc_urls(f'resolve {SENTINEL_HOST_A}', [f'http://{SENTINEL_HOST_A}{rest}/simulators-ai-perf/'])
    _assert_no_sentinel_host(redacted)


@pytest.mark.unit
def test_lfc_401_traceback_has_no_host(monkeypatch, tmp_path):
    """The full traceback, not only the message, names no host."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('LFC_SERVER_URLS', SENTINEL_URLS)
    http_error = urllib.error.HTTPError(
        url=f'http://{SENTINEL_HOST_A}:8080/simulators-ai-perf/test/file.yaml',
        code=401, msg=f'Authentication required at {SENTINEL_HOST_A}', hdrs={}, fp=None,  # type: ignore[arg-type]
    )
    with patch('urllib.request.urlopen', side_effect=http_error):
        with pytest.raises(RuntimeError, match='Authentication required') as excinfo:
            resolve_lfc_path('lfc://test/file.yaml')
    _assert_no_sentinel_host(''.join(traceback.format_exception(excinfo.value)))


@pytest.mark.unit
def test_lfc_download_failure_has_no_host(monkeypatch, tmp_path):
    """A urllib error that names the host is redacted and not chained (CI mode)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('LFC_SERVER_URLS', SENTINEL_URLS)
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    error = urllib.error.URLError(f"hostname '{SENTINEL_HOST_A}' doesn't match; {SENTINEL_HOST_B}:80 refused")
    with patch('urllib.request.urlopen', side_effect=error):
        with pytest.raises(RuntimeError, match='Download failed') as excinfo:
            resolve_lfc_path('lfc://test/file.yaml')
    _assert_no_sentinel_host(str(excinfo.value))
    assert excinfo.value.__cause__ is None


@pytest.mark.unit
def test_lfc_stale_fallback_warning_has_no_host(monkeypatch, tmp_path, capsys):
    """Dev mode: the stale-cache warning on stderr names no host."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('LFC_SERVER_URLS', SENTINEL_URLS)
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    local_path = Path('__ext/test/file.yaml')
    local_path.parent.mkdir(parents=True)
    local_path.write_text('cached content')
    old_time = time.time() - CACHE_AGE_SECONDS - 1
    os.utime(local_path, (old_time, old_time))
    error = urllib.error.URLError(f'cannot connect to {SENTINEL_HOST_A}:8080')
    with patch('urllib.request.urlopen', side_effect=error):
        assert resolve_lfc_path('lfc://test/file.yaml') == str(local_path)
    captured = capsys.readouterr()
    assert 'using existing local file' in captured.err
    _assert_no_sentinel_host(captured.out + captured.err)
