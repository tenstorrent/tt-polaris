# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""tools/ci/lfc_downloader.sh must never print an LFC base URL or host.

Server URLs are configuration, not output. Each test runs the script with
sentinel hosts under the reserved ``.invalid`` TLD,
so no request can reach a real server, and asserts that neither sentinel host
appears in stdout or stderr.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / 'tools' / 'ci' / 'lfc_downloader.sh'
SENTINEL_HOST_A = 'lfc-sentinel-a.invalid'
SENTINEL_HOST_B = 'lfc-sentinel-b.invalid'

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(shutil.which('wget') is None, reason='lfc_downloader.sh requires wget'),
]


def _run(tmp_path: Path, urls: str, *args: str) -> subprocess.CompletedProcess:
    # Drop proxy settings: a proxy answers for any host, which makes the
    # sentinel hosts look reachable and changes the code path under test.
    env = {k: v for k, v in os.environ.items() if not k.lower().endswith('_proxy')}
    env.update(LFC_SERVER_URLS=urls, GITHUB_ACTIONS='true')
    return subprocess.run(
        ['bash', str(SCRIPT), *args],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )


def _assert_no_sentinel_host(result: subprocess.CompletedProcess) -> None:
    output = result.stdout + result.stderr
    for host in (SENTINEL_HOST_A, SENTINEL_HOST_B):
        assert host not in output, f'LFC host found in output: {output!r}'


def test_dryrun_prints_no_host(tmp_path):
    result = _run(tmp_path, f'http://{SENTINEL_HOST_A}:8080/', '--dryrun', 'archive.tar.gz')
    assert result.returncode == 0, result.stderr
    assert '<LFC server 1 of 1>/simulators-ai-perf/archive.tar.gz' in result.stdout
    _assert_no_sentinel_host(result)


def test_single_url_download_failure_prints_no_host(tmp_path):
    """CI mode with one URL skips the probe; the wget failure output is redacted."""
    result = _run(tmp_path, f'http://{SENTINEL_HOST_A}:8080', 'archive.tar.gz')
    assert result.returncode != 0
    assert 'Failed to download' in result.stderr
    _assert_no_sentinel_host(result)


@pytest.mark.parametrize(
    'bad',
    [f'http://{SENTINEL_HOST_A}：8080', f'http://{SENTINEL_HOST_A}\x01:8080',
     f'http://{SENTINEL_HOST_A}ë:8080', f'http://{SENTINEL_HOST_A} x:8080'],
    ids=['fullwidth-colon', 'control-byte', 'non-ascii', 'space'],
)
def test_malformed_url_rejected_without_host(tmp_path, bad):
    """An entry outside printable ASCII is rejected by position, before wget runs."""
    result = _run(tmp_path, f'http://ok.invalid,{bad}', 'archive.tar.gz')
    assert result.returncode != 0
    assert 'entry 2 of 2 contains characters outside printable ASCII' in result.stderr
    assert 'Downloading from' not in result.stdout
    _assert_no_sentinel_host(result)


@pytest.mark.parametrize(
    'host_suffix',
    ['%01', '%20x', '%01?@cover.invalid', '%01#@cover.invalid'],
    ids=['pct-control-byte', 'pct-space', 'at-in-query', 'at-in-fragment'],
)
def test_invalid_host_rejected_without_host(tmp_path, host_suffix):
    """A host outside the DNS-name / IPv6-literal grammar is rejected before wget runs."""
    result = _run(tmp_path, f'http://ok.invalid,http://{SENTINEL_HOST_A}{host_suffix}:8080', 'archive.tar.gz')
    assert result.returncode != 0
    assert 'entry 2 of 2 has an invalid host' in result.stderr
    assert 'Downloading from' not in result.stdout
    _assert_no_sentinel_host(result)


@pytest.mark.parametrize(
    'url',
    ['http://user%40corp@lfc.example:8080', 'https://[2001:db8::1]:443/', 'http://LFC.Example'],
    ids=['pct-user-info', 'ipv6-literal', 'mixed-case'],
)
def test_valid_host_forms_pass_validation(tmp_path, url):
    """User-info (even %-encoded), ports, and bracketed IPv6 literals stay valid.

    One URL per run: CI mode skips the reachability probe for a single URL, so a dry run needs no network.
    """
    result = _run(tmp_path, url, '--dryrun', 'archive.tar.gz')
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    'urls',
    [f'http://{SENTINEL_HOST_A}\nextra:8080', f'http://ok.invalid,\nhttp://{SENTINEL_HOST_A}'],
    ids=['inside-entry', 'between-entries'],
)
def test_newline_rejected_before_split(tmp_path, urls):
    """A newline would end `read` early and drop the rest unchecked; it is rejected first."""
    result = _run(tmp_path, urls, '--dryrun', 'archive.tar.gz')
    assert result.returncode != 0
    assert 'LFC_SERVER_URLS contains a newline' in result.stderr
    assert 'Downloading from' not in result.stdout
    _assert_no_sentinel_host(result)


def test_user_info_url_download_failure_prints_no_host(tmp_path):
    """A base URL with user-info still has its bare host redacted in wget output."""
    result = _run(tmp_path, f'http://user:pass@{SENTINEL_HOST_A}:8080', 'archive.tar.gz')
    assert result.returncode != 0
    assert 'Failed to download' in result.stderr
    _assert_no_sentinel_host(result)


def test_mixed_case_url_download_failure_prints_no_host(tmp_path):
    """wget prints the host in lowercase; that spelling is redacted too."""
    result = _run(tmp_path, f'http://{SENTINEL_HOST_A.upper()}:8080', 'archive.tar.gz')
    assert result.returncode != 0
    assert 'Failed to download' in result.stderr
    output = (result.stdout + result.stderr).lower()
    assert SENTINEL_HOST_A not in output, f'LFC host found in output: {output!r}'


def test_host_extraction_handles_ipv6_and_user_info():
    """The script's host parsing yields the bare address for an IPv6 literal.

    Runs the script's own LFC_HOSTS block, without the network.
    """
    script = SCRIPT.read_text()
    start = script.index('LFC_HOSTS=()')
    end = script.index('\ndone\n', start) + len('\ndone\n')
    probe = (
        "LFC_BASE_URLS=('http://user:pass@[2001:db8::558]:8080' 'http://h.invalid:80'"
        " 'http://q.invalid?c=u@o.invalid')\n"
        + script[start:end]
        + 'printf "%s\\n" "${LFC_HOSTS[@]}" "${LFC_BARE_HOSTS[@]}"\n'
    )
    out = subprocess.run(['bash', '-c', probe], capture_output=True, text=True, check=True).stdout.split()
    assert out == ['[2001:db8::558]:8080', 'h.invalid:80', 'q.invalid', '2001:db8::558', 'h.invalid', 'q.invalid'], out


def test_redact_filter_hides_ipv6_and_keeps_times(tmp_path):
    """The IP filter in redact_lfc hides IPv6 addresses but not hh:mm:ss times."""
    script = SCRIPT.read_text()
    start = script.index('    done | sed -E \\\n')
    end = script.index("<ip>/g'\n}", start) + len("<ip>/g'")
    sed_cmd = script[start:end].replace('    done | ', '', 1)
    line = 'Connecting to x|2001:db8::558|:8080 and x|2001:db8:0:0:1:0:0:1|:80 at 07:36:25\n'
    out = subprocess.run(['bash', '-c', sed_cmd], input=line, capture_output=True, text=True, check=True).stdout
    assert '2001:db8' not in out, out
    assert '07:36:25' in out, out


def test_multi_url_unreachable_prints_no_host(tmp_path):
    """CI mode with several URLs probes each; the unreachable error lists none."""
    urls = f' http://{SENTINEL_HOST_A}:8080/ , http://{SENTINEL_HOST_B}'
    result = _run(tmp_path, urls, '--verbose', 'archive.tar.gz')
    assert result.returncode != 0
    assert 'candidate LFC server(s) are reachable' in result.stderr
    _assert_no_sentinel_host(result)
