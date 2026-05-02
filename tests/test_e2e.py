"""Browser-level E2E tests using Playwright.

These require a running Streamlit server and a Chromium browser install.

Setup:
    pip install pytest-playwright playwright
    playwright install chromium

Run:
    pytest tests/test_e2e.py -m e2e -v
"""
from __future__ import annotations

import os
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api", reason="playwright not installed — skipping E2E suite")

FIXTURES = Path(__file__).parent / "fixtures"
APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="session")
def streamlit_url(tmp_path_factory):
    """Start a Streamlit server for the session and return its URL."""
    data_dir = tmp_path_factory.mktemp("e2e_data")
    env = {
        **os.environ,
        "APP_ENV": "test_e2e",
        "APP_DATA_ROOT": str(data_dir),
        "OPENAI_API_KEY": "",
    }
    proc = subprocess.Popen(
        [
            "python", "-m", "streamlit", "run", APP_PATH,
            "--server.port", "8502",
            "--server.headless", "true",
            "--server.runOnSave", "false",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(20):
        try:
            urllib.request.urlopen("http://localhost:8502", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.terminate()
        pytest.fail("Streamlit server did not start within 10 seconds")

    yield "http://localhost:8502"
    proc.terminate()


class TestE2EUploadFlow:
    def test_app_loads(self, page, streamlit_url):
        page.goto(streamlit_url)
        page.wait_for_load_state("networkidle")
        assert "Document" in page.title() or page.locator("h1").count() > 0

    def test_upload_invoice_and_see_results(self, page, streamlit_url):
        fixture = FIXTURES / "invoice_format_a_full.pdf"
        if not fixture.exists():
            pytest.skip("invoice_format_a_full.pdf not in fixtures")
        page.goto(streamlit_url)
        page.wait_for_load_state("networkidle")
        upload_input = page.locator("input[type=file]").first
        upload_input.set_input_files(str(fixture))
        page.wait_for_timeout(1000)
        process_btn = page.get_by_role("button", name_re=r"[Pp]rocess")
        if process_btn.count() > 0:
            process_btn.first.click()
            page.wait_for_load_state("networkidle")
        page_text = page.inner_text("body").lower()
        assert any(kw in page_text for kw in ("invoice", "vendor", "extracted", "validation")), (
            f"No extraction results found. Page text preview: {page_text[:500]}"
        )

    def test_duplicate_upload_shows_duplicate_message(self, page, streamlit_url):
        fixture = FIXTURES / "invoice_format_a_full.pdf"
        dup = FIXTURES / "invoice_format_a_full_dup.pdf"
        if not fixture.exists() or not dup.exists():
            pytest.skip("duplicate fixtures not present")
        page.goto(streamlit_url)
        page.wait_for_load_state("networkidle")
        upload_input = page.locator("input[type=file]").first
        upload_input.set_input_files(str(fixture))
        page.wait_for_timeout(500)
        process_btn = page.get_by_role("button", name_re=r"[Pp]rocess")
        if process_btn.count() > 0:
            process_btn.first.click()
            page.wait_for_load_state("networkidle")
        upload_input.set_input_files(str(dup))
        page.wait_for_timeout(500)
        if process_btn.count() > 0:
            process_btn.first.click()
            page.wait_for_load_state("networkidle")
        page_text = page.inner_text("body").lower()
        assert "duplicate" in page_text, (
            f"Expected 'duplicate' in page after uploading dup. Preview: {page_text[:500]}"
        )

    def test_bulk_upload_processes_multiple_docs(self, page, streamlit_url):
        pdfs = [
            FIXTURES / "invoice_format_a_full.pdf",
            FIXTURES / "lab_report_format_a_full.pdf",
        ]
        missing = [p.name for p in pdfs if not p.exists()]
        if missing:
            pytest.skip(f"Missing fixtures: {missing}")
        page.goto(streamlit_url)
        page.wait_for_load_state("networkidle")
        bulk_tab = page.get_by_role("tab", name_re=r"[Bb]ulk")
        if bulk_tab.count() > 0:
            bulk_tab.first.click()
            page.wait_for_timeout(500)
        upload_input = page.locator("input[type=file][multiple]").first
        if upload_input.count() == 0:
            upload_input = page.locator("input[type=file]").nth(1)
        upload_input.set_input_files([str(p) for p in pdfs])
        page.wait_for_timeout(500)
        process_btn = page.get_by_role("button", name_re=r"[Pp]rocess")
        if process_btn.count() > 0:
            process_btn.first.click()
            page.wait_for_load_state("networkidle")
        page_text = page.inner_text("body").lower()
        assert any(kw in page_text for kw in ("processed", "invoice", "lab", "results")), (
            f"No bulk results found. Preview: {page_text[:500]}"
        )
