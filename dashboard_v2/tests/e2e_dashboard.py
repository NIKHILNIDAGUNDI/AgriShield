"""Browser check of the built dashboard: load, region switch, simulation, no errors.

Needs Playwright with Chromium. Run from the repository root after `npm run build`:
    python dashboard_v2/tests/e2e_dashboard.py [screenshot-directory]
"""

import logging
import sys
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api_server import make_server

VIEWPORTS = {"desktop": (1440, 900), "mobile": (390, 844)}
REGIONS = ("Ukraine", "India", "United States")


def check_viewport(browser, base_url, name, size, screenshots):
    page = browser.new_page(viewport={"width": size[0], "height": size[1]})
    problems = []
    page.on("console", lambda m: m.type == "error" and problems.append(f"console: {m.text}"))
    page.on("pageerror", lambda error: problems.append(f"page: {error}"))
    page.on("requestfailed", lambda r: problems.append(f"request failed: {r.url}"))
    page.on("response", lambda r: r.status >= 400 and problems.append(f"HTTP {r.status}: {r.url}"))

    page.goto(base_url, wait_until="networkidle")
    expect(page.locator(".api-state")).to_contain_text("API")
    expect(page.locator(".metrics-table")).to_contain_text("25.20")
    result = page.locator(".simulation-result")

    for region in REGIONS:
        page.locator(".region-strip button", has_text=region).click()
        expect(page.locator(".analytics-section h2")).to_have_text(f"{region} forecast trail")
        expect(result).to_contain_text("press Recalculate")
        page.get_by_role("button", name="Recalculate").click()
        expect(result).to_contain_text("Model prediction for April 2026")
        expect(page.locator(".simulator-baseline")).to_contain_text("Forecast month: April 2026")

    page.locator(".slider-row input").first.fill("3")
    expect(result).to_contain_text("press Recalculate")
    page.get_by_role("button", name="Recalculate").click()
    expect(result).to_contain_text("Model prediction")

    overflow = page.evaluate("document.documentElement.scrollWidth - innerWidth")
    assert overflow <= 0, f"{name}: page scrolls sideways by {overflow}px"
    alerts = page.locator(".alerts-panel").bounding_box()
    assert alerts["x"] >= 0 and alerts["x"] + alerts["width"] <= size[0], f"{name}: alerts cut off"
    if name == "desktop":
        hidden = page.locator(".heatmap-scroll").evaluate("e => e.scrollWidth - e.clientWidth")
        assert hidden <= 0, f"heatmap hides {hidden}px"

    if screenshots:
        page.evaluate("scrollTo({top: 0, behavior: 'instant'})")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(screenshots / f"{name}.png"), full_page=True)
    page.close()
    assert not problems, f"{name}: {problems}"
    print(f"PASS {name} {size[0]}px: load, region switch, simulation, 0 console/network errors")


def main():
    logging.disable(logging.CRITICAL)  # keep request logs out of the report
    screenshots = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if screenshots:
        screenshots.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        server = make_server(
            host="127.0.0.1", port=0, artifact_key="",
            audit_path=Path(directory) / "audit.jsonl", dashboard_root=ROOT,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                for name, size in VIEWPORTS.items():
                    check_viewport(browser, base_url, name, size, screenshots)
                browser.close()
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()


if __name__ == "__main__":
    main()
