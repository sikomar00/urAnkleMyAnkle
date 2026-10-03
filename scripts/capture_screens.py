"""데모 모드 앱을 띄워 화면 4개 × 3개 해상도를 캡처하고 점검 결과를 표로 출력한다.

    python -m scripts.capture_screens [--out docs/images] [--port 8073] [--theme dark]

필요: pip install -r requirements-dev.txt && playwright install chromium
점검: 화면별 innerText의 금지 패턴 건수, 문서 가로 스크롤(scrollWidth > clientWidth) 여부,
      .pf-table 셀 중 글자가 잘린 셀 수, 계정 버튼이 뷰포트 안에 있는지(데모 모드는 계정 메뉴를
      숨기므로 운영 모드 헤더처럼 데모 배지를 숨기고 계정 메뉴를 보인 뒤 잰다),
      /assets CSS 응답 코드·순서, 서버 로그의 DB 접속 오류·401·503.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SIZES = [(1920, 1080), (1536, 864), (1366, 768)]
SCREENS = ["1", "2", "3", "4"]
FORBIDDEN = {
    "치수": r"\d{2,4}\s?×\s?\d{2,4}",
    "와이어프레임": r"와이어프레임",
    "프리셋 N": r"프리셋 \d",
    "빈 대괄호": r"\[ \]",
    "메타 줄": r"메타 줄",
    "localStorage": r"localStorage",
    "엔티티 무관": r"엔티티 무관",
    "고장율": r"고장율",
    "렌더 실패": r"화면을 불러오지 못했습니다",  # render_screen이 예외를 잡았을 때의 문구
}
TRUNCATED_CELLS_JS = """() => [...document.querySelectorAll('#screen-content table.pf-table th, #screen-content table.pf-table td')]
    .filter(c => c.scrollWidth > c.clientWidth + 1).length"""
ACCOUNT_IN_VIEW_JS = """() => {
    const badge = document.getElementById('demo-badge');
    const menu = document.getElementById('profile-menu-toggle').parentElement;
    const saved = [badge.style.display, menu.style.display];
    badge.style.display = 'none'; menu.style.display = 'block';
    const r = document.getElementById('profile-menu-toggle').getBoundingClientRect();
    const inside = r.width > 0 && r.left >= 0 && r.right <= window.innerWidth && r.top >= 0;
    badge.style.display = saved[0]; menu.style.display = saved[1];
    return inside;
}"""
LOG_ERRORS = re.compile(r"OperationalError|pymysql|\b401\b|\b503\b|Traceback")


def wait_until_up(url: str, timeout: float = 60) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"앱이 {timeout:.0f}초 안에 기동하지 않았습니다: {url}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "docs" / "images"))
    parser.add_argument("--port", type=int, default=8073)
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}/"

    env = {**os.environ, "DASHBOARD_HOST": "127.0.0.1", "DASHBOARD_PORT": str(args.port),
           # 데모 모드가 DB를 건드리지 않음을 확인하려고 도달 불가 주소를 준다.
           "MACHINE_DATABASE_URL": "mysql+pymysql://demo:demo@127.0.0.1:1/predictive_maintenance"}
    log_file = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
    server = subprocess.Popen([sys.executable, "-m", "src.wireframe_app", "--demo"], cwd=ROOT, env=env,
                              stdout=log_file, stderr=subprocess.STDOUT)
    rows = []
    css_responses = []
    css_link_order = []
    suffix = "_dark" if args.theme == "dark" else ""
    try:
        wait_until_up(url)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for w, h in SIZES:
                page = browser.new_page(viewport={"width": w, "height": h})
                if not css_responses:
                    page.on("response", lambda r: css_responses.append((r.status, r.url.split("?")[0]))
                            if "/assets/" in r.url and r.url.split("?")[0].endswith(".css") else None)
                page.goto(url)
                page.wait_for_selector("#screen-content *", timeout=30000)
                if not css_link_order:
                    css_link_order = page.eval_on_selector_all(
                        "link[rel=stylesheet]", "ls => ls.map(l => l.href.split('?')[0].split('/').pop())")
                if args.theme == "dark":
                    page.click("#theme-btn")
                    page.wait_for_timeout(800)
                account_in_view = page.evaluate(ACCOUNT_IN_VIEW_JS)
                for sid in SCREENS:
                    page.locator("#screen-tabs .tab").nth(int(sid) - 1).click()
                    page.wait_for_timeout(1500)
                    page.screenshot(path=str(out / f"screen{sid}_{w}x{h}{suffix}.png"), full_page=False)
                    text = page.inner_text("body")
                    counts = {k: len(re.findall(pat, text)) for k, pat in FORBIDDEN.items()}
                    hscroll = page.evaluate(
                        "document.documentElement.scrollWidth > document.documentElement.clientWidth")
                    truncated = page.evaluate(TRUNCATED_CELLS_JS)
                    rows.append((sid, f"{w}x{h}", counts, hscroll, truncated, account_in_view))
                page.close()
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
        log_file.flush()
        log_file.seek(0)
        log_text = log_file.read()
        log_file.close()

    names = list(FORBIDDEN)
    print("화면 | 해상도 | " + " | ".join(names) + " | 가로스크롤 | 잘린 표 셀 | 계정 버튼 뷰포트 안")
    for sid, size, counts, hscroll, truncated, account_in_view in rows:
        print(f"{sid} | {size} | " + " | ".join(str(counts[n]) for n in names)
              + f" | {'예' if hscroll else '아니오'} | {truncated} | {'예' if account_in_view else '아니오'}")
    print("\n/assets CSS 응답:", ", ".join(f"{css_url.rsplit('/', 1)[-1]}={status}"
                                         for status, css_url in sorted(css_responses, key=lambda r: r[1])))
    print("<head> 스타일시트 적용 순서:", " → ".join(css_link_order))
    bad_log = [line for line in log_text.splitlines() if LOG_ERRORS.search(line)]
    print(f"\n서버 로그 오류 줄: {len(bad_log)}건")
    for line in bad_log[:10]:
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
