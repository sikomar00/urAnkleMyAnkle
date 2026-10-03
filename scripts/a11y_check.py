"""axe-core로 대시보드 4개 화면과 로그인 화면의 접근성을 검사한다.

    python -m scripts.a11y_check [--out docs/a11y]

- axe-core(버전 고정)는 처음 실행할 때 cdnjs에서 .cache/에 내려받는다(저장소에 넣지 않는다).
- 화면 4개는 데모 모드, 로그인은 운영 모드(DB 없이도 GET /login은 열린다)로 띄운다.
- 결과: <out>/<page>.json(위반 전체), <out>/summary.md(영향도별 건수). serious·critical이
  있으면 종료 코드 1.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
AXE_VERSION = "4.10.2"
AXE_URL = f"https://cdnjs.cloudflare.com/ajax/libs/axe-core/{AXE_VERSION}/axe.min.js"
AXE_PATH = ROOT / ".cache" / f"axe-core-{AXE_VERSION}.min.js"
IMPACTS = ["critical", "serious", "moderate", "minor"]
# 운영 모드 서버가 DB를 건드리지 않도록 도달할 수 없는 주소를 준다.
UNREACHABLE_DB = "mysql+pymysql://demo:demo@127.0.0.1:1/predictive_maintenance"


def axe_source() -> str:
    if not AXE_PATH.is_file():
        AXE_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(AXE_URL, AXE_PATH)
    return AXE_PATH.read_text(encoding="utf-8")


def serve(port: int, demo: bool, log) -> subprocess.Popen:
    env = {**os.environ, "DASHBOARD_HOST": "127.0.0.1", "DASHBOARD_PORT": str(port),
           "MACHINE_DATABASE_URL": UNREACHABLE_DB}
    args = [sys.executable, "-m", "src.wireframe_app"] + (["--demo"] if demo else [])
    server = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}/" + ("" if demo else "login")
    for _ in range(120):
        try:
            urllib.request.urlopen(url, timeout=2)
            return server
        except Exception:
            time.sleep(0.5)
    server.terminate()
    raise RuntimeError(f"서버가 기동하지 않았습니다: {url}")


def run_axe(page, source: str) -> list[dict]:
    page.add_script_tag(content=source)
    result = page.evaluate("async () => await axe.run(document, {resultTypes: ['violations']})")
    return [{"id": v["id"], "impact": v["impact"], "help": v["help"], "nodes": len(v["nodes"]),
             "targets": [n["target"] for n in v["nodes"][:10]]} for v in result["violations"]]


MEASURE_JS = """() => {
  const lum = c => { const v = c.match(/[\\d.]+/g).slice(0, 3).map(Number).map(x => x / 255)
      .map(x => x <= 0.03928 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4));
      return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]; };
  const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
  const ph = document.querySelector('#plant-dd .Select-placeholder, #plant-dd .Select-value-label');
  const control = document.querySelector('#plant-dd .Select-control');
  // 데모 모드는 계정 메뉴를 숨기므로 운영 모드 헤더처럼 보이게 한 뒤 잰다.
  const menu = document.getElementById('profile-menu-toggle').parentElement;
  menu.style.display = 'block'; menu.open = true;
  const extend = document.getElementById('session-extend-btn').getBoundingClientRect();
  menu.open = false; menu.style.display = 'none';
  return {
    lang: document.documentElement.getAttribute('lang'),
    placeholder_contrast: ratio(getComputedStyle(ph).color, getComputedStyle(control).backgroundColor),
    dropdowns_named: [...document.querySelectorAll('.dash-dropdown .Select-input input, .dash-dropdown .Select-input')]
        .filter(e => e.getAttribute('aria-label')).length,
    dropdowns_total: document.querySelectorAll('.dash-dropdown').length,
    h1: document.querySelectorAll('h1').length,
    card_titles_h2: [...document.querySelectorAll('.pf-card__title')].every(e => e.tagName === 'H2'),
    session_button: [Math.round(extend.width), Math.round(extend.height)],
  };
}"""

FOCUS_JS = """() => { const e = document.activeElement, s = getComputedStyle(e);
  return {element: e.tagName.toLowerCase() + (e.id ? '#' + e.id : ''), width: s.outlineWidth, style: s.outlineStyle, color: s.outlineColor,
          offset: s.outlineOffset}; }"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "docs" / "a11y"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = axe_source()
    results: dict[str, list[dict]] = {}
    log = open(out / ".server.log", "w")
    demo, ops = serve(8091, True, log), serve(8092, False, log)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.goto("http://127.0.0.1:8091/")
            page.wait_for_selector("#screen-content *", timeout=30000)
            page.wait_for_timeout(1000)
            measured = page.evaluate(MEASURE_JS)
            page.keyboard.press("Tab")
            focus = page.evaluate(FOCUS_JS)
            for index in range(4):
                page.locator("#screen-tabs .tab").nth(index).click()
                page.wait_for_timeout(2500)
                results[f"screen{index + 1}"] = run_axe(page, source)
            login = browser.new_page(viewport={"width": 1366, "height": 768})
            login.goto("http://127.0.0.1:8092/login")
            login.wait_for_timeout(800)
            results["login"] = run_axe(login, source)
            browser.close()
    finally:
        for server in (demo, ops):
            server.terminate()
            server.wait(timeout=10)
        log.close()
        (out / ".server.log").unlink(missing_ok=True)

    lines = [f"# 접근성 검사 결과 (axe-core {AXE_VERSION})", "",
             "`python -m scripts.a11y_check`로 생성한다. 화면 1~4는 데모 모드 1920×1080, 로그인은 1366×768.", "",
             "| 페이지 | " + " | ".join(IMPACTS) + " |", "|---|" + "---|" * len(IMPACTS)]
    blocking = 0
    for name, violations in results.items():
        (out / f"{name}.json").write_text(json.dumps(violations, ensure_ascii=False, indent=2), encoding="utf-8")
        counts = {impact: sum(v["nodes"] for v in violations if v["impact"] == impact) for impact in IMPACTS}
        blocking += counts["critical"] + counts["serious"]
        lines.append(f"| {name} | " + " | ".join(str(counts[i]) for i in IMPACTS) + " |")
        print(name, counts, [f"{v['id']}({v['impact']},{v['nodes']})" for v in violations])
    lines += ["", "건수는 위반 노드 수다. 규칙별 상세는 같은 폴더의 `<페이지>.json`.", "",
              "## 지시서 항목 실측 (화면 ①, 1920×1080)", "", "| 항목 | 값 | 목표 |", "|---|---|---|",
              f"| html lang | {measured['lang']} | ko |",
              f"| 드롭다운 placeholder 대비 | {measured['placeholder_contrast']:.2f}:1 | 4.5:1 이상 |",
              f"| 콤보박스 접근 이름 | {measured['dropdowns_named']}개 입력에 이름 (드롭다운 {measured['dropdowns_total']}개) | 전부 |",
              f"| h1 개수 / 카드 제목 h2 | {measured['h1']} / {'예' if measured['card_titles_h2'] else '아니오'} | 1 / 예 |",
              f"| 세션 연장 버튼 | {measured['session_button'][0]}×{measured['session_button'][1]} | 24×24 이상 |",
              f"| 포커스 링(Tab 첫 요소 `{focus['element']}`) | {focus['width']} {focus['style']} {focus['color']}, 오프셋 {focus['offset']} | 2px border-focus + 1px |"]
    print(measured, focus)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
