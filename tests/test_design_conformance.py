"""DESIGN.md §10 수용 기준 중 코드로 검사할 수 있는 항목."""
import ast
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "wireframe_app.py"
HEX = re.compile(r"#[0-9a-fA-F]{6}\b")
# 인라인 style에 두면 테마 전환이 깨지는 키 — 색·폰트·테두리·그림자(치수·배치만 허용).
FORBIDDEN_STYLE_KEYS = {
    "color", "background", "backgroundColor", "fontFamily", "fontSize", "fontWeight", "lineHeight",
    "letterSpacing", "border", "borderTop", "borderBottom", "borderLeft", "borderRight", "borderColor",
    "borderRadius", "outline", "outlineOffset", "boxShadow", "textShadow", "opacity",
}


def test_no_hex_colors_in_dashboard_sources():
    hits = []
    for path in (APP, ROOT / "src" / "dashboard_auth.py", ROOT / "assets" / "03-app.css"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if HEX.search(line):
                hits.append(f"{path.name}:{number}: {line.strip()}")
    assert hits == []


def test_inline_style_dicts_carry_no_color_font_border_or_shadow():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    hits = [
        f"{key.value} (line {key.lineno})"
        for node in ast.walk(tree) if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and key.value in FORBIDDEN_STYLE_KEYS
    ]
    assert hits == []


def test_css_keeps_focus_rings_and_load_order():
    css_files = sorted(p.name for p in (ROOT / "assets").glob("*.css"))
    assert css_files == ["00-tokens.css", "01-type.css", "02-bundle.css", "03-app.css"]
    for name in css_files:
        assert "outline: none" not in (ROOT / "assets" / name).read_text(encoding="utf-8")


paths = ([Path(os.environ["MACHINE_DATA_PATH"])] if os.environ.get("MACHINE_DATA_PATH")
         else [ROOT / "data/raw/synthetic_industrial_machine_data.csv"])
needs_data = pytest.mark.skipif(not any(p.is_file() for p in paths), reason="원본 CSV가 없습니다.")


@needs_data
def test_app_serves_design_assets_and_every_figure_uses_plantfloor_template():
    from dash import dcc

    from src import wireframe_app as w
    from src.theme import C

    assert Path(w.app.config.assets_folder) == ROOT / "assets"

    def graphs(node):
        if isinstance(node, dcc.Graph):
            yield node
        children = getattr(node, "children", None)
        for child in children if isinstance(children, (list, tuple)) else [children]:
            if child is not None and not isinstance(child, (str, int, float)):
                yield from graphs(child)

    layouts = [w.screen_1(), w.screen_2(), w.screen_4()] + [
        w.screen_3(seg_state={**w.DEFAULT_SEG, "task": task}) for task in (0, 1, 2)]
    figures = [g.figure for layout in layouts for g in graphs(layout)]
    assert figures
    for fig in figures:
        assert fig.layout.template.layout.paper_bgcolor == C("surface-card", "light")
        assert not (fig.layout.title and fig.layout.title.text)
