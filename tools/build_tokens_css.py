"""design/tokens.json -> assets/00-tokens.css. 토큰 갱신 시 재실행한다.

design/tokens.json은 Plantfloor 아티팩트 project/tokens.json의 사본에
DESIGN.md 2.3의 파생 토큰(danger-fill, ink-on-danger) 두 개를 더한 것이다.
아티팩트 쪽 토큰이 바뀌면 사본을 다시 받아 두 토큰을 재삽입하고 이 스크립트를 돌린다.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "design/tokens.json"
OUT = ROOT / "assets/00-tokens.css"

d = json.loads(SRC.read_text(encoding="utf-8"))


def color_block(theme: str) -> str:
    return "\n".join(
        f"  --{t['name']}: {t['value'][theme]};" for t in d["color"]["tokens"]
    )


flat = []
for group in ("spacing", "radius", "layout", "opacity", "border"):
    for t in d[group]["tokens"]:
        v = t["value"]
        if isinstance(v, dict):      # shadow 등 테마별 값
            continue
        flat.append(f"  --{t['name']}: {v};")

shadow_light, shadow_dark = [], []
for t in d["shadow"]["tokens"]:
    v = t["value"]
    if isinstance(v, dict):
        shadow_light.append(f"  --{t['name']}: {v['light']};")
        shadow_dark.append(f"  --{t['name']}: {v['dark']};")
    else:
        flat.append(f"  --{t['name']}: {v};")

fams = d["type"]["families"]
flat.append(f"  --font-sans: {fams['sans']};")
flat.append(f"  --font-mono: {fams['mono']};")

css = f""":root {{
{color_block("light")}
{chr(10).join(shadow_light)}
{chr(10).join(flat)}
  color-scheme: light;
}}

:root[data-theme="dark"] {{
{color_block("dark")}
{chr(10).join(shadow_dark)}
  color-scheme: dark;
}}
"""
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(css, encoding="utf-8")
print(f"{OUT} <- {len(d['color']['tokens'])} colors")
