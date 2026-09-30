# DESIGN.md — KYS-dashboard 시각 양식 지시서

> 적용 대상: `KYS-dashboard` 브랜치 (Dash + Plotly, 로그인·역할 계정·감사 로그 포함)
> 근거: Plantfloor 디자인 시스템 (artifact `bdf1f048-58a6-490a-8119-2d5da193e816`)
> Claude Code 세션마다 이 파일을 먼저 읽힌다.

---

## 0. 이 문서의 지위

**다루는 것**: 색 역할, 타이포, 간격·모서리·테두리, 컴포넌트 시각 스타일, Plotly 시각 양식.

**다루지 않는 것**: 화면 레이아웃 규격(12열 그리드·행 높이·카드 치수), 화면 구성, 정보 구조. 이것들은 Plantfloor의 `guidelines/10-layout.md`에 있고 이 문서와 별개다. **레이아웃 토큰(`--layout-*`)은 값만 그대로 유지하고 이 문서에서 배치를 지시하지 않는다.**

**충돌 시 정본 순서**

1. 이 문서
2. Plantfloor `project/tokens.json` (토큰 값의 유일한 정본)
3. Plantfloor `components/bundle.css`
4. Plantfloor 컴포넌트별 `README.md` — **§9에 적힌 항목은 이 순위에서 제외한다(오류 확인됨)**

**표기**: `[원본]`은 Plantfloor에 그대로 있는 규칙, `[파생]`은 KYS-dashboard에 필요해서 기존 토큰에서 새로 정한 규칙이다. 파생 항목은 대비비 계산값을 함께 적었다.

---

## 1. 선행 작업 — 없는 파일부터 만든다

Plantfloor 아티팩트에는 **CSS 변수 파일과 타이포 클래스 정의가 존재하지 않는다.** `bundle.css`는 `var(--surface-card)` 같은 변수를 쓰고 `value-28` 같은 클래스를 참조하지만, 그 정의는 아티팩트 뷰어가 `tokens.json`에서 런타임 생성한다. 파일을 그대로 복사하면 **색도 글자 크기도 하나도 적용되지 않는다.**

### 1.1 파일 배치

```
assets/
  00-tokens.css      ← 생성 필요 (§1.2)
  01-type.css        ← 생성 필요 (§4.1)
  02-bundle.css      ← Plantfloor components/bundle.css 그대로 복사
  03-app.css         ← KYS-dashboard 전용 확장 (§7)
  fonts/             ← Pretendard woff2 (§4.2)
design/
  tokens.json        ← Plantfloor project/tokens.json 사본 + §2.3 파생 토큰 2개
tools/
  build_tokens_css.py
```

Dash는 `assets/`의 CSS를 **파일명 알파벳 순**으로 로드한다. 숫자 접두사는 로드 순서를 고정하기 위한 것이므로 바꾸지 않는다.

`design/tokens.json`은 아티팩트 사본을 그대로 두지 않는다 — §2.3의 `danger-fill`·`ink-on-danger`가 더해진 상태다. 아티팩트 쪽 토큰이 갱신되면 사본을 다시 받아 두 토큰을 재삽입하고 `tools/build_tokens_css.py`를 돌린다. 아티팩트 자체는 수정하지 않는다.

### 1.2 `tools/build_tokens_css.py`

값을 손으로 옮겨 적지 않는다. `design/tokens.json`을 읽어 생성한다.

```python
"""design/tokens.json -> assets/00-tokens.css. 토큰 갱신 시 재실행한다."""
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
```

경로는 실행 위치가 아니라 `__file__` 기준으로 고정한다 — 상대 경로로 두면 `src/`에서 실행할 때 깨진다.

### 1.3 테마 전환

`<html data-theme="dark">`를 토글한다. `prefers-color-scheme`로 자동 전환하지 않는다 — 공장 현장 조명 조건을 앱이 추정할 수 없다. 기본값은 light.

Dash에서는 `index_string`의 `<html>` 태그에 `data-theme`를 넣고, clientside callback으로 `document.documentElement.setAttribute`를 호출한다. Python 콜백으로 CSS 변수를 바꾸려 하지 않는다.

### 1.4 dash-bootstrap-components 주의 `[파생]`

기획서 스택에 dbc가 들어 있다. **dbc 테마 CSS(`dbc.themes.*`)를 함께 로드하지 않는다.** Bootstrap의 `body`, `table`, `.btn`, `.form-control` 기본 스타일이 Plantfloor 규칙을 덮어쓰고, 특히 Bootstrap의 그림자·라운드·색이 "그림자 없음 / `radius-md` 4px / 토큰 색" 원칙을 정면으로 위반한다. dbc를 쓰려면 레이아웃 유틸(`dbc.Row`, `dbc.Col`)만 쓰고 테마는 로드하지 않거나, dbc를 빼고 `html.Div` + `bundle.css`로 간다. **후자를 권장한다.**

---

## 2. 색 — 역할 배정

색은 취향이 아니라 역할이다. **한 색이 두 역할을 겸하지 않는다.** `[원본]`

| 역할 | 무엇을 나르는가 | 토큰 |
|---|---|---|
| 계열 (categorical) | 정체성 — 어느 모델, 어느 군집, 어느 공장 | `series-1` … `series-8`, 고정 순서 |
| 순차 (sequential) | 크기 — 밀도, 건수 | `seq-blue-100` … `seq-blue-700` |
| 상태 (status) | 상태 — 정상·주의·경계·위험 | `status-good` / `-warning` / `-serious` / `-critical` |
| 강조 (accent) | 조작 — 선택, 링크, 포커스 | `accent`, `accent-text`, `accent-fill`, `accent-subtle` |
| 잉크·표면 | 그 밖의 전부 | `ink*`, `surface*`, `border*` |

### 2.1 하드 룰 `[원본]`

- **계열색은 개체에 고정 배정하고 순환시키지 않는다.** 필터로 계열이 사라져도 남은 계열의 색은 바뀌지 않는다. 계열이 9개면 9번째 색을 만들지 말고 `기타`로 접는다.
- **단일 계열 차트는 예외 없이 `series-1`만 쓴다.** 막대 10개도 전부 파랑 하나다. 막대 길이가 이미 크기를 말하므로 값에 따라 색을 짙게 하는 것은 채널 낭비다.
- **상태색은 예약어다.** 고장·경보는 `series-8`(빨강)이 아니라 `status-critical`이다. 반대로 정체성을 뜻하는 계열에 `status-*`를 쓰지 않는다.
- **상태색은 혼자 다니지 않는다.** 8px 점 또는 채워진 마크 + **글자 라벨**을 항상 같이 둔다. 라이트에서 `status-warning` 1.79:1, `status-serious` 2.57:1로 3:1 미만이며, 글자 라벨이 유일한 보완 장치다. 라벨을 빼면 접근성 기준을 잃는다.
- **글자는 데이터 색을 입지 않는다.** 값·축 라벨·범례 글자는 전부 `ink` / `ink-secondary` / `ink-muted`다. 정체성은 글자 **옆의** 색 마크(점, 짧은 선, 스와치)가 나른다. `series-3`(2.74:1) `series-4`(2.11:1) `series-5`(2.62:1)는 라이트 표면에서 글자로 읽히지 않는다.
  - 유일한 예외: 채워진 면 **안**의 인셀 라벨. `seq-blue-400` 이하 → `ink`, `seq-blue-450` 이상 → `ink-on-seq-dark`.
- **축 눈금 글자에 `chart-muted`를 쓰지 않는다.** 라이트 3.50:1이라 본문 기준 미달이다. 눈금·캡션 글자는 `ink-muted`(5.60:1), 밀려난 **마크**에만 `chart-muted`를 쓴다.
- **0은 색이 아니라 표면이다.** 히트맵의 0 셀은 `seq-blue-100`이 아니라 `surface-sunken`이다.
- **그라데이션·이모지·아이콘 장식·애니메이션·전환 효과를 쓰지 않는다.** motion 토큰이 없는 것은 누락이 아니라 결정이다.

### 2.2 전체 색 토큰 (정본은 `tokens.json`)

| 토큰 | light | dark |
|---|---|---|
| `surface-page` | `#f9f9f7` | `#0d0d0d` |
| `surface-card` | `#fcfcfb` | `#1a1a19` |
| `surface-sunken` | `#f0efec` | `#131312` |
| `surface-raised` | `#ffffff` | `#232320` |
| `surface-hover` | `#f3f2ef` | `#232320` |
| `surface-selected` | `#e4effc` | `#16304f` |
| `surface-scrim` | `rgba(11,11,11,0.45)` | `rgba(0,0,0,0.62)` |
| `ink` | `#0b0b0b` | `#ffffff` |
| `ink-secondary` | `#52514e` | `#c3c2b7` |
| `ink-muted` | `#68665f` | `#9a988f` |
| `ink-on-accent` | `#ffffff` | `#0b0b0b` |
| `ink-on-seq-dark` | `#ffffff` | `#ffffff` |
| `border-hairline` | `#e1e0d9` | `#2c2c2a` |
| `border-control` | `#898781` | `#75756a` |
| `border-focus` | `#2a78d6` | `#3987e5` |
| `accent` | `#2a78d6` | `#3987e5` |
| `accent-text` | `#1c5cab` | `#86b6ef` |
| `accent-fill` | `#256abf` | `#3987e5` |
| `accent-subtle` | `#cde2fb` | `#184f95` |
| `series-1` | `#2a78d6` | `#3987e5` |
| `series-2` | `#eb6834` | `#d95926` |
| `series-3` | `#1baf7a` | `#199e70` |
| `series-4` | `#eda100` | `#c98500` |
| `series-5` | `#e87ba4` | `#d55181` |
| `series-6` | `#008300` | `#008300` |
| `series-7` | `#4a3aa7` | `#9085e9` |
| `series-8` | `#e34948` | `#e66767` |
| `seq-blue-100` | `#cde2fb` | `#cde2fb` |
| `seq-blue-150` | `#b7d3f6` | `#b7d3f6` |
| `seq-blue-200` | `#9ec5f4` | `#9ec5f4` |
| `seq-blue-250` | `#86b6ef` | `#86b6ef` |
| `seq-blue-300` | `#6da7ec` | `#6da7ec` |
| `seq-blue-350` | `#5598e7` | `#5598e7` |
| `seq-blue-400` | `#3987e5` | `#3987e5` |
| `seq-blue-450` | `#2a78d6` | `#2a78d6` |
| `seq-blue-500` | `#256abf` | `#256abf` |
| `seq-blue-550` | `#1c5cab` | `#1c5cab` |
| `seq-blue-600` | `#184f95` | `#184f95` |
| `seq-blue-650` | `#104281` | `#104281` |
| `seq-blue-700` | `#0d366b` | `#0d366b` |
| `status-good` | `#0ca30c` | `#0ca30c` |
| `status-good-text` | `#046b04` | `#4ade4a` |
| `status-good-tint` | `#e3f5e3` | `#123b12` |
| `status-good-border` | `#b9e5b9` | `#1f5c1f` |
| `status-warning` | `#fab219` | `#fab219` |
| `status-warning-text` | `#8a5a00` | `#fab219` |
| `status-warning-tint` | `#fdf0d6` | `#3d2e08` |
| `status-warning-border` | `#f2d99a` | `#5c4610` |
| `status-serious` | `#ec835a` | `#ec835a` |
| `status-serious-text` | `#a23f14` | `#f0a27f` |
| `status-serious-tint` | `#fbe6dc` | `#3d2317` |
| `status-serious-border` | `#f2c0aa` | `#5c3722` |
| `status-critical` | `#d03b3b` | `#d03b3b` |
| `status-critical-text` | `#b02626` | `#f07a7a` |
| `status-critical-tint` | `#f9e1e1` | `#3d1616` |
| `status-critical-border` | `#efb8b8` | `#5c2222` |
| `chart-grid` | `#e1e0d9` | `#2c2c2a` |
| `chart-axis` | `#c3c2b7` | `#383835` |
| `chart-muted` | `#898781` | `#898781` |
| `chart-band-critical` | `#f9e1e1` | `#3d1616` |

### 2.3 추가 토큰 — KYS-dashboard 전용 `[파생]`

로그인·계정 삭제·세션 만료에 필요한 파괴적 조작용 채움색이 원본에 없다. `accent-fill`의 상태색 대응물로 아래 두 개만 추가한다. **`00-tokens.css`를 손으로 고쳐 덧붙이지 말고, `design/tokens.json`에 추가해 재생성한다.**

| 토큰 | light | dark | 검증 |
|---|---|---|---|
| `danger-fill` | `#b02626` | `#d03b3b` | 그 위 `#ffffff` = 6.66:1 (light) / 4.80:1 (dark) |
| `ink-on-danger` | `#ffffff` | `#ffffff` | 위와 동일 |

**`ink-on-accent`를 파괴적 버튼에 재사용하지 않는다.** 다크 테마의 `ink-on-accent`는 `#0b0b0b`이고 `status-critical` 위에서 **4.10:1로 4.5:1에 미달한다.** 두 테마 모두 흰 글자여야 한다.

---

## 3. 문구 — 시각 양식의 일부로 취급한다 `[원본]`

라벨이 규칙을 깨면 색과 타이포를 아무리 맞춰도 화면이 거짓말한다.

- 라벨은 **명사구**로 끝내고 문장부호를 붙이지 않는다. `평균 소비 전력`이지 `평균 소비 전력:`이 아니다.
- **단위는 값이 아니라 라벨에 붙인다.** 타일·표에서 라벨은 `평균 소비 전력 (kW)`, 값은 `16.09`. 값 옆에 단위를 붙이면 숫자 열 정렬이 깨진다. 문장 안에서는 값에 붙여 쓴다.
- 문체는 평서형 `-이다`. 사용자를 부르지 않는다 — `확인하세요`가 아니라 `확인 대상`.
- **확정되지 않은 수치는 `—`나 `0`이 아니라 `EmptyState`로 이유를 적는다.** `0`을 그리면 "성능이 0"으로 읽힌다.
- **`합성 데이터 · 교육용` 배지(`.pf-chip-synthetic`)는 조건부로 숨기지 않는다.** 로그인 화면, 앱 헤더, 데이터 화면 하단, 보고서 내보내기 산출물 네 곳에 상시 노출한다.

고정 문구 (바꾸지 않는다):

| 쓴다 | 쓰지 않는다 |
|---|---|
| 고장 표시 있음 / 없음 | 고장 / 정상 |
| 7일 내 고장 표시 확률 | 고장 예정, 고장 예측일 |
| 점검 우선순위 | 수리 지시, 교체 필요 |
| 이상 점수 | 이상 탐지 결과 |
| 군집 1 / 2 / 3 | 정상 군집 / 이상 군집 |
| 기준모델 대비 MAE 개선 | 전력 절감액, 절감 효과 |

---

## 4. 타이포그래피

### 4.1 `assets/01-type.css` — 전문 그대로 생성

클래스명에 `pf-` 접두사가 **없다**(Plantfloor 프리뷰가 `class="pf-kpi__value value-28"` 형태로 쓴다). 전역 클래스이므로 Bootstrap 등과 이름이 충돌하지 않는지 확인하고, 충돌하면 `.pf-app` 하위로 스코프를 좁힌다.

```css
/* Plantfloor 타이포 스케일. tokens.json type.groups 에서 옮긴 값이며 손으로 바꾸지 않는다. */

.value-36 { font-family: var(--font-sans); font-size: 36px; line-height: 40px; font-weight: 600; letter-spacing: -0.01em; font-variant-numeric: proportional-nums; }
.value-28 { font-family: var(--font-sans); font-size: 28px; line-height: 32px; font-weight: 600; letter-spacing: -0.01em; font-variant-numeric: proportional-nums; }
.value-20 { font-family: var(--font-sans); font-size: 20px; line-height: 26px; font-weight: 600; font-variant-numeric: proportional-nums; }

.title-16 { font-family: var(--font-sans); font-size: 16px; line-height: 22px; font-weight: 600; }
.title-14 { font-family: var(--font-sans); font-size: 14px; line-height: 20px; font-weight: 600; }
.body-14  { font-family: var(--font-sans); font-size: 14px; line-height: 20px; font-weight: 400; }
.body-13  { font-family: var(--font-sans); font-size: 13px; line-height: 18px; font-weight: 400; }
.label-12 { font-family: var(--font-sans); font-size: 12px; line-height: 16px; font-weight: 500; }
.micro-11 { font-family: var(--font-sans); font-size: 11px; line-height: 14px; font-weight: 500; letter-spacing: 0.02em; }

.num-13   { font-family: var(--font-mono); font-size: 13px; line-height: 18px; font-weight: 500; font-variant-numeric: tabular-nums; }
.code-12  { font-family: var(--font-mono); font-size: 12px; line-height: 16px; font-weight: 500; }
```

### 4.2 사용 규칙 `[원본]`

| 규칙 | 내용 |
|---|---|
| 큰 숫자 | `value-36` / `-28` / `-20`은 **비례 숫자**. `tabular-nums`를 걸지 않는다 — 큰 크기에서 `121` 같은 값이 헐거워 보인다 |
| 열로 쌓이는 숫자 | 표의 숫자 열, 축 눈금은 `num-13` + `tabular-nums` + **우측 정렬** |
| 식별자 | `asset_tag`, `part_no`, `plant_code`, **사용자 ID·IP·세션 ID**는 `code-12` |
| 카드 제목 | `title-14`. **`title-16`으로 올리지 않는다** — 카드끼리 경쟁한다 |
| 화면 제목 | `title-16` |
| `value-36` | **한 화면에 최대 하나.** 표가 주인공인 화면에서는 아예 쓰지 않는다 |
| `micro-11` | 축 눈금·배지·단위 접미사 **전용**. 문장에 쓰지 않는다 |

**폰트**: Pretendard가 1순위지만 **Plantfloor에 폰트 파일이 실려 있지 않다.** 미설치 환경에서 `system-ui` → `Malgun Gothic` / `Apple SD Gothic Neo`로 떨어지고, 팀이 Windows(LG그램)와 macOS를 섞어 쓰므로 **두 OS에서 글자 폭이 달라져 표 열 정렬이 깨진다.** 내부망 배포이므로 CDN에 의존하지 말고 `assets/fonts/`에 Pretendard woff2를 자체 호스팅하고 `@font-face`를 `00-tokens.css` 앞에 선언한다.

> **미해결**: 폰트 자체 호스팅은 아직 하지 않았다. `assets/fonts/`도 `@font-face`도 없고, `--font-sans`의 폴백 체인만 동작한다. §10 체크리스트의 해당 항목은 미충족 상태다.

---

## 5. 간격·모서리·테두리·불투명도·그림자 `[원본]`

| 토큰 | 값 | 용도 |
|---|---|---|
| `space-1` | 4px | 아이콘–글자 사이, 배지 세로 여백 |
| `space-2` | 8px | 배지·버튼 가로 여백, 타일 내부 줄 간격 |
| `space-3` | 12px | 표 셀 좌우 여백, 컨트롤 안쪽 여백 |
| `space-4` | 16px | 카드 내부 여백, 그리드 거터 |
| `space-5` | 20px | 화면 바깥 여백 |
| `space-6` | 24px | 카드 안 섹션 사이 |
| `space-8` | 32px | 화면 내 큰 블록 사이 |
| `space-10` | 40px | 빈 상태 카드 상하 여백 |

| 토큰 | 값 | 용도 |
|---|---|---|
| `radius-none` | 0 | 표 셀, 히트맵 셀, 스몰 멀티플 패널 — 격자를 유지해야 하는 면 |
| `radius-sm` | 2px | 배지, 인풋, 작은 버튼. 데이터 막대 끝단은 이 값의 두 배(4px) |
| `radius-md` | 4px | 카드·패널·드롭다운 |
| `radius-lg` | 6px | 모달, 큰 팝오버 |
| `radius-pill` | 999px | 필터 칩, 상태 배지 |

| 토큰 | 값 | 용도 |
|---|---|---|
| `stroke-hairline` | 1px | 카드 테두리, 표 구분선, 그리드선. **실선만, 점선 금지** |
| `stroke-mark` | 2px | 라인 굵기, 마크 사이 surface gap, 마커 둘레 링 |
| `stroke-focus` | 2px | 포커스 링 |
| `stroke-bar-max` | 24px | 막대 두께 상한 |
| `opacity-area` | 0.1 | 라인 아래 영역 채움 |
| `opacity-deemph` | 0.28 | 비교군으로 밀려난 마크 |
| `opacity-disabled` | 0.4 | 비활성 컨트롤 |
| `opacity-refetch` | 0.55 | 재조회 중 이전 렌더 |

**그림자**: 카드는 `shadow-none`이다. 층은 `border-hairline` 1px과 `surface-page` 대비가 나눈다. 떠 있는 것(드롭다운·툴팁·모달)만 `surface-raised` + `shadow-popover`를 쓰고, **화면에 동시에 떠 있는 그림자는 하나다.**

**포커스**: `border-focus` 2px 실선 + 바깥 1px 오프셋. 라이트 4.30:1 / 다크 4.79:1. **포커스 링을 제거하지 않는다.** `outline: none`이 코드에 있으면 회귀로 취급한다. 로그인 폼은 키보드만으로 완주되는 화면이므로 이 규칙이 가장 먼저 지켜져야 한다.

**hover / selected**: hover는 `surface-hover`(무채색), 선택은 `surface-selected`(파랑기). 채도로 구분되므로 둘을 바꿔 쓰지 않는다.

**재조회**: 스켈레톤으로 갈아끼우지 않는다. 이전 렌더에 `.pf-refetching`(0.55)을 걸어 붙잡는다. `dcc.Loading`의 기본 스피너를 쓰면 레이아웃이 튀므로 쓰지 않는다.

**호버 타깃**: 마크보다 크게, 최소 24px. 8px 점을 정확히 찍게 만들지 않는다.

---

## 6. 기존 컴포넌트 시각 규칙

`02-bundle.css`가 이미 구현하고 있다. Dash 컴포넌트는 아래 클래스를 붙이기만 한다. **인라인 `style=`로 색·폰트·그림자를 넣지 않는다** — 토큰을 우회하는 순간 테마 전환이 깨진다. 예외는 `height`/`width` 같은 치수뿐이다.

| 컴포넌트 | 루트 클래스 | 시각 규칙 핵심 |
|---|---|---|
| AppHeader | `.pf-header` | `surface-card` 바탕 + 하단 hairline. 로고 이미지 없음. **시계·실시간 갱신 표시 금지**(데이터가 일 단위 정적 파일이다) |
| FilterBar | `.pf-filterbar` | `surface-sunken` 바탕. 컨트롤 높이 32px. 적용 버튼 없음(즉시 반영) |
| Card / ChartCard | `.pf-card` | `surface-card` + hairline + `radius-md` + **그림자 없음**. 제목 줄 `.pf-card__head` |
| KpiTile | `.pf-kpi` | 값 `value-28` + `proportional-nums`, 라벨 `label-12`/`ink-secondary`, 보조 `micro-11`/`ink-muted`. **왼쪽 색 테두리 금지, 타일 하나만 크게 만들지 않는다** |
| StatusBadge | `.pf-badge--{good\|warning\|serious\|critical\|neutral}` | 8px 점 + 글자. 글자색은 마크색이 아니라 `status-*-text`. 다크에서는 `status-*-border` 1px 필수. **배지 안에 숫자를 넣지 않는다** |
| RiskBar | `.pf-risk` | 채움은 `series-1` **단색**. 값에 따라 상태색으로 바꾸거나 그라데이션을 넣지 않는다 — 연속량에 상태색을 쓰면 임계값이 어디인지 화면이 거짓말한다. 숫자를 항상 같이 둔다 |
| Table (공통) | `.pf-table` | 헤더 `surface-sunken` sticky, 행 구분선 hairline. 행 하이라이트는 **한 종류만**(`status-critical-tint`). 정렬 화살표는 `▲▼` 글리프 |
| AssetTile | `.pf-assettile` | **타일 배경을 상태색으로 칠하지 않는다** — 10개가 동시에 칠해지면 화면이 신호를 잃는다. 스파크라인은 `series-1` 2px, 축·범례 없음 |
| SensorStack | `.pf-sensorstack` | **모든 패널 `series-1` 단색.** 패널 높이 동일. 패널마다 범례 금지 |
| Legend | `.pf-legend` | 계열 2개 이상이면 **범례 필수**, 1개면 **범례 금지**(제목이 계열명이다) |
| EmptyState | `.pf-empty` | 일러스트·아이콘 없음. 사과 문구 없음. 로딩 스피너로 쓰지 않음 |

**아이콘**: 세트를 싣지 않는다. 필요한 기호는 네 가지뿐이고 전부 글꼴 문자와 CSS 도형으로 해결한다 — 상태 점(8px 원), 정렬 방향 `▲▼`, 추세 방향 `▲▼`, 펼침 `›`. `dash-iconify`, Font Awesome, 이모지를 도입하지 않는다.

---

## 7. KYS-dashboard 확장 — 인증·관리 UI 시각 양식 `[파생]`

Plantfloor에는 로그인 폼, 버튼 변형, 모달, 토스트, 권한 배지가 **정의되어 있지 않다.** 아래는 기존 토큰만으로 파생시킨 것이며, 새 색을 도입하지 않는다(§2.3의 `danger-fill` 두 개 제외).

`assets/03-app.css`에 아래를 작성한다.

### 7.1 버튼 변형

`bundle.css`에 `.pf-btn`(중립)과 `.pf-btn--primary`만 있다. 세 가지를 추가한다.

```css
.pf-btn--danger {
  background: var(--danger-fill);
  color: var(--ink-on-danger);
  border-color: var(--danger-fill);
}
.pf-btn--ghost {              /* 취소, 뒤로 — 테두리 없는 중립 */
  background: transparent;
  border-color: transparent;
  color: var(--ink-secondary);
}
.pf-btn--ghost:hover { background: var(--surface-hover); color: var(--ink); }
.pf-btn--lg { height: 40px; padding: 0 var(--space-5); }   /* 로그인 제출 전용 */
```

- 한 화면에 `--primary`는 **하나**다. 두 개면 무엇이 기본 동작인지 화면이 말하지 못한다.
- `--danger`는 되돌릴 수 없는 조작(계정 삭제, 강제 로그아웃)에만 쓰고, 항상 확인 모달을 거친다.
- 버튼 글자는 `label-12`, `--lg`만 `body-14`.

### 7.2 폼 필드

`.pf-control`(32px)을 그대로 쓰되 로그인 폼만 44px로 키운다. 로그인은 밀도가 아니라 정확도가 목적인 화면이다.

```css
.pf-field--stack { display: flex; flex-direction: column; gap: var(--space-1); align-items: stretch; }
.pf-control--lg { height: 44px; font-size: 14px; line-height: 20px; }
.pf-control[aria-invalid="true"] { border-color: var(--status-critical); }
.pf-field__error { color: var(--status-critical-text); }      /* micro-11 */
.pf-field__hint  { color: var(--ink-muted); }                 /* micro-11 */
```

- **오류를 테두리 색만으로 표시하지 않는다.** `aria-invalid` + 필드 아래 `.pf-field__error` 글자를 항상 같이 둔다(§2.1의 상태색 규칙과 같은 이유).
- 라벨은 필드 **위**에 `label-12` + `ink-secondary`. placeholder를 라벨 대신 쓰지 않는다 — 입력이 시작되면 라벨이 사라진다.
- 비밀번호 필드에 강도 미터(색 그라데이션 막대)를 넣지 않는다. 그라데이션 금지 규칙에 걸리고, 규칙 충족 여부는 체크리스트 글자로 쓴다.

### 7.3 로그인 화면 표면 규칙

배치는 이 문서 범위 밖이다. **표면만 정한다.**

- 페이지 바탕 `surface-page`, 로그인 카드 `.pf-card`(= `surface-card` + hairline + `radius-md` + **그림자 없음**).
- 카드에 `shadow-popover`를 넣지 않는다. 떠 있는 것이 아니라 놓여 있는 것이다.
- 브랜드 영역은 `title-16` 글자 하나다. **로고 이미지·일러스트·배경 사진을 넣지 않는다.**
- `.pf-chip-synthetic`을 카드 하단에 둔다(§3).
- 로그인 실패는 토스트가 아니라 **폼 상단 인라인 메시지**로 표시한다(§7.5).

### 7.4 권한 배지

역할은 상태가 아니라 **정체성**이다. 따라서 `status-*`를 쓰면 안 되고, 계열색을 글자에 입혀도 안 된다(§2.1). `.pf-badge--neutral`을 재사용한다.

```css
.pf-badge--role { background: var(--surface-sunken); border-color: var(--border-control); color: var(--ink-secondary); }
.pf-badge--role .pf-badge__dot { display: none; }   /* 상태가 아니므로 점을 쓰지 않는다 */
```

`관리자` / `분석가` / `조회` 같은 글자만으로 구분한다. 역할별로 색을 다르게 주지 않는다 — 역할이 3개를 넘어가는 순간 계열색 규칙과 충돌하고, 화면에서 역할은 상태만큼 자주 읽히지 않는다.

### 7.5 알림 — 인라인 우선, 토스트 최소

```css
.pf-notice {
  display: flex; align-items: flex-start; gap: var(--space-2);
  padding: var(--space-3);
  border: var(--stroke-hairline) solid transparent;
  border-radius: var(--radius-sm);
}
.pf-notice--critical { background: var(--status-critical-tint); border-color: var(--status-critical-border); color: var(--status-critical-text); }
.pf-notice--warning  { background: var(--status-warning-tint);  border-color: var(--status-warning-border);  color: var(--status-warning-text); }
.pf-notice--good     { background: var(--status-good-tint);     border-color: var(--status-good-border);     color: var(--status-good-text); }

.pf-toast { background: var(--surface-raised); border: var(--stroke-hairline) solid var(--border-hairline);
            border-radius: var(--radius-md); box-shadow: var(--shadow-popover); padding: var(--space-3) var(--space-4); }
```

- 조작의 결과는 **그 조작이 일어난 자리 근처**에 `.pf-notice`로 띄운다. 토스트는 화면을 떠난 뒤 결과가 나오는 경우(감사 로그 내보내기 완료)에만 쓴다.
- 토스트는 동시에 **하나**만 띄운다(`shadow-popover`는 화면에 하나 규칙).
- 토스트를 자동으로 사라지게 하되 **오류 토스트는 자동으로 사라지지 않게 한다.**
- 등장·퇴장 애니메이션을 넣지 않는다. 시스템에 motion 토큰이 없다.

### 7.6 모달

```css
.pf-scrim { position: fixed; inset: 0; background: var(--surface-scrim); }
.pf-modal { background: var(--surface-raised); border: var(--stroke-hairline) solid var(--border-hairline);
            border-radius: var(--radius-lg); box-shadow: var(--shadow-popover); padding: var(--space-6); }
```

- 제목 `title-16`, 본문 `body-14`, 버튼은 우측 정렬로 `[취소 = --ghost] [실행 = --primary 또는 --danger]`.
- 모달이 떠 있는 동안 배경 카드에 그림자를 추가하지 않는다.
- 세션 만료는 모달로 알리고, 만료 후 화면 데이터를 지운 채 로그인으로 보낸다. **만료된 값을 흐리게 남기지 않는다** — `opacity-refetch`는 "갱신 중"을 뜻하므로 의미가 충돌한다.

### 7.7 감사 로그 표

`.pf-table`을 그대로 쓴다. 추가 규칙만:

- 타임스탬프 `num-13` + `tabular-nums`, 사용자 ID·IP·세션 ID `code-12`, 행위 설명 `body-13` + `ink-secondary`.
- 날짜는 `YYYY-MM-DD HH:mm:ss`로 정규화한다.
- **실패·거부 이벤트에만** `status-critical-tint` 행 배경을 쓴다. 표 하나에 행 하이라이트는 한 종류뿐이다.
- 가상 스크롤로 전량을 그리지 않는다 — 페이지네이션을 쓴다.
- **IP·사용자 식별자를 URL 쿼리스트링이나 파일명에 넣지 않는다.** 내보내기 파일명에는 기간과 행 수만 넣는다.

---

## 8. Plotly 시각 양식 `[원본]`

**CSS 변수를 Plotly가 읽지 못한다.** 색을 Python 쪽에 상수로 이중 관리해야 하며, 이 이중 관리가 어긋나는 것이 이 시스템에서 두 번째로 흔한 회귀다. **`design/tokens.json`을 Python에서도 읽어 상수를 만든다.**

```python
# src/theme.py
import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

_TOKENS_PATH = Path(__file__).resolve().parents[1] / "design/tokens.json"

_TOK = {t["name"]: t["value"] for t in
        json.loads(_TOKENS_PATH.read_text(encoding="utf-8"))["color"]["tokens"]}

def C(name: str, theme: str = "light") -> str:
    return _TOK[name][theme]

def build_template(theme: str) -> go.layout.Template:
    return go.layout.Template(layout=dict(
        paper_bgcolor=C("surface-card", theme),
        plot_bgcolor=C("surface-card", theme),
        colorway=[C(f"series-{i}", theme) for i in range(1, 9)],
        font=dict(family="Pretendard, system-ui, 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif",
                  size=11, color=C("ink-muted", theme)),
        xaxis=dict(showgrid=False, zerolinecolor=C("chart-axis", theme),
                   linecolor=C("chart-axis", theme), ticks="outside",
                   tickcolor=C("chart-muted", theme)),
        yaxis=dict(showgrid=True, gridcolor=C("chart-grid", theme), gridwidth=1,
                   zerolinecolor=C("chart-axis", theme)),
        margin=dict(l=44, r=16, t=8, b=32),
        hovermode="x unified",
        showlegend=False,
    ))

for _t in ("light", "dark"):
    pio.templates[f"plantfloor_{_t}"] = build_template(_t)
```

규칙:

- **템플릿을 한 곳에 등록하고 모든 figure가 그것을 쓰게 한다.** 차트마다 `layout`을 따로 쓰면 반드시 어긋난다.
- `griddash`를 건드리지 않는다 — 기본 실선이다. **점선은 추정치·임계선으로 읽힌다.**
- `xaxis.showgrid=False`, `yaxis.showgrid=True`. 수직 그리드를 켜지 않는다.
- **figure에 `title`을 넣지 않는다.** 제목은 `.pf-card__head`가 그린다.
- `hovermode`는 시계열 `"x unified"`, 산점도 `"closest"`.
- 테마 전환은 콜백에서 `fig.update_layout(template=f"plantfloor_{theme}")`로 갈아끼운다.
- **이중 축(y축 두 개) 금지, 파이·도넛 금지, 막대 하나짜리 막대 차트 금지**(그건 `KpiTile`이다).
- 막대 두께 24px 상한, 데이터 끝단 4px 라운드, 기준선 쪽은 각지게.
- 맞닿은 마크(히트맵 셀, 혼동행렬 셀, 붙은 막대) 사이에 `surface-card` 색 2px 간격. 겹치는 점·끝점에 `surface-card` 2px 링. **마크 둘레에 구분용 테두리를 그리지 않는다.**
- **모든 점에 숫자를 찍지 않는다.** 직접 라벨은 끝점·극값·주인공 계열에만.
- **들어가지 않는 라벨은 자르지 않는다.** 막대 바깥으로 빼거나 툴팁으로 내린다. `overflow: hidden`으로 글자를 잘라내는 것은 라벨이 없는 것보다 나쁘다.
- **툴팁은 값에 이르는 유일한 경로가 될 수 없다.** 라이트에서 3:1 미만인 `series-3/4/5`를 쓰는 차트는 `표 보기` 토글이 선택이 아니라 필수다.

---

## 9. 원본 불일치·누락 — Claude Code가 마주칠 함정

아래는 Plantfloor 소스 안에서 **실제로 어긋나 있는 부분**이다. 그대로 따라 구현하면 깨진다.

| # | 위치 | 문제 | 이 문서의 결정 |
|---|---|---|---|
| 1 | `tokens.json`, `bundle.css` | **`tokens.css`와 타이포 클래스 정의 파일이 존재하지 않는다.** 아티팩트 뷰어가 런타임 생성한다 | §1.2, §4.1로 직접 생성 |
| 2 | `AssetTile/README.md` vs `10-layout.md` | 타일 치수가 **347×88 / 367×72**로, 스파크라인이 **120×32 / 112×28**로 충돌 | `10-layout.md`와 `bundle.css`(`112×28`)가 맞다. README를 버린다 |
| 3 | `README.md` "행 높이는 세 가지만 쓴다" | 정작 화면 ②③④가 88·520·288·32·524를 쓴다 | 실제 규칙은 **"행 높이 합 + 거터 = 928"**이다. 세 가지 제한은 무시 |
| 4 | `README.md` 모델 계열 배정 | `Logistic Regression / Random Forest / XGBoost`로 적혀 있다 | 팀 결정은 **RandomForest가 production**, LR·HGB·XGB는 기준선. 계열 배정을 팀 결정에 맞춰 다시 고정하고, **한 번 고정한 뒤에는 바꾸지 않는다** |
| 5 | `10-layout.md` FilterBar | `공장(plant_code, 4곳 선택)` | 데이터는 **공장 3곳**이다. 3으로 고친다 |
| 6 | 전반 | 부품 단위 위험 우선순위 전제(`부품 출고 금액` KPI 등) | 팀 방향은 **기계 단위 고위험일 분류**로 전환됨. 시각 양식은 그대로 쓰되 라벨·지표를 기계 단위로 다시 쓴다 |
| 7 | `status-warning` / `-serious` | 라이트 표면 대비 1.79:1 / 2.57:1로 **3:1 미달** | 원본 팔레트 값을 유지한다. 대신 **점 + 글자 라벨이 접근성 보완 장치**이므로 라벨을 빼는 코드는 회귀로 취급 |
| 8 | `ink-on-accent` (dark) | `#0b0b0b`, `status-critical` 위에서 **4.10:1** | 파괴적 버튼에 재사용 금지. §2.3의 `ink-on-danger` 사용 |
| 9 | Pretendard | 폰트 파일 미포함, Windows/macOS 혼용 환경 | 자체 호스팅 필수(§4.2). **아직 미해결** |

---

## 10. 수용 기준 — PR 전 체크리스트

### 토큰·파일

- [ ] `assets/00-tokens.css`가 `design/tokens.json`에서 **생성**되었다(손으로 적은 색이 없다)
- [ ] `assets/` CSS 로드 순서가 `00 → 01 → 02 → 03`이다
- [ ] `grep -rn "#[0-9a-fA-F]\{6\}" assets/03-app.css src/` 결과가 **0건**이다(토큰 외 하드코딩 색 없음)
- [ ] `grep -rn "style=" src/`에서 색·폰트·그림자를 인라인으로 넣은 곳이 없다
- [ ] dbc 테마 CSS를 로드하지 않는다
- [ ] Pretendard woff2가 `assets/fonts/`에 있고 `@font-face`가 선언되어 있다

### 색

- [ ] 단일 계열 차트가 전부 `series-1`이다
- [ ] 고장·경보에 `series-8`이 쓰이지 않았다(`status-critical`을 쓴다)
- [ ] 계열색이 글자색으로 쓰인 곳이 없다(인셀 라벨 제외)
- [ ] 축 눈금 글자에 `chart-muted`가 쓰이지 않았다
- [ ] 상태색이 글자 라벨 없이 단독으로 쓰인 곳이 없다
- [ ] 히트맵 0 셀이 `surface-sunken`이다
- [ ] `RiskBar` 채움이 값에 따라 색을 바꾸지 않는다
- [ ] 그라데이션·이모지·아이콘 폰트·CSS transition이 0건이다

### 타이포·표

- [ ] 표 숫자 열이 `num-13` + `tabular-nums` + 우측 정렬이다
- [ ] `value-*`에 `tabular-nums`가 걸려 있지 않다
- [ ] 식별자가 `code-12`다
- [ ] 한 화면에 `value-36`이 둘 이상인 곳이 없다
- [ ] 단위가 값이 아니라 라벨/열 헤더에 붙어 있다
- [ ] 날짜가 전부 `YYYY-MM-DD` (감사 로그는 `YYYY-MM-DD HH:mm:ss`)다

### 상태·상호작용

- [ ] `outline: none`이 0건이다
- [ ] 모든 인터랙티브 요소에 2px `border-focus` 링 + 1px 오프셋이 보인다
- [ ] 폼 오류가 테두리 색만이 아니라 글자로도 표시된다
- [ ] `dcc.Loading` 기본 스피너 대신 `.pf-refetching`을 쓴다
- [ ] 화면에 동시에 떠 있는 `shadow-popover`가 하나 이하다
- [ ] 카드에 그림자가 없다

### 문구

- [ ] `합성 데이터 · 교육용` 배지가 로그인·헤더·데이터 화면·내보내기 산출물 네 곳에 있다
- [ ] §3 금지 문구(`고장 예정`, `교체 필요`, `기계 정지`, `절감액` 등)가 0건이다
- [ ] 값이 없는 자리에 `0`이 아니라 `EmptyState` 사유 문구가 들어 있다
- [ ] 감사 로그 내보내기 파일명과 URL에 사용자 ID·IP가 들어가지 않는다

### Plotly

- [ ] `pio.templates`에 `plantfloor_light` / `plantfloor_dark`가 등록되어 있고 모든 figure가 이를 쓴다
- [ ] figure에 `title`이 들어간 곳이 없다
- [ ] `griddash`를 건드린 곳이 없다
- [ ] 이중 축·파이·도넛이 0건이다
- [ ] 계열 2개 이상 차트에 범례가 있고, 1개 차트에 범례가 없다

---

## 11. 확정된 결정

이 문서가 정하지 않고 남겨두었던 세 항목은 모두 승인되었다.

- **관리 화면의 진입 지점** — 계정 관리·감사 로그를 탭으로 추가하지 않는다. Plantfloor AppHeader의 "탭 4개를 넘기지 않는다"를 지키고, 두 화면은 **헤더 우측 계정 메뉴 하위**로 내린다.
- **감사 로그 표의 본문 스크롤** — 허용한다. 단 **헤더 고정(sticky) + 페이지네이션**을 전제로 하며, 가상 스크롤로 전량을 그리지 않는다(§7.7).
- **역할 수** — 3개까지는 `.pf-badge--role` 단일 스타일로 간다. **3개를 초과하면 배지를 버리고 표의 별도 열로 내린다.**
