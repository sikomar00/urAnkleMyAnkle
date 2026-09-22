"""산업 설비 분석 화면. 저장소 루트에서 python -m src.dashboard 로 실행합니다."""
from pathlib import Path
import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, State, ctx, dcc, html, dash_table, no_update
from dash.dash_table.Format import Format, Scheme, Group

ROOT = Path(__file__).resolve().parents[1]
NAME = 'synthetic_industrial_machine_data.csv'
SENSORS = {
    'vibration_h_mms': ('수평 진동', 'mm/s'), 'vibration_v_mms': ('수직 진동', 'mm/s'),
    'temp_bearing_degC': ('베어링 온도', '°C'), 'temp_motor_degC': ('모터 온도', '°C'),
    'oil_pressure_bar': ('오일 압력', 'bar'), 'load_pct': ('부하율', '%'),
    'shaft_rpm': ('회전 속도', 'RPM'), 'power_consumption_kw': ('소비 전력', 'kW'),
}
TYPE_NAMES = {'CNC Lathe': 'CNC 선반', 'Hydraulic Press': '유압 프레스', 'Belt Conveyor': '벨트 컨베이어', 'Screw Compressor': '스크루 압축기', 'EOT Crane': '천장 크레인'}
BLUE, TEAL, AMBER = '#3565d8', '#158c87', '#c78027'


def load_data():
    candidates = [Path(os.environ['MACHINE_DATA_PATH'])] if os.environ.get('MACHINE_DATA_PATH') else [ROOT / 'data/raw' / NAME, Path.home() / 'Downloads' / NAME]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f'{NAME} 파일을 data/raw 또는 Downloads에 넣어 주세요. 다른 경로는 MACHINE_DATA_PATH로 지정할 수 있습니다.')
    frame = pd.read_csv(path, parse_dates=['transaction_date'])
    required = set(SENSORS) | {'transaction_date', 'asset_tag', 'machine_type', 'plant_code', 'part_no', 'part_description', 'part_family', 'criticality', 'uom', 'unit_cost_inr', 'qty_issued', 'issue_value_inr', 'breakdown_flag'}
    if required - set(frame):
        raise ValueError(f'필수 열이 없습니다: {sorted(required - set(frame))}')
    if frame.duplicated(['asset_tag', 'part_no', 'transaction_date']).any():
        raise ValueError('기계·부품·날짜가 중복된 기록이 있습니다. 원본을 확인해 주세요.')
    sensor_groups = frame.groupby(['asset_tag', 'transaction_date'])[list(SENSORS)].nunique()
    if sensor_groups.gt(1).any().any():
        raise ValueError('같은 기계·날짜의 센서값이 부품별로 다릅니다. 집계 기준을 확인해 주세요.')
    return path, frame


DATA_PATH, RAW = load_data()
# 센서만 중복 제거합니다. 부품별 고장 표시와 출고 기록은 원래 단위를 유지합니다.
DAILY = RAW.drop_duplicates(['asset_tag', 'transaction_date']).sort_values('transaction_date')
ASSETS = RAW[['asset_tag', 'machine_type', 'plant_code']].drop_duplicates().sort_values('asset_tag')
MIN_DATE, MAX_DATE = RAW.transaction_date.min(), RAW.transaction_date.max()
ALL_ASSETS = [{'label': '전체 설비', 'value': 'all'}] + [{'label': f'{r.asset_tag} · {TYPE_NAMES.get(r.machine_type, r.machine_type)}', 'value': r.asset_tag} for r in ASSETS.itertuples()]
START_DATE = (MAX_DATE - pd.Timedelta(days=89)).date().isoformat()
END_DATE = MAX_DATE.date().isoformat()

app = Dash(__name__, title='설비 인사이트 | 산업 설비 분석', assets_folder=str(Path(__file__).parent / 'dashboard_assets'))
server = app.server
app.index_string = '''<!DOCTYPE html><html lang="ko"><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}</head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''


def field(label, child, target=None):
    return html.Div([html.Label(label, htmlFor=target), child], className='field')


def metric(label, value, unit, note, accent=False):
    return html.Div([html.Div(label, className='metric-label'), html.Div([str(value), html.Span(unit, className='metric-unit')], className='metric-value accent' if accent else 'metric-value'), html.Div(note, className='metric-note')], className='metric')


def section_head(title, note='', right=None):
    return html.Div([html.Div([html.H3(title), html.Div(note, className='muted')]), right], className='section-head')


def plot(identity):
    heights = {'sensor-trend': 355, 'sensor-distribution': 265, 'operating-scatter': 260}
    return dcc.Graph(id=identity, style={'height': f'{heights.get(identity, 310)}px'}, config={'displayModeBar': False, 'responsive': True})


def base_figure(fig=None, height=290, unit=''):
    if fig is None:
        fig = go.Figure()
    fig.update_layout(template='plotly_white', height=height, margin=dict(l=48, r=25, t=28, b=42), font=dict(family='Malgun Gothic, Arial', size=11, color='#596b83'), paper_bgcolor='white', plot_bgcolor='white', legend=dict(orientation='h', y=1.14, x=0, font=dict(size=11)), hoverlabel=dict(bgcolor='white', font_size=12))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor='#edf1f6', zeroline=False)
    if unit:
        fig.update_yaxes(title_text=unit)
    return fig


def empty_figure(message='조회할 기록이 없습니다.', height=290):
    fig = base_figure(height=height)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(text=message, x=.5, y=.5, xref='paper', yref='paper', showarrow=False, font=dict(size=13, color='#748298'))
    return fig


# 숫자를 문자열로 바꾸지 않아 숫자 정렬과 CSV 계산이 유지됩니다.
def column(key, label, decimals=None):
    out = {'id': key, 'name': label, 'type': 'numeric' if decimals is not None else 'text'}
    if decimals is not None:
        out['format'] = Format(precision=decimals, scheme=Scheme.fixed, group=Group.yes)
    return out


ASSET_COLUMNS = [column('asset', '설비'), column('type', '기계 종류'), column('plant', '공장'), column('flags', '기준일 표시 부품', 0), column('days', '고장 표시일', 0), column('power', '기준일 전력 (kW)', 2)]
PART_COLUMNS = [column('part', '부품 번호'), column('name', '부품명'), column('status', '기준일 상태'), column('days', '표시일', 0), column('qty', '출고량', 0), column('uom', '단위'), column('cost', '금액 (INR)', 0)]
EVENT_COLUMNS = [column('date', '출고일'), column('qty', '수량', 0), column('uom', '단위'), column('cost', '금액 (INR)', 0)]


def data_table(identity, columns, size=6):
    numeric = [c['id'] for c in columns if c['type'] == 'numeric']
    return dash_table.DataTable(id=identity, data=[], columns=columns, sort_action='native', filter_action='native', page_size=size, cell_selectable=True, tooltip_duration=None, css=[{'selector': '.dash-filter input', 'rule': 'font-size: 11px;'}], style_table={'overflowX': 'auto'}, style_cell={'fontFamily': 'Malgun Gothic, Arial', 'fontSize': 12, 'padding': '13px 12px', 'textAlign': 'left', 'border': 'none', 'borderBottom': '1px solid #e9eef5', 'color': '#34475f', 'maxWidth': '240px', 'overflow': 'hidden', 'textOverflow': 'ellipsis'}, style_header={'backgroundColor': '#f4f7fb', 'color': '#586b83', 'fontWeight': '600', 'fontSize': 11, 'whiteSpace': 'normal', 'height': 'auto'}, style_filter={'backgroundColor': '#fafbfd'}, style_cell_conditional=[{'if': {'column_id': key}, 'textAlign': 'right', 'fontVariantNumeric': 'tabular-nums'} for key in numeric], style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#fcfdff'}, {'if': {'state': 'active'}, 'backgroundColor': '#edf3ff', 'border': '1px solid #bdcfee'}, {'if': {'filter_query': '{status} = "고장 표시 있음"', 'column_id': 'status'}, 'color': '#9c611a', 'backgroundColor': '#fff4e5'}])


def validation_layout():
    return [html.Div([metric('연결 데이터', f'{len(RAW):,}', '행', '기계·부품·날짜별 원본 기록'), metric('센서 관측', f'{len(DAILY):,}', '개', '기계·날짜별 중복 제거'), metric('예측 결과', '미연결', '', '성능과 위험 점수는 아직 산출하지 않았습니다')], className='metrics'),
        html.Div([html.Section([section_head('검증 진행 상태', '모델 성능 수치는 평가 후 연결합니다'), html.Div([html.Div([html.Span(step, className='step-number'), html.Div([html.Strong(title), html.P(desc)]), html.Span(status, className='pill')], className='validation-step') for step, title, desc, status in [('01', '데이터 구조 확인', '기계·부품·날짜 중복 및 센서 반복값 일치 여부를 확인했습니다.', '완료'), ('02', '예측 목표와 분할', '현재 고장 표시가 없는 대상의 미래 7일 정답과 시간 분할을 검증해야 합니다.', '대기'), ('03', '모델 비교', '기준모델과 분류·회귀 모델의 평가 결과가 필요합니다.', '대기'), ('04', '대시보드 연결', '위험 점수·영향 변수·전력 예측을 연결할 예정입니다.', '대기')]])], className='panel'), html.Section([section_head('평가 기준', '예측력과 설명 근거를 구분합니다'), html.Div([html.H4('고장 위험 예측'), html.P('PR-AUC · Recall · F1'), html.P('놓친 고장과 오경보를 혼동행렬로 확인합니다.', className='muted'), html.H4('확률의 신뢰도'), html.P('모델 출력 구간별 실제 고장 표시 비율을 비교합니다.', className='muted'), html.H4('다음 관측일 전력'), html.P('MAE · RMSE'), html.P('전일값 기준모델과 같은 평가 구간에서 비교합니다.', className='muted')], className='criteria')], className='panel')], className='two-column'), html.Div('합성 데이터에서 확인한 결과를 실제 현장의 고장 확률이나 절감 효과로 해석하지 않습니다. 날짜 경계에서 미래 7일 정답이 겹치지 않도록 분리해야 합니다.', className='explanation')]


app.layout = html.Div([
    html.Aside([
        html.Div([html.Div('M', className='brand-mark'), html.Div([html.Strong('설비 인사이트'), html.Span('MAINTENANCE ANALYTICS')])], className='brand'),
        html.Div('분석 범위', className='side-section'),
        field('공장', dcc.Dropdown(id='plant', options=[{'label': '전체 공장', 'value': 'all'}] + [{'label': p, 'value': p} for p in sorted(ASSETS.plant_code.unique())], value='all', clearable=False), 'plant'),
        field('설비', dcc.Dropdown(id='machine', options=ALL_ASSETS, value='all', clearable=False), 'machine'),
        html.Div('조회 기간', className='side-section'),
        field('시작일', dcc.Input(id='start', type='date', value=START_DATE, min=str(MIN_DATE.date()), max=END_DATE), 'start'),
        field('종료일 / 기준일', dcc.Input(id='end', type='date', value=END_DATE, min=str(MIN_DATE.date()), max=END_DATE), 'end'),
        html.Div([html.Button('30일', id='range-30', n_clicks=0), html.Button('90일', id='range-90', n_clicks=0), html.Button('1년', id='range-365', n_clicks=0)], className='quick-ranges'),
        html.Div('센서 분석', className='side-section'),
        field('센서 항목', dcc.Dropdown(id='sensor', options=[{'label': v[0], 'value': k} for k, v in SENSORS.items()], value='vibration_h_mms', clearable=False), 'sensor'),
        field('평소 분포 비교 범위', dcc.Dropdown(id='reference', options=[{'label': '기준일 이전 90개 관측', 'value': '90'}, {'label': '기준일 이전 30개 관측', 'value': '30'}, {'label': '기준일 이전 전체 관측', 'value': 'all'}], value='90', clearable=False), 'reference'),
        html.Div([html.Span('DATA COVERAGE'), html.Strong(f'{MIN_DATE:%Y.%m.%d} — {MAX_DATE:%Y.%m.%d}'), html.P('3개 공장 · 10대 설비 · 20종 부품'), html.P('일 단위 합성 데이터\n과거 기록 분석용')], className='side-footer'),
    ], className='sidebar'),
    html.Div([
        html.Header([html.Div([html.Span('산업 설비 분석', className='breadcrumb'), html.Span(' / ', className='separator'), html.Span(id='breadcrumb-current')]), html.Div([html.Span('합성 데이터', className='pill'), html.Span('모델 미연결', className='pill neutral')], className='header-badges')], className='topbar'),
        html.Main([
            dcc.RadioItems(id='view', options=[{'label': '전체 현황', 'value': 'overview'}, {'label': '설비 상세 분석', 'value': 'detail'}, {'label': '모델 검증', 'value': 'validation'}], value='overview', inline=True, className='nav'),
            html.Div([html.Div([html.Div(id='page-kicker', className='kicker'), html.H1(id='page-title'), html.P(id='page-description', className='page-description')]), html.Div(id='period-caption', className='period-caption')], className='page-heading'),
            html.Div(id='error', role='status'),
            html.Div([
                html.Div(id='overview-metrics', className='metrics'),
                html.Div([html.Section([section_head('설비별 고장 표시일', '선택 기간 중 하나 이상의 부품에 고장 표시가 있었던 날', html.Span('관측 기록', className='pill neutral')), plot('fault-ranking')], className='panel'), html.Section([section_head('일별 소비 전력', '선택 설비의 일별 평균값', html.Span('kW', className='pill neutral')), plot('power-trend')], className='panel')], className='two-column'),
                html.Section([section_head('설비 기록 목록', '행을 클릭하면 해당 설비의 상세 분석으로 이동합니다.', html.Button('현재 목록 CSV', id='export-assets', n_clicks=0, className='button')), data_table('asset-table', ASSET_COLUMNS, 10), html.Div('고장 표시일은 사건 횟수나 미래 위험 순위가 아닙니다. 열 아래 입력칸으로 검색하고 제목의 화살표로 정렬할 수 있습니다.', className='table-note')], className='panel'),
            ], id='overview-page'),
            html.Div([
                html.Div(id='detail-empty', className='empty-state'),
                html.Div([
                    html.Div(id='asset-identity', className='asset-identity'),
                    html.Div(id='detail-metrics', className='metrics'),
                    html.Div([html.Section([section_head('센서 변화와 고장 기록', '', html.Div([html.Span(id='trend-unit', className='pill neutral'), html.Button('부품 선택 해제', id='clear-part', n_clicks=0, className='button')], className='chart-actions')), html.Div(id='trend-caption', className='chart-caption'), plot('sensor-trend')], className='panel trend-panel'), html.Section([section_head('평소 분포에서의 위치'), html.Div(id='distribution-caption', className='chart-caption'), plot('sensor-distribution'), html.Div(id='distribution-summary', className='distribution-summary')], className='panel')], className='analysis-grid'),
                    html.Div([
                        html.Section([section_head('부품별 기록', '행을 선택하면 위 그래프의 고장 표시와 오른쪽 상세가 바뀝니다.', html.Button('현재 목록 CSV', id='export-parts', n_clicks=0, className='button')), data_table('parts-table', PART_COLUMNS, 6), html.Div('센서는 설비 전체의 측정값이며 부품을 바꿔도 같게 유지됩니다.', className='table-note')], className='panel'),
                        html.Section([html.Div(id='part-summary'), html.Details([html.Summary('향후 7일 예측 위험'), html.Div([html.Div('미연결', className='gauge-placeholder'), html.P('검증된 모델 결과를 연결하면 점수를 표시합니다. 빈 게이지는 0점을 의미하지 않습니다.', className='muted')], className='model-placeholder')])], className='panel part-panel'),
                    ], className='records-grid'),
                    html.Details([html.Summary('운전 조건과 출고 내역 더 보기'), html.Div([html.Section([section_head('부하와 소비 전력', '운전 조건에 따른 전력 차이를 살펴보세요.'), plot('operating-scatter')], className='panel'), html.Section([section_head('선택 부품의 출고 내역', '선택 기간 중 수량이 0보다 큰 출고 기록'), data_table('event-table', EVENT_COLUMNS, 5)], className='panel')], className='two-column')], className='more-analysis'),
                ], id='detail-content'),
            ], id='detail-page', style={'display': 'none'}),
            html.Div(validation_layout(), id='validation-page', style={'display': 'none'}),
            html.Footer([html.Span('LS JUMP-UP · 발목잡히지말자'), html.Span('센서: 기계·날짜 / 고장·출고: 기계·부품·날짜')]),
        ]),
    ], className='main-shell'),
    dcc.Store(id='scope'), dcc.Download(id='download'),
])


def filtered_data(plant, machine, start, end):
    start, end = pd.to_datetime(start, errors='coerce'), pd.to_datetime(end, errors='coerce')
    if pd.isna(start) or pd.isna(end) or start > end:
        return RAW.iloc[:0], DAILY.iloc[:0]
    mask = RAW.transaction_date.between(start, end)
    if plant != 'all':
        mask &= RAW.plant_code.eq(plant)
    if machine != 'all':
        mask &= RAW.asset_tag.eq(machine)
    raw = RAW.loc[mask]
    return raw, raw.drop_duplicates(['asset_tag', 'transaction_date']).sort_values('transaction_date')


@app.callback(Output('machine', 'options'), Output('machine', 'value'), Output('view', 'value'), Input('plant', 'value'), Input('asset-table', 'active_cell'), State('machine', 'value'), State('view', 'value'))
def select_machine(plant, active, current, view):
    assets = ASSETS if plant == 'all' else ASSETS[ASSETS.plant_code == plant]
    valid = set(assets.asset_tag)
    options = [ALL_ASSETS[0]] + [o for o in ALL_ASSETS[1:] if o['value'] in valid]
    if ctx.triggered_id == 'asset-table' and active and active.get('row_id') in valid:
        return options, active['row_id'], 'detail'
    return options, current if current in valid or current == 'all' else 'all', view


@app.callback(Output('start', 'value'), Input('range-30', 'n_clicks'), Input('range-90', 'n_clicks'), Input('range-365', 'n_clicks'), State('end', 'value'), prevent_initial_call=True)
def quick_range(_, __, ___, end):
    date = pd.to_datetime(end, errors='coerce')
    if pd.isna(date):
        return no_update
    count = int(ctx.triggered_id.split('-')[1])
    return max(MIN_DATE, date - pd.Timedelta(days=count - 1)).date().isoformat()


@app.callback(Output('overview-page', 'style'), Output('detail-page', 'style'), Output('validation-page', 'style'), Output('page-title', 'children'), Output('page-kicker', 'children'), Output('page-description', 'children'), Output('breadcrumb-current', 'children'), Input('view', 'value'))
def switch_view(view):
    info = {'overview': ('전체 현황', 'FLEET OVERVIEW', '설비별 기록을 비교하고, 자세히 확인할 설비를 선택하세요.'), 'detail': ('설비 상세 분석', 'ASSET ANALYSIS', '센서 변화와 부품 기록을 같은 시간 흐름에서 확인하세요.'), 'validation': ('모델 검증', 'MODEL VALIDATION', '예측의 신뢰도를 확인하기 위한 준비 상태와 평가 기준입니다.')}
    title, kicker, desc = info[view]
    styles = [{'display': 'block' if view == key else 'none'} for key in info]
    return *styles, title, kicker, desc, title


def overview_data(raw, daily):
    latest = raw.transaction_date.max()
    last = raw[raw.transaction_date.eq(latest)]
    dates = raw.groupby(['asset_tag', 'transaction_date']).breakdown_flag.max()
    fault_days = dates.groupby('asset_tag').sum().sort_values()
    cards = [metric('조회 설비', daily.asset_tag.nunique(), '대', f'{daily.transaction_date.nunique():,}개 관측일 · {raw.plant_code.nunique()}개 공장'), metric('기준일 고장 표시 설비', last[last.breakdown_flag.eq(1)].asset_tag.nunique(), '대', f'{latest:%Y.%m.%d} · 부품 1개 이상 표시', True), metric('평균 소비 전력', f'{daily.power_consumption_kw.mean():.2f}', 'kW', '기계·날짜별 관측값 평균')]
    ranking = base_figure(go.Figure(go.Bar(x=fault_days.values, y=fault_days.index, orientation='h', marker_color=BLUE, marker_line_width=0, text=fault_days.values, textposition='outside', cliponaxis=False, hovertemplate='%{y}<br>고장 표시 %{x}일<extra></extra>')), 310)
    ranking.update_layout(margin=dict(l=83, r=38, t=12, b=40), bargap=.35)
    ranking.update_xaxes(title='고장 표시일', range=[0, max(1, float(fault_days.max()) * 1.13)], dtick=1 if fault_days.max() < 5 else None)
    grouped = daily.groupby('transaction_date').power_consumption_kw.mean()
    power = base_figure(go.Figure(go.Scatter(x=grouped.index, y=grouped.values, mode='lines+markers' if len(grouped) == 1 else 'lines', line=dict(color=TEAL, width=2.3), fill='tozeroy', fillcolor='rgba(21,140,135,.055)', hovertemplate='%{x|%Y.%m.%d}<br>%{y:.2f} kW<extra></extra>')), 310, 'kW')
    power.update_xaxes(tickformat='%m.%d')
    rows = []
    for asset, group in last.groupby('asset_tag'):
        row = group.iloc[0]
        rows.append({'id': asset, 'asset': asset, 'type': TYPE_NAMES.get(row.machine_type, row.machine_type), 'plant': row.plant_code, 'flags': int(group.breakdown_flag.sum()), 'days': int(fault_days.get(asset, 0)), 'power': round(float(row.power_consumption_kw), 2)})
    rows.sort(key=lambda r: (-r['flags'], -r['days'], r['asset']))
    return cards, ranking, power, rows


@app.callback(Output('overview-metrics', 'children'), Output('fault-ranking', 'figure'), Output('power-trend', 'figure'), Output('asset-table', 'data'), Output('parts-table', 'data'), Output('parts-table', 'tooltip_data'), Output('scope', 'data'), Output('period-caption', 'children'), Output('error', 'children'), Input('plant', 'value'), Input('machine', 'value'), Input('start', 'value'), Input('end', 'value'))
def refresh_scope(plant, machine, start, end):
    raw, daily = filtered_data(plant, machine, start, end)
    scope = dict(plant=plant, machine=machine, start=start, end=end)
    period = [html.Span('조회 기간'), html.Strong(f'{start or "미선택"} — {end or "미선택"}')]
    if raw.empty:
        scope['empty'] = True
        return [], empty_figure(), empty_figure(), [], [], [], scope, period, html.Div('조회 결과가 없습니다. 시작일·종료일과 선택한 설비를 확인해 주세요.', className='explanation')
    cards, rank, power, rows = overview_data(raw, daily)
    parts = []
    if machine != 'all':
        latest = raw.transaction_date.max()
        for part, group in raw.groupby('part_no'):
            row = group.iloc[0]
            flag = int(group.loc[group.transaction_date.eq(latest), 'breakdown_flag'].max())
            parts.append({'id': part, 'part': part, 'name': row.part_description, 'status': '고장 표시 있음' if flag else '표시 없음', 'flag': flag, 'days': int(group.breakdown_flag.sum()), 'qty': int(group.qty_issued.sum()), 'uom': row.uom, 'cost': float(group.issue_value_inr.sum())})
        parts.sort(key=lambda r: (-r['flag'], -r['days'], r['part']))
    tooltip = [{'name': {'value': r['name'], 'type': 'text'}} for r in parts]
    return cards, rank, power, rows, parts, tooltip, scope, period, ''


def sensor_context(machine, latest, sensor, reference):
    history = DAILY[(DAILY.asset_tag == machine) & (DAILY.transaction_date <= latest)].copy()
    history['moving'] = history[sensor].rolling(7, min_periods=1).mean()
    prior = history[history.transaction_date < latest]
    if reference != 'all':
        prior = prior.tail(int(reference))
    value = float(history.iloc[-1][sensor])
    median = None if prior.empty else float(prior[sensor].median())
    delta = None if median is None else value - median
    percentile = None if prior.empty else float(prior[sensor].le(value).mean() * 100)
    return history, prior, value, median, delta, percentile


@app.callback(Output('parts-table', 'active_cell'), Input('clear-part', 'n_clicks'), Input('machine', 'value'))
def clear_part_selection(_, __):
    return None


def detail_figures(raw, daily, sensor, reference, selected_part):
    machine, latest = daily.iloc[0].asset_tag, daily.transaction_date.max()
    label, unit = SENSORS[sensor]
    history, prior, value, median, delta, percentile = sensor_context(machine, latest, sensor, reference)
    trend = history[history.transaction_date.isin(daily.transaction_date)]
    records = raw if selected_part is None else raw[raw.part_no == selected_part]
    flags = records.groupby('transaction_date').breakdown_flag.sum().reindex(trend.transaction_date, fill_value=0)
    fault_label = '표시 부품 수' if selected_part is None else '선택 부품 표시'
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.08, row_heights=[.79, .21])
    fig.add_trace(go.Scatter(x=trend.transaction_date, y=trend[sensor], name=label, mode='lines' if len(trend) > 1 else 'lines+markers', line=dict(color=BLUE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=trend.transaction_date, y=trend.moving, name='7개 관측일 이동평균', line=dict(color=TEAL, width=2, dash='dot')), row=1, col=1)
    fig.add_trace(go.Bar(x=trend.transaction_date, y=flags.values, name=fault_label, marker_color=AMBER, hovertemplate='%{x|%Y.%m.%d}<br>' + fault_label + ' %{y}<extra></extra>'), row=2, col=1)
    base_figure(fig, 355)
    fig.update_layout(hovermode='x unified', margin=dict(l=50, r=18, t=38, b=32), legend=dict(y=1.16, font=dict(size=10)))
    fig.update_yaxes(title_text=unit, row=1, col=1)
    fig.update_yaxes(title_text='표시', rangemode='tozero', dtick=1 if flags.max() <= 3 else None, row=2, col=1)
    fig.update_xaxes(tickformat='%m.%d', row=2, col=1)
    if selected_part and flags.max() == 0:
        fig.add_annotation(text='선택 기간에 이 부품의 고장 표시 없음', x=.5, y=.06, xref='paper', yref='paper', showarrow=False, font=dict(size=10, color='#75859b'))
    dist = base_figure(go.Figure(), 265, '관측 수')
    if prior.empty:
        dist = empty_figure('기준일 이전 기록이 없습니다.', 265)
        dist_note = '비교 가능한 과거 관측값이 없어 위치를 계산하지 않았습니다.'
    else:
        dist.add_trace(go.Histogram(x=prior[sensor], nbinsx=min(24, max(5, len(prior) // 4)), marker_color='#b0c4e8', marker_line_color='white', marker_line_width=1, hovertemplate=f'{label}: %{{x}} {unit}<br>%{{y}}개 관측<extra></extra>'))
        dist.add_vline(x=value, line_color=BLUE, line_width=2, annotation_text='기준일', annotation_position='top right')
        dist.add_vline(x=median, line_color=TEAL, line_dash='dot', line_width=1.5)
        dist.update_xaxes(title=f'{label} ({unit})')
        dist_note = f'기준일 값 이하인 과거 관측이 {percentile:.1f}%입니다. 고장 확률을 의미하지 않습니다.'
    scatter = base_figure(go.Figure(go.Scatter(x=daily.load_pct, y=daily.power_consumption_kw, mode='markers', customdata=daily.transaction_date.dt.strftime('%Y.%m.%d'), marker=dict(color=BLUE, size=6, opacity=.45), hovertemplate='%{customdata}<br>부하 %{x:.1f}%<br>전력 %{y:.2f} kW<extra></extra>')), 260, '소비 전력 (kW)')
    last_daily = daily.iloc[-1]
    scatter.add_trace(go.Scatter(x=[last_daily.load_pct], y=[last_daily.power_consumption_kw], mode='markers', name='기준일', marker=dict(color=AMBER, size=10, line=dict(color='white', width=1)), hovertemplate='기준일<br>부하 %{x:.1f}%<br>전력 %{y:.2f} kW<extra></extra>'))
    scatter.update_layout(showlegend=False)
    scatter.update_xaxes(title='부하율 (%)')
    return fig, dist, scatter, history, prior, value, median, delta, percentile, dist_note


def part_summary(raw, selected_part):
    if selected_part is None:
        return [html.Div('COMPONENT DETAIL', className='kicker'), html.H3('부품을 선택해 주세요'), html.P('왼쪽 표의 행을 클릭하면 선택 부품의 상태와 출고 이력을 확인할 수 있습니다.', className='muted'), html.Div('현재 그래프 아래에는 해당 설비의 고장 표시 부품 수를 보여드립니다.', className='explanation compact')], []
    group = raw[raw.part_no.eq(selected_part)].sort_values('transaction_date')
    last = group.iloc[-1]
    fault_dates = group[group.breakdown_flag.eq(1)].transaction_date
    issued = group[group.qty_issued.gt(0)].sort_values('transaction_date', ascending=False)
    detail = [html.Div('SELECTED COMPONENT', className='kicker'), html.H3(selected_part), html.P(last.part_description, className='part-name'), html.Span('고장 표시 있음' if last.breakdown_flag else '고장 표시 없음', className='pill warning' if last.breakdown_flag else 'pill neutral'), html.Div([html.Div([html.Span(k), html.Strong(v)], className='fact-row') for k, v in [('기준일', f'{last.transaction_date:%Y.%m.%d}'), ('부품 분류', last.part_family), ('중요도 코드', str(last.criticality)), ('기간 내 표시일', f'{len(fault_dates):,}일'), ('기간 내 최근 표시', fault_dates.max().strftime('%Y.%m.%d') if len(fault_dates) else '없음'), ('출고량', f'{group.qty_issued.sum():,.0f} {last.uom}'), ('출고 금액', f'{group.issue_value_inr.sum():,.0f} INR')]], className='facts'), html.Div('고장 표시가 이어진 날짜를 별개 사건으로 계산하지 않습니다.', className='muted')]
    rows = [{'date': f'{r.transaction_date:%Y-%m-%d}', 'qty': int(r.qty_issued), 'uom': r.uom, 'cost': float(r.issue_value_inr)} for r in issued.itertuples()]
    return detail, rows


@app.callback(Output('detail-content', 'style'), Output('detail-empty', 'children'), Output('asset-identity', 'children'), Output('detail-metrics', 'children'), Output('sensor-trend', 'figure'), Output('sensor-distribution', 'figure'), Output('operating-scatter', 'figure'), Output('trend-caption', 'children'), Output('distribution-caption', 'children'), Output('distribution-summary', 'children'), Output('trend-unit', 'children'), Output('part-summary', 'children'), Output('event-table', 'data'), Input('scope', 'data'), Input('sensor', 'value'), Input('reference', 'value'), Input('parts-table', 'active_cell'))
def refresh_detail(scope, sensor, reference, active):
    scope = scope or {}
    if not scope or scope.get('empty') or scope.get('machine') == 'all':
        message = '왼쪽에서 설비 한 대를 선택하거나, 전체 현황의 설비 목록에서 행을 클릭해 주세요.'
        if scope.get('empty'):
            message = '선택 조건에 맞는 기록이 없습니다.'
        return {'display': 'none'}, message, [], [], empty_figure(), empty_figure(), empty_figure(), '', '', '', '', [], []
    raw, daily = filtered_data(**scope)
    selected_part = active.get('row_id') if active else None
    if selected_part not in set(raw.part_no):
        selected_part = None
    fig, dist, scatter, history, prior, value, median, delta, percentile, dist_note = detail_figures(raw, daily, sensor, reference, selected_part)
    row, latest = daily.iloc[-1], daily.transaction_date.max()
    label, unit = SENSORS[sensor]
    identity = [html.Div([html.Strong(row.asset_tag), html.Span(TYPE_NAMES.get(row.machine_type, row.machine_type)), html.Span(row.plant_code, className='pill neutral')]), html.Span(f'분석 기준일 {latest:%Y.%m.%d}', className='muted')]
    delta_text = '비교 불가' if delta is None else f'{delta:+.2f}'
    delta_note = '기준일 이전 관측값이 없습니다.' if median is None else f'과거 {len(prior)}개 관측의 중앙값 {median:.2f} {unit} 대비'
    latest_parts = raw[raw.transaction_date.eq(latest)]
    cards = [metric(f'기준일 {label}', f'{value:,.2f}', unit, f'{latest:%Y.%m.%d} 관측값'), metric('평소 중앙값 대비 변화', delta_text, unit if delta is not None else '', delta_note), metric('기준일 고장 표시 부품', int(latest_parts.breakdown_flag.sum()), f'/ {len(latest_parts)}종', '미래 예측이 아닌 기준일의 부품 기록', True)]
    caption = f'{label} · 과거 7개 관측일 이동평균 / 아래: ' + (f'{selected_part} 고장 표시' if selected_part else '설비의 고장 표시 부품 수')
    dist_caption = '비교 기록 없음' if prior.empty else f'{prior.transaction_date.min():%Y.%m.%d} – {prior.transaction_date.max():%Y.%m.%d} · {len(prior):,}개 관측'
    distribution = [html.Div([html.Span('과거 분포 내 위치'), html.Strong('—' if percentile is None else f'{percentile:.1f}%')], className='fact-row'), html.P(dist_note), html.Div([html.Span('━ 기준일', style={'color': BLUE}), html.Span('┄ 과거 중앙값', style={'color': TEAL})], className='mini-legend')]
    summary, events = part_summary(raw, selected_part)
    return {}, '', identity, cards, fig, dist, scatter, caption, dist_caption, distribution, unit, summary, events


def build_export(rows, columns, scope, kind):
    keys = [c['id'] for c in columns]
    frame = pd.DataFrame(rows, columns=keys).rename(columns={c['id']: c['name'] for c in columns})
    frame.insert(0, '조회 시작일', scope['start'])
    frame.insert(1, '조회 종료일', scope['end'])
    frame.insert(2, '공장 필터', scope['plant'])
    frame.insert(3, '설비 필터', scope['machine'])
    # Excel에서도 한국어가 보이도록 BOM을 포함합니다.
    content = '\ufeff' + frame.to_csv(index=False)
    return {'content': content, 'filename': f'maintenance_{kind}_{scope["end"]}.csv', 'type': 'text/csv', 'base64': False}


@app.callback(Output('download', 'data'), Input('export-assets', 'n_clicks'), Input('export-parts', 'n_clicks'), State('asset-table', 'derived_virtual_data'), State('asset-table', 'data'), State('parts-table', 'derived_virtual_data'), State('parts-table', 'data'), State('scope', 'data'), prevent_initial_call=True)
def export_csv(_, __, assets_filtered, assets, parts_filtered, parts, scope):
    if not scope or scope.get('empty'):
        return no_update
    if ctx.triggered_id == 'export-assets':
        rows, columns, kind = assets if assets_filtered is None else assets_filtered, ASSET_COLUMNS, 'assets'
    else:
        rows, columns, kind = parts if parts_filtered is None else parts_filtered, PART_COLUMNS, 'parts'
    return build_export(rows, columns, scope, kind)


if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=int(os.environ.get('DASHBOARD_PORT', '8052')))
