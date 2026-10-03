"""합성 CSV의 관측일 기준 점수 경보 생성. 미래 예측을 가장하지 않는다."""

from threading import Lock

from .dashboard_data import HIGH_RISK_THRESHOLD, _daily, _data_path, _load_raw
from .db_service import (ANY_FAILURE_EVENT_CODE, ANY_FAILURE_RULE_VERSION,
                         save_observed_alerts)

_scan_lock = Lock()
_last_saved_file: tuple[str, int, int] | None = None


def scan_when_data_changes() -> dict | None:
    """첫 화면 방문 또는 CSV 갱신 시에만 최신 관측일을 DB에 반영한다.

    DB 저장에 실패하면 파일 상태를 기억하지 않아 다음 확인 때 재시도한다.
    여러 브라우저가 동시에 열려도 한 프로세스에서는 한 번씩만 검사한다.
    """
    global _last_saved_file
    path = _data_path().resolve()
    stat = path.stat()
    file_state = (str(path), stat.st_mtime_ns, stat.st_size)
    with _scan_lock:
        if file_state == _last_saved_file:
            return None
        # 현황 표도 변경된 CSV를 읽도록 두 단계의 기존 캐시를 비운다.
        _daily.cache_clear()
        _load_raw.cache_clear()
        result = scan_latest_observation()
        _last_saved_file = file_state
        return result


def scan_latest_observation() -> dict:
    daily = _daily()
    latest = daily["transaction_date"].max()
    latest_rows = daily[daily["transaction_date"].eq(latest)]
    any_failure = latest_rows[latest_rows["failure_points"] > 0]
    severe_failure = latest_rows[latest_rows["failure_points"] >= HIGH_RISK_THRESHOLD]
    any_rows = any_failure[["asset_tag", "failure_points"]].to_dict("records")
    severe_rows = severe_failure[["asset_tag", "failure_points"]].to_dict("records")

    # 같은 설비가 양쪽 기준에 걸리면 별도의 경보 2건으로 저장한다.
    # 두 경보는 rule_version으로 구분되며, 각각 같은 설비·날짜 중복을 막는다.
    any_result = save_observed_alerts(
        latest.date(), any_rows, 1,
        rule_version=ANY_FAILURE_RULE_VERSION,
        event_code=ANY_FAILURE_EVENT_CODE,
    )
    severe_result = save_observed_alerts(latest.date(), severe_rows, HIGH_RISK_THRESHOLD)
    return {
        "observed_on": latest.date().isoformat(),
        "evaluated": len(latest_rows),
        "any_failure": any_result,
        "severity_12": severe_result,
        "created": any_result["created"] + severe_result["created"],
        "existing": any_result["existing"] + severe_result["existing"],
    }
