"""실제로 날짜가 연속된 전날과 오늘의 고장 상태 기준 두 개를 비교합니다."""

from pd_bd_common import run_experiment


if __name__ == '__main__':
    run_experiment('CONSEC')
