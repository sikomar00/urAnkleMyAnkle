"""대시보드 행동, 관측 고장, 실행 오류를 분리한 MySQL ORM 모델."""

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import Boolean, Date, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db_models import Base


def kst_now() -> datetime:
    """MySQL DATETIME에 표시할 한국 표준시(KST, UTC+9)를 반환한다."""
    return datetime.now(timezone(timedelta(hours=9))).replace(tzinfo=None)


class ActionLog(Base):
    __tablename__ = "action_logs"

    log_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, default=kst_now, nullable=False, index=True)
    actor_id: Mapped[str] = mapped_column(String(80), nullable=False, default="ANONYMOUS")
    actor_role: Mapped[str] = mapped_column(String(10), nullable=False, default="UNKNOWN", index=True)
    event_code: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    action_type: Mapped[str] = mapped_column(String(80), nullable=False)
    target_type: Mapped[str | None] = mapped_column(String(40))
    target_id: Mapped[str | None] = mapped_column(String(100))
    action_detail: Mapped[str | None] = mapped_column(Text)
    result_status: Mapped[str] = mapped_column(String(12), nullable=False)
    block_reason: Mapped[str | None] = mapped_column(String(160))


class LoginLog(Base):
    """로그인·로그아웃·세션 상태만 따로 저장하는 인증 로그."""

    __tablename__ = "login_logs"

    log_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, default=kst_now, nullable=False, index=True)
    actor_id: Mapped[str] = mapped_column(String(80), nullable=False, default="ANONYMOUS")
    actor_role: Mapped[str] = mapped_column(String(10), nullable=False, default="UNKNOWN", index=True)
    event_code: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    result_status: Mapped[str] = mapped_column(String(12), nullable=False)
    block_reason: Mapped[str | None] = mapped_column(String(160))
    source: Mapped[str] = mapped_column(String(120), nullable=False)


class FailureLog(Base):
    __tablename__ = "failure_logs"
    __table_args__ = (UniqueConstraint("observation_date", "asset_tag", "part_no", "event_code",
                                     name="uq_failure_observation"),)

    log_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, default=kst_now, nullable=False)
    event_code: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    observation_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    target_date: Mapped[date | None] = mapped_column(Date)
    plant_code: Mapped[str] = mapped_column(String(40), nullable=False)
    asset_tag: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    part_no: Mapped[str] = mapped_column(String(80), nullable=False)
    part_family: Mapped[str | None] = mapped_column(String(80))
    risk_score: Mapped[float | None] = mapped_column(Float)
    risk_level: Mapped[str | None] = mapped_column(String(30))
    main_signal: Mapped[str | None] = mapped_column(String(80))
    model_version: Mapped[str | None] = mapped_column(String(80))
    actual_failure: Mapped[bool | None] = mapped_column(Boolean)
    failure_status: Mapped[str] = mapped_column(String(30), nullable=False)
    source: Mapped[str] = mapped_column(String(40), nullable=False)


class ErrorLog(Base):
    __tablename__ = "error_logs"

    log_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, default=kst_now, nullable=False, index=True)
    actor_id: Mapped[str] = mapped_column(String(80), nullable=False, default="SYSTEM")
    action_type: Mapped[str] = mapped_column(String(80), nullable=False)
    error_code: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    error_type: Mapped[str] = mapped_column(String(80), nullable=False)
    error_message: Mapped[str] = mapped_column(String(500), nullable=False)
    source: Mapped[str] = mapped_column(String(120), nullable=False)
    target_id: Mapped[str | None] = mapped_column(String(100))
    resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
