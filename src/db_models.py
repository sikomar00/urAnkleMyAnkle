"""대시보드가 직접 소유하는 MySQL 테이블 세 개의 ORM 정의."""

from datetime import date, datetime, timezone

from sqlalchemy import Date, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    """MySQL DATETIME에 저장할 UTC 시각을 반환한다."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class AdminInfo(Base):
    __tablename__ = "dashboard_admin_info"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    login_id: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    name_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    phone_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    email_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)


class FailureAlert(Base):
    __tablename__ = "dashboard_failure_alerts"
    __table_args__ = (
        UniqueConstraint("observed_on", "asset_tag", "rule_version", name="uq_dashboard_alert_observation"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    observed_on: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    asset_tag: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    failure_points: Mapped[int] = mapped_column(Integer, nullable=False)
    threshold: Mapped[int] = mapped_column(Integer, nullable=False)
    rule_version: Mapped[str] = mapped_column(String(60), nullable=False)
    event_code: Mapped[str] = mapped_column(String(60), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="unreviewed", nullable=False)
    source: Mapped[str] = mapped_column(String(40), default="synthetic_csv", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class SystemLog(Base):
    __tablename__ = "dashboard_system_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False, index=True)
    event_code: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    outcome: Mapped[str] = mapped_column(String(20), nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    asset_tag: Mapped[str | None] = mapped_column(String(80), nullable=True)
    alert_id: Mapped[int | None] = mapped_column(ForeignKey("dashboard_failure_alerts.id"), nullable=True)
    admin_id: Mapped[int | None] = mapped_column(ForeignKey("dashboard_admin_info.id"), nullable=True)
    detail: Mapped[str | None] = mapped_column(String(255), nullable=True)
