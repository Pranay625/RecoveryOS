from datetime import datetime, timezone
from sqlalchemy import DateTime, ForeignKey, Index, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base


class RecoveryAttempt(Base):
    __tablename__ = "recovery_attempts"
    __table_args__ = (
        Index("ix_recovery_attempts_payment_id", "payment_id"),
    )

    recovery_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    payment_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("payments.payment_id"), nullable=False
    )
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    reason: Mapped[str] = mapped_column(String(1000), nullable=False)
    ml_probability: Mapped[float | None] = mapped_column(Numeric(6, 4), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    payment: Mapped["Payment"] = relationship("Payment", back_populates="recovery_attempts")
    audit_logs: Mapped[list["AuditLog"]] = relationship(
        "AuditLog", back_populates="recovery_attempt"
    )
