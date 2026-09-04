from datetime import datetime, timezone
from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_payment_id", "payment_id"),
        Index("ix_audit_logs_recovery_id", "recovery_id"),
    )

    audit_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    payment_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("payments.payment_id"), nullable=False
    )
    recovery_id: Mapped[str | None] = mapped_column(
        String(50), ForeignKey("recovery_attempts.recovery_id"), nullable=True
    )
    event: Mapped[str] = mapped_column(String(100), nullable=False)
    agent_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    policy_result: Mapped[str | None] = mapped_column(String(500), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    payment: Mapped["Payment"] = relationship("Payment")
    recovery_attempt: Mapped["RecoveryAttempt | None"] = relationship(
        "RecoveryAttempt", back_populates="audit_logs"
    )
