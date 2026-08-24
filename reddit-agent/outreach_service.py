"""Durable, fail-closed control plane for autonomous Reddit outreach."""
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional

from sqlalchemy.orm import Session

from config import get_settings
from database import (
    DailyOutreachQuota,
    Engagement,
    OutboundAction,
    OutreachControl,
    SafetyLog,
    SessionLocal,
    is_sqlite,
)
from reddit_client import get_reddit_client


class OutreachUnavailableError(RuntimeError):
    """Raised when an operator tries to enable a channel that is not ready."""


class OutreachService:
    """Coordinates all outbound writes across independent web and worker processes."""

    def __init__(self):
        self.settings = get_settings()
        self.reddit = get_reddit_client()
        self._dispatch_lock = asyncio.Lock()

    @property
    def daily_limit(self) -> int:
        # The product requirement is an absolute maximum of ten provider calls.
        return min(self.settings.max_daily_engagements, 10)

    @staticmethod
    def _now() -> datetime:
        return datetime.utcnow()

    @staticmethod
    def _quota_day() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    @staticmethod
    def _readiness_message(reason: Optional[str]) -> str:
        messages = {
            "comment_action_unavailable": "The configured Reddit provider does not offer a public-comment action.",
            "dm_action_unavailable": "The configured Reddit provider does not offer a direct-message action.",
            "required_reddit_channels_unavailable": "The required Reddit outreach channels are unavailable.",
        }
        if reason and reason.startswith("composio_unavailable"):
            return "The configured Composio Reddit connection is unavailable."
        if reason and reason.startswith("transport_check_failed"):
            return "The Reddit transport readiness check failed."
        return messages.get(reason or "", "The required Reddit outreach channels are unavailable.")

    @contextmanager
    def _write_session(self) -> Iterator[Session]:
        """Serialize claims in SQLite tests and row-lock them in PostgreSQL."""
        db = SessionLocal()
        try:
            if is_sqlite:
                db.connection().exec_driver_sql("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _ensure_control(db: Session) -> None:
        if db.get(OutreachControl, 1) is None:
            db.add(OutreachControl(id=1, enabled=False, updated_at=datetime.utcnow()))
            db.flush()

    def _locked_control(self, db: Session) -> OutreachControl:
        self._ensure_control(db)
        query = db.query(OutreachControl).filter(OutreachControl.id == 1)
        if not is_sqlite:
            query = query.with_for_update()
        return query.one()

    async def readiness(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Return non-sensitive capability status for the two required channels."""
        try:
            capabilities = await self.reddit.get_outreach_capabilities(force_refresh=force_refresh)
        except Exception as exc:
            return {
                "ready": False,
                "comment": False,
                "dm": False,
                "reason": f"transport_check_failed:{type(exc).__name__}",
                "message": self._readiness_message(f"transport_check_failed:{type(exc).__name__}"),
            }
        comment_ready = bool(capabilities.get("comment"))
        dm_ready = bool(capabilities.get("dm"))
        reason = None
        if not comment_ready or not dm_ready:
            reason = str(capabilities.get("reason") or "required_reddit_channels_unavailable")
        return {
            "ready": comment_ready and dm_ready,
            "comment": comment_ready,
            "dm": dm_ready,
            "reason": reason,
            "message": None if comment_ready and dm_ready else self._readiness_message(reason),
            "transport": capabilities.get("transport"),
        }

    async def get_status(self) -> Dict[str, Any]:
        readiness = await self.readiness(force_refresh=True)
        status = await asyncio.to_thread(self._status_from_database)
        status["readiness"] = readiness
        return status

    def _status_from_database(self) -> Dict[str, Any]:
        quota_day = self._quota_day()
        db = SessionLocal()
        try:
            self._ensure_control(db)
            db.commit()
            control = db.get(OutreachControl, 1)
            quota = db.get(DailyOutreachQuota, quota_day)
            return {
                "enabled": bool(control.enabled),
                "updated_at": control.updated_at.isoformat() if control.updated_at else None,
                "last_readiness_error": control.last_readiness_error,
                "daily_limit": self.daily_limit,
                "reserved_today": quota.reserved_count if quota else 0,
            }
        finally:
            db.close()

    async def set_enabled(self, enabled: bool, operator: str) -> Dict[str, Any]:
        readiness = await self.readiness(force_refresh=True)
        if enabled and not readiness["ready"]:
            reason = readiness.get("reason") or "required_reddit_channels_unavailable"
            await asyncio.to_thread(self._record_enable_refusal, reason, operator)
            raise OutreachUnavailableError(reason)
        # Dispatch holds the database control lock until the provider call has
        # begun. Run the OFF transaction in a worker thread so a waiting lock
        # never blocks that async provider call on the same event loop.
        await asyncio.to_thread(self._persist_enabled, enabled, operator)
        return await self.get_status()

    def _record_enable_refusal(self, reason: str, operator: str) -> None:
        with self._write_session() as db:
            control = self._locked_control(db)
            control.enabled = False
            control.updated_at = self._now()
            control.updated_by = operator
            control.last_readiness_error = reason
            db.add(SafetyLog(
                event_type="outreach_enable_refused",
                severity="warning",
                description=f"Autonomous outreach remained off: {reason}",
                triggered_by=operator,
            ))

    def _persist_enabled(self, enabled: bool, operator: str) -> None:
        with self._write_session() as db:
            control = self._locked_control(db)
            control.enabled = enabled
            control.updated_at = self._now()
            control.updated_by = operator
            control.last_readiness_error = None
            db.add(SafetyLog(
                event_type="outreach_enabled" if enabled else "outreach_disabled",
                severity="warning" if enabled else "info",
                description="Autonomous Reddit comments and DMs enabled" if enabled
                else "Autonomous Reddit outreach disabled; no new actions may be claimed",
                triggered_by=operator,
            ))
            if not enabled:
                # A claimed row has reserved quota but has not started its
                # provider call. Cancel it while holding the same control lock
                # so no new dispatch can begin after OFF is committed.
                db.query(OutboundAction).filter(OutboundAction.status == "claimed").update(
                    {
                        OutboundAction.status: "blocked",
                        OutboundAction.last_error: "outreach_disabled_before_dispatch",
                        OutboundAction.updated_at: self._now(),
                    },
                    synchronize_session=False,
                )

    def _claim(
        self,
        *,
        channel: str,
        dedupe_key: str,
        body: str,
        source_post_id: Optional[str],
        recipient_username: Optional[str],
        subject: Optional[str],
        engagement_id: Optional[str],
    ) -> Dict[str, Any]:
        """Create one durable claim and reserve quota before any provider call."""
        quota_day = self._quota_day()
        with self._write_session() as db:
            control = self._locked_control(db)
            if not control.enabled:
                return {"claimed": False, "status": "blocked", "reason": "outreach_disabled"}

            action = db.query(OutboundAction).filter_by(dedupe_key=dedupe_key).first()
            if action and action.status in {"claimed", "dispatching", "sent", "unknown", "blocked"}:
                return {
                    "claimed": False,
                    "status": action.status,
                    "reason": action.last_error or "duplicate_or_terminal_action",
                    "action_id": action.id,
                }
            if action is None:
                action = OutboundAction(
                    channel=channel,
                    dedupe_key=dedupe_key,
                    body=body,
                    source_post_id=source_post_id,
                    recipient_username=recipient_username,
                    subject=subject,
                    engagement_id=engagement_id,
                    status="queued",
                )
                db.add(action)
                db.flush()

            quota = db.get(DailyOutreachQuota, quota_day)
            if quota is None:
                quota = DailyOutreachQuota(quota_day=quota_day, reserved_count=0, updated_at=self._now())
                db.add(quota)
                db.flush()
            if quota.reserved_count >= self.daily_limit:
                action.status = "queued"
                action.last_error = "daily_outreach_limit_reached"
                action.updated_at = self._now()
                return {
                    "claimed": False,
                    "status": "queued",
                    "reason": "daily_outreach_limit_reached",
                    "action_id": action.id,
                }

            quota.reserved_count += 1
            quota.updated_at = self._now()
            action.status = "claimed"
            action.quota_day = quota_day
            action.claimed_at = self._now()
            action.updated_at = self._now()
            action.last_error = None
            return {"claimed": True, "action_id": action.id, "status": "claimed"}

    def _apply_finalization(
        self,
        db: Session,
        action: OutboundAction,
        *,
        status: str,
        provider_reference: Optional[str] = None,
        provider_result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        action.status = status
        action.provider_reference = provider_reference
        action.provider_result = provider_result or {}
        action.last_error = error
        action.updated_at = self._now()
        if status == "sent":
            action.sent_at = self._now()
        if action.engagement_id:
            engagement = db.get(Engagement, action.engagement_id)
            if engagement:
                engagement.outbound_status = status
                if status == "sent" and provider_reference:
                    engagement.reddit_comment_id = provider_reference
                if status == "sent" and engagement.lead:
                    engagement.lead.last_engaged_at = self._now()

    def _finalize(
        self,
        action_id: str,
        *,
        status: str,
        provider_reference: Optional[str] = None,
        provider_result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._write_session() as db:
            action = db.get(OutboundAction, action_id)
            if action:
                self._apply_finalization(
                    db,
                    action,
                    status=status,
                    provider_reference=provider_reference,
                    provider_result=provider_result,
                    error=error,
                )

    def _mark_dispatching(self, action_id: str) -> Dict[str, Any]:
        """Persist a no-retry marker before waiting for the dispatch fence."""
        with self._write_session() as db:
            control = self._locked_control(db)
            action = db.get(OutboundAction, action_id)
            if not action or action.status != "claimed":
                return {
                    "marked": False,
                    "status": action.status if action else "blocked",
                    "reason": action.last_error if action else "outbound_action_missing",
                }
            if not control.enabled:
                self._apply_finalization(
                    db,
                    action,
                    status="blocked",
                    error="outreach_disabled_before_dispatch",
                )
                return {"marked": False, "status": "blocked", "reason": "outreach_disabled_before_dispatch"}
            action.status = "dispatching"
            action.updated_at = self._now()
            return {"marked": True}

    async def _send(
        self,
        *,
        channel: str,
        dedupe_key: str,
        body: str,
        source_post_id: Optional[str] = None,
        recipient_username: Optional[str] = None,
        subject: Optional[str] = None,
        engagement_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Prevent same-process tasks from synchronously waiting on a database
        # lock while another task awaits the provider under that lock.
        async with self._dispatch_lock:
            readiness = await self.readiness(force_refresh=True)
            if not readiness["ready"]:
                return {"sent": False, "status": "blocked", "reason": readiness.get("reason") or "required_reddit_channels_unavailable"}
            claim = await asyncio.to_thread(
                self._claim,
                channel=channel,
                dedupe_key=dedupe_key,
                body=body,
                source_post_id=source_post_id,
                recipient_username=recipient_username,
                subject=subject,
                engagement_id=engagement_id,
            )
            if not claim["claimed"]:
                return {"sent": False, **claim}

            return await self._dispatch(
                claim["action_id"],
                channel=channel,
                body=body,
                source_post_id=source_post_id,
                recipient_username=recipient_username,
                subject=subject,
            )

    async def _dispatch(
        self,
        action_id: str,
        *,
        channel: str,
        body: str,
        source_post_id: Optional[str],
        recipient_username: Optional[str],
        subject: Optional[str],
    ) -> Dict[str, Any]:
        """Fence dispatch against an OFF transition, then invoke the provider."""
        readiness = await self.readiness(force_refresh=True)
        if not readiness["ready"]:
            await asyncio.to_thread(
                self._finalize,
                action_id,
                status="blocked",
                error=readiness.get("reason") or "required_reddit_channels_unavailable",
            )
            return {
                "sent": False,
                "status": "blocked",
                "reason": readiness.get("reason") or "required_reddit_channels_unavailable",
                "action_id": action_id,
            }
        marker = await asyncio.to_thread(self._mark_dispatching, action_id)
        if not marker["marked"]:
            return {"sent": False, **marker, "action_id": action_id}
        return await self._deliver_with_fence(
            action_id,
            channel=channel,
            body=body,
            source_post_id=source_post_id,
            recipient_username=recipient_username,
            subject=subject,
        )

    async def _deliver_with_fence(
        self,
        action_id: str,
        *,
        channel: str,
        body: str,
        source_post_id: Optional[str],
        recipient_username: Optional[str],
        subject: Optional[str],
    ) -> Dict[str, Any]:
        """Hold the control lock through provider invocation authorization.

        OFF waits for this short critical section. Therefore an OFF response
        cannot be followed by a provider call that had not already begun.
        """
        readiness = await self.readiness(force_refresh=True)
        if not readiness["ready"]:
            await asyncio.to_thread(
                self._finalize,
                action_id,
                status="blocked",
                error=readiness.get("reason") or "required_reddit_channels_unavailable",
            )
            return {
                "sent": False,
                "status": "blocked",
                "reason": readiness.get("reason") or "required_reddit_channels_unavailable",
                "action_id": action_id,
            }
        db = SessionLocal()
        action: Optional[OutboundAction] = None
        try:
            if is_sqlite:
                db.connection().exec_driver_sql("BEGIN IMMEDIATE")
            control = self._locked_control(db)
            action = db.get(OutboundAction, action_id)
            if not action or action.status != "dispatching":
                db.commit()
                return {
                    "sent": False,
                    "status": action.status if action else "blocked",
                    "reason": action.last_error if action else "outbound_action_missing",
                    "action_id": action_id,
                }
            if not control.enabled:
                self._apply_finalization(
                    db,
                    action,
                    status="blocked",
                    error="outreach_disabled_before_provider_call",
                )
                db.commit()
                return {
                    "sent": False,
                    "status": "blocked",
                    "reason": "outreach_disabled_before_provider_call",
                    "action_id": action_id,
                }

            if channel == "comment":
                comment = await self.reddit.post_comment(source_post_id or "", body)
                if not comment:
                    self._apply_finalization(db, action, status="blocked", error="provider_declined_comment")
                    db.commit()
                    return {"sent": False, "status": "blocked", "reason": "provider_declined_comment", "action_id": action_id}
                self._apply_finalization(
                    db,
                    action,
                    status="sent",
                    provider_reference=comment.id,
                    provider_result={"created_utc": comment.created_utc},
                )
                db.commit()
                return {"sent": True, "status": "sent", "action_id": action_id}

            sent = await self.reddit.send_dm(recipient_username or "", subject or "", body)
            if not sent:
                self._apply_finalization(db, action, status="blocked", error="provider_declined_dm")
                db.commit()
                return {"sent": False, "status": "blocked", "reason": "provider_declined_dm", "action_id": action_id}
            self._apply_finalization(db, action, status="sent", provider_result={"accepted": True})
            db.commit()
            return {"sent": True, "status": "sent", "action_id": action_id}
        except Exception as exc:
            # A timeout can mean the provider accepted the action. Preserve that
            # uncertainty and never automatically attempt the same key again.
            error = f"provider_result_unknown:{type(exc).__name__}"
            db.rollback()
            await asyncio.to_thread(self._finalize, action_id, status="unknown", error=error)
            return {"sent": False, "status": "unknown", "reason": error, "action_id": action_id}
        finally:
            db.close()

    async def send_comment(self, engagement_id: str, post_id: str, body: str) -> Dict[str, Any]:
        return await self._send(
            channel="comment",
            dedupe_key=f"comment:{post_id}",
            body=body,
            source_post_id=post_id,
            engagement_id=engagement_id,
        )

    async def send_dm(
        self,
        engagement_id: str,
        source_post_id: str,
        username: str,
        subject: str,
        body: str,
    ) -> Dict[str, Any]:
        return await self._send(
            channel="dm",
            dedupe_key=f"dm:{source_post_id}:{username.lower()}",
            body=body,
            source_post_id=source_post_id,
            recipient_username=username,
            subject=subject,
            engagement_id=engagement_id,
        )


_outreach_service: Optional[OutreachService] = None


def get_outreach_service() -> OutreachService:
    global _outreach_service
    if _outreach_service is None:
        _outreach_service = OutreachService()
    return _outreach_service