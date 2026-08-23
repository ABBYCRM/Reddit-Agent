"""
CaseClosedFL Reddit Agent - Qualifier Agent
Scores leads, extracts contact info, and manages lead pipeline.
"""
import logging
import re
from typing import Dict, Any, Optional
from datetime import datetime

from config import get_settings
from database import get_db_session, Lead, Engagement, AgentRun

logger = logging.getLogger(__name__)
settings = get_settings()


class QualifierAgent:
    """
    Manages lead qualification pipeline:
    - Scores leads based on engagement and profile data
    - Extracts volunteered contact information
    - Updates lead temperature and status
    - Generates daily lead reports
    """

    def score_lead(self, lead: Lead) -> int:
        """
        Calculate lead score based on:
        - Intent score (from discovery)
        - Engagement history
        - Reply sentiment
        - Contact volunteered
        """
        base_score = lead.intent_score * 0.4 + lead.qualification_score * 0.4

        # Bonus for contact volunteered
        if lead.contact_volunteered:
            base_score += 15

        # Bonus for warm/hot temperature
        if lead.lead_temperature == "warm":
            base_score += 5
        elif lead.lead_temperature == "hot":
            base_score += 10

        # Penalty if has attorney
        if lead.has_attorney:
            base_score -= 50

        # Penalty if old
        days_since = (datetime.utcnow() - lead.discovered_at).days
        if days_since > 30:
            base_score -= 10

        return min(100, max(0, int(base_score)))

    def extract_contact(self, text: str) -> Optional[Dict[str, str]]:
        """
        Extract voluntarily provided contact info from text.
        Only extracts if user clearly volunteers it.
        """
        contact = {}

        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact["email"] = emails[0]

        # Phone pattern (US)
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phones = re.findall(phone_pattern, text)
        if phones:
            contact["phone"] = phones[0]

        return contact if contact else None

    async def process_new_replies(self, replies: list) -> Dict[str, Any]:
        """Process replies found by monitor agent."""
        db = get_db_session()
        results = {
            "replies_processed": 0,
            "leads_updated": 0,
            "contacts_extracted": 0,
            "status_changes": []
        }

        for reply in replies:
            try:
                # Find associated lead
                lead = db.query(Lead).filter(
                    Lead.reddit_post_id == reply["post_id"]
                ).first()

                if not lead:
                    continue

                # Check if reply is from OP (original poster)
                if reply["author"] == lead.reddit_username:
                    # Extract any volunteered contact
                    contact = self.extract_contact(reply["body"])
                    if contact:
                        lead.contact_volunteered = True
                        lead.contact_method = "comment_reply"
                        lead.contact_value = str(contact)
                        results["contacts_extracted"] += 1

                    # Update temperature based on engagement
                    lead.lead_temperature = "warm"
                    lead.last_engaged_at = datetime.utcnow()

                    # Re-score
                    new_score = self.score_lead(lead)
                    lead.qualification_score = new_score

                    if new_score >= settings.auto_qualify_threshold:
                        lead.status = "qualified"
                        results["status_changes"].append({
                            "lead_id": str(lead.id),
                            "new_status": "qualified",
                            "score": new_score
                        })

                    db.commit()
                    results["leads_updated"] += 1

                results["replies_processed"] += 1

            except Exception as e:
                logger.error(f"Error processing reply {reply['reply_id']}: {e}")

        db.close()
        return results

    async def run_daily_qualification(self) -> Dict[str, Any]:
        """Run daily lead re-scoring and pipeline management."""
        db = get_db_session()
        results = {
            "leads_scored": 0,
            "promoted_to_qualified": 0,
            "marked_stale": 0
        }

        # Re-score all active leads
        active_leads = db.query(Lead).filter(
            Lead.status.in_(["new", "contacted"])
        ).all()

        for lead in active_leads:
            new_score = self.score_lead(lead)

            if new_score >= settings.auto_qualify_threshold and lead.status == "new":
                lead.status = "qualified"
                results["promoted_to_qualified"] += 1

            # Mark stale if no activity for 14 days
            if lead.last_engaged_at:
                days_since = (datetime.utcnow() - lead.last_engaged_at).days
                if days_since > 14 and lead.status == "new":
                    lead.status = "dead"
                    results["marked_stale"] += 1

            lead.qualification_score = new_score
            results["leads_scored"] += 1

        try:
            db.commit()
            return results
        finally:
            db.close()


qualifier_agent_instance = None


def get_qualifier_agent() -> QualifierAgent:
    global qualifier_agent_instance
    if qualifier_agent_instance is None:
        qualifier_agent_instance = QualifierAgent()
    return qualifier_agent_instance
