"""
CaseClosedFL Reddit Agent - Safety & Compliance Guardrails
Realistic safety parameters for Reddit ToS compliance, Florida bar rules,
and ethical lead generation. No AI-bs safety theater.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class SafetyCheck:
    """Result of a safety/compliance check."""
    passed: bool
    rule_name: str
    severity: str  # info, warning, critical
    message: str
    action: str  # allow, flag, block


class SafetyGuardrails:
    """
    Production safety layer. These are REAL guardrails, not theater.

    Covers:
    1. Reddit ToS compliance (no spam, no ban evasion)
    2. Florida Bar advertising rules (no legal advice, no solicitation)
    3. CaseClosedFL business rules (no existing attorney clients)
    4. Content quality (no low-effort responses)
    """

    # Patterns that indicate user already has an attorney
    HAS_ATTORNEY_PATTERNS = [
        r"my attorney", r"my lawyer", r"represented by",
        r"legal counsel", r"hired a lawyer", r"hired an attorney",
        r"retained.*(lawyer|attorney)", r"my firm",
        r"speaking with my attorney", r"my legal team"
    ]

    # Patterns that indicate this is a legal advice request (not for us)
    LEGAL_ADVICE_PATTERNS = [
        r"should I sue", r"can I sue", r"do I have a case",
        r"what are my rights", r"what should I do legally",
        r"is this illegal", r"can they do this legally"
    ]

    # Spam / low quality indicators
    SPAM_PATTERNS = [
        r"click here", r"DM me", r"message me",
        r"free consultation.*now", r"limited time",
        r"act now", r"guaranteed.*win", r"millions"
    ]

    # Subreddits we should NEVER post in (protected communities)
    BLOCKED_SUBREDDITS = {
        "suicidewatch", "depression", "addiction", "domesticviolence",
        "rape", "ptsd", "grief", "bereavement", "mentalhealth",
        "personalfinance"  # Only monitor, never solicit
    }

    # Required disclaimer phrases
    REQUIRED_DISCLAIMERS = [
        "not a law firm",
        "not legal advice",
        "general information"
    ]

    def __init__(self):
        self._daily_blocks = 0
        self._daily_flags = 0
        self._last_reset = datetime.utcnow().date()

    def _check_reset(self):
        """Reset daily counters at midnight."""
        today = datetime.utcnow().date()
        if today != self._last_reset:
            self._daily_blocks = 0
            self._daily_flags = 0
            self._last_reset = today

    def check_post_eligibility(
        self,
        post_title: str,
        post_body: str,
        subreddit: str,
        author_info: Optional[Dict] = None
    ) -> List[SafetyCheck]:
        """
        Run full eligibility check on a post before any engagement.
        Returns list of checks - ALL must pass for engagement.
        """
        self._check_reset()
        checks = []
        text = f"{post_title} {post_body}".lower()

        # 1. Subreddit blocklist
        if subreddit.lower() in self.BLOCKED_SUBREDDITS:
            checks.append(SafetyCheck(
                passed=False,
                rule_name="blocked_subreddit",
                severity="critical",
                message=f"r/{subreddit} is in the protected community list",
                action="block"
            ))
            self._daily_blocks += 1
        else:
            checks.append(SafetyCheck(
                passed=True,
                rule_name="subreddit_allowed",
                severity="info",
                message=f"r/{subreddit} is allowed",
                action="allow"
            ))

        # 2. Check if user already has an attorney
        has_attorney = any(re.search(p, text, re.I) for p in self.HAS_ATTORNEY_PATTERNS)
        if has_attorney:
            checks.append(SafetyCheck(
                passed=False,
                rule_name="existing_attorney",
                severity="critical",
                message="User appears to already have legal representation",
                action="block"
            ))
            self._daily_blocks += 1
        else:
            checks.append(SafetyCheck(
                passed=True,
                rule_name="no_existing_attorney",
                severity="info",
                message="No indication of existing attorney",
                action="allow"
            ))

        # 3. Check account age (if info available)
        if author_info and "account_age_days" in author_info:
            age = author_info["account_age_days"]
            if age < settings.min_account_age_days:
                checks.append(SafetyCheck(
                    passed=False,
                    rule_name="account_too_new",
                    severity="warning",
                    message=f"Account age ({age}d) below minimum ({settings.min_account_age_days}d)",
                    action="flag"
                ))
                self._daily_flags += 1
            else:
                checks.append(SafetyCheck(
                    passed=True,
                    rule_name="account_age_ok",
                    severity="info",
                    message=f"Account age acceptable ({age}d)",
                    action="allow"
                ))

        # 4. Check for legal advice requests (we do not answer these)
        seeks_legal_advice = any(re.search(p, text, re.I) for p in self.LEGAL_ADVICE_PATTERNS)
        if seeks_legal_advice:
            checks.append(SafetyCheck(
                passed=False,
                rule_name="seeks_legal_advice",
                severity="critical",
                message="Post is requesting specific legal advice - we cannot answer",
                action="block"
            ))
            self._daily_blocks += 1
        else:
            checks.append(SafetyCheck(
                passed=True,
                rule_name="no_legal_advice_request",
                severity="info",
                message="Post does not request specific legal advice",
                action="allow"
            ))

        # 5. Florida relevance check
        florida_indicators = ["florida", "fl", "miami", "orlando", "tampa", "jacksonville", 
                           "fort lauderdale", "west palm beach", "broward", "dade", "orange county",
                           "hillsborough", "pinellas", "palm beach county"]
        is_florida_relevant = any(ind in text for ind in florida_indicators)

        if not is_florida_relevant:
            # Not a hard block, but flag as low priority
            checks.append(SafetyCheck(
                passed=True,
                rule_name="florida_relevance",
                severity="warning",
                message="No clear Florida location indicator - will score lower",
                action="flag"
            ))
            self._daily_flags += 1
        else:
            checks.append(SafetyCheck(
                passed=True,
                rule_name="florida_relevant",
                severity="info",
                message="Florida location detected",
                action="allow"
            ))

        return checks

    def check_response_compliance(self, response_text: str) -> List[SafetyCheck]:
        """
        Verify a crafted response meets all compliance requirements
        BEFORE sending to Reddit.
        """
        self._check_reset()
        checks = []
        text_lower = response_text.lower()

        # 1. Required disclaimers
        has_disclaimer = any(phrase in text_lower for phrase in self.REQUIRED_DISCLAIMERS)
        if not has_disclaimer:
            checks.append(SafetyCheck(
                passed=False,
                rule_name="missing_disclaimer",
                severity="critical",
                message="Response missing required legal disclaimer",
                action="block"
            ))
            self._daily_blocks += 1
        else:
            checks.append(SafetyCheck(
                passed=True,
                rule_name="has_disclaimer",
                severity="info",
                message="Required disclaimer present",
                action="allow"
            ))

        # 2. No legal advice claims
        legal_advice_claims = [
            r"you should.*(sue|file|claim)",
            r"you have a case", r"you will win",
            r"you are entitled to.*\$",
            r"your case is worth"
        ]
        gives_advice = any(re.search(p, text_lower) for p in legal_advice_claims)
        if gives_advice:
            checks.append(SafetyCheck(
                passed=False,
                rule_name="gives_legal_advice",
                severity="critical",
                message="Response contains specific legal advice or predictions",
                action="block"
            ))
            self._daily_blocks += 1
        else:
            checks.append(SafetyCheck(
                passed=True,
                rule_name="no_legal_advice",
                severity="info",
                message="No specific legal advice detected",
                action="allow"
            ))

        # 3. No spam patterns
        spam_detected = any(re.search(p, text_lower) for p in self.SPAM_PATTERNS)
        if spam_detected:
            checks.append(SafetyCheck(
                passed=False,
                rule_name="spam_detected",
                severity="critical",
                message="Response contains spam-like language",
                action="block"
            ))
            self._daily_blocks += 1
        else:
            checks.append(SafetyCheck(
                passed=True,
                rule_name="no_spam",
                severity="info",
                message="No spam patterns detected",
                action="allow"
            ))

        # 4. Response length check
        word_count = len(response_text.split())
        if word_count > 200:
            checks.append(SafetyCheck(
                passed=False,
                rule_name="response_too_long",
                severity="warning",
                message=f"Response too long ({word_count} words, max 200)",
                action="flag"
            ))
            self._daily_flags += 1
        else:
            checks.append(SafetyCheck(
                passed=True,
                rule_name="response_length_ok",
                severity="info",
                message=f"Response length acceptable ({word_count} words)",
                action="allow"
            ))

        # 5. URL check - only allow caseclosedfl.com
        urls = re.findall(r'https?://[^\s]+', response_text)
        bad_urls = [u for u in urls if "caseclosedfl.com" not in u and "reddit.com" not in u]
        if bad_urls:
            checks.append(SafetyCheck(
                passed=False,
                rule_name="unauthorized_url",
                severity="critical",
                message=f"Response contains unauthorized URL: {bad_urls[0]}",
                action="block"
            ))
            self._daily_blocks += 1
        else:
            checks.append(SafetyCheck(
                passed=True,
                rule_name="urls_allowed",
                severity="info",
                message="All URLs are authorized",
                action="allow"
            ))

        return checks

    def can_proceed(self, checks: List[SafetyCheck]) -> Tuple[bool, List[str]]:
        """
        Determine if engagement can proceed based on checks.
        Returns (can_proceed, list_of_blocking_reasons).
        """
        blocking = [c for c in checks if c.action == "block" and not c.passed]
        if blocking:
            return False, [f"{c.rule_name}: {c.message}" for c in blocking]
        return True, []

    def get_stats(self) -> Dict[str, Any]:
        self._check_reset()
        return {
            "daily_blocks": self._daily_blocks,
            "daily_flags": self._daily_flags,
            "blocked_subreddits": list(self.BLOCKED_SUBREDDITS),
            "min_account_age_days": settings.min_account_age_days,
            "florida_bar_compliant": settings.florida_bar_compliant
        }


# Singleton
_guardrails: Optional[SafetyGuardrails] = None


def get_guardrails() -> SafetyGuardrails:
    global _guardrails
    if _guardrails is None:
        _guardrails = SafetyGuardrails()
    return _guardrails
