"""Deterministic compliance checks run before any outbound Reddit action."""
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class SafetyCheck:
    passed: bool
    rule_name: str
    severity: str
    message: str
    action: str


class SafetyGuardrails:
    HAS_ATTORNEY_PATTERNS = [
        r"\bmy attorney\b", r"\bmy lawyer\b", r"\brepresented by\b",
        r"\blegal counsel\b", r"\bhired (?:a|an) (?:lawyer|attorney)\b",
        r"\bretained\b.*\b(?:lawyer|attorney)\b", r"\bmy firm\b", r"\bmy legal team\b",
    ]
    LEGAL_ADVICE_PATTERNS = [
        r"\bshould i sue\b", r"\bcan i sue\b", r"\bdo i have a case\b",
        r"\bwhat are my rights\b", r"\bwhat should i do legally\b",
        r"\bis this illegal\b", r"\bcan they do this legally\b",
    ]
    SPAM_PATTERNS = [
        r"\bclick here\b", r"\bdm me\b", r"\bmessage me\b",
        r"\bfree consultation\b.*\bnow\b", r"\blimited time\b", r"\bact now\b",
        r"\bguaranteed\b.*\bwin\b", r"\bmillions\b",
    ]
    BLOCKED_SUBREDDITS = {
        "suicidewatch", "depression", "addiction", "domesticviolence", "rape",
        "ptsd", "grief", "bereavement", "mentalhealth", "personalfinance",
    }
    REQUIRED_DISCLAIMERS = ("not a law firm", "not legal advice", "general information")
    ALLOWED_HOSTS = {"caseclosedfl.com", "www.caseclosedfl.com", "reddit.com", "www.reddit.com"}
    INVISIBLE_CONTROL_CHARS = re.compile(r"[\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180e\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u206f\ufeff]")
    CONFUSABLE_CHARS = set("аɑеɇіɪоουµѕӏɫт")

    def __init__(self):
        self.settings = get_settings()
        self._daily_blocks = 0
        self._daily_flags = 0
        self._last_reset = datetime.now(timezone.utc).date()

    def _check_reset(self):
        today = datetime.now(timezone.utc).date()
        if today != self._last_reset:
            self._daily_blocks = self._daily_flags = 0
            self._last_reset = today

    def check_post_eligibility(self, post_title: str, post_body: str, subreddit: str,
                               author_info: Optional[Dict] = None) -> List[SafetyCheck]:
        self._check_reset()
        text = f"{post_title} {post_body}".lower()
        checks: List[SafetyCheck] = []
        blocked = subreddit.lower() in self.BLOCKED_SUBREDDITS
        checks.append(SafetyCheck(not blocked, "subreddit_allowed", "critical" if blocked else "info",
                                  "Protected community" if blocked else "Subreddit is allowed", "block" if blocked else "allow"))
        if blocked:
            self._daily_blocks += 1
        has_attorney = any(re.search(pattern, text, re.I) for pattern in self.HAS_ATTORNEY_PATTERNS)
        checks.append(SafetyCheck(not has_attorney, "no_existing_attorney", "critical",
                                  "Existing representation detected" if has_attorney else "No existing representation detected",
                                  "block" if has_attorney else "allow"))
        if has_attorney:
            self._daily_blocks += 1
        if author_info and "account_age_days" in author_info:
            too_new = int(author_info["account_age_days"]) < self.settings.min_account_age_days
            checks.append(SafetyCheck(not too_new, "account_age_ok", "warning",
                                      "Account is below minimum age" if too_new else "Account age is acceptable",
                                      "flag" if too_new else "allow"))
            if too_new:
                self._daily_flags += 1
        seeks_advice = any(re.search(pattern, text, re.I) for pattern in self.LEGAL_ADVICE_PATTERNS)
        checks.append(SafetyCheck(not seeks_advice, "no_legal_advice_request", "critical",
                                  "Specific legal advice requested" if seeks_advice else "No specific legal advice requested",
                                  "block" if seeks_advice else "allow"))
        if seeks_advice:
            self._daily_blocks += 1
        florida = re.search(r"\b(?:florida|fl|miami|orlando|tampa|jacksonville|broward|dade|pinellas)\b", text, re.I)
        checks.append(SafetyCheck(True, "florida_relevance", "info" if florida else "warning",
                                  "Florida location detected" if florida else "No clear Florida location indicator", "allow" if florida else "flag"))
        if not florida:
            self._daily_flags += 1
        return checks

    def check_response_compliance(self, response_text: str) -> List[SafetyCheck]:
        self._check_reset()
        text = response_text.lower()
        checks: List[SafetyCheck] = []
        invisible_count = len(self.INVISIBLE_CONTROL_CHARS.findall(response_text))
        confusable_count = sum(char in self.CONFUSABLE_CHARS for char in response_text)
        obfuscated = invisible_count > 0 or confusable_count > 0
        checks.append(SafetyCheck(
            not obfuscated,
            "no_obfuscated_unicode",
            "critical",
            "Invisible or confusable Unicode characters detected" if obfuscated else "No obfuscated Unicode detected",
            "block" if obfuscated else "allow",
        ))
        if obfuscated:
            self._daily_blocks += 1
        missing = [phrase for phrase in self.REQUIRED_DISCLAIMERS if phrase not in text]
        checks.append(SafetyCheck(not missing, "has_disclaimer", "critical",
                                  f"Missing disclosure phrases: {', '.join(missing)}" if missing else "Complete disclosure present",
                                  "block" if missing else "allow"))
        if missing:
            self._daily_blocks += 1
        advice = any(re.search(pattern, text, re.I) for pattern in [
            r"\byou should\b.*\b(?:sue|file|claim)\b", r"\byou have a case\b",
            r"\byou will win\b", r"\byou are entitled to\b.*\$", r"\byour case is worth\b",
        ])
        checks.append(SafetyCheck(not advice, "no_legal_advice", "critical",
                                  "Specific legal advice detected" if advice else "No specific legal advice detected",
                                  "block" if advice else "allow"))
        spam = any(re.search(pattern, text, re.I) for pattern in self.SPAM_PATTERNS)
        checks.append(SafetyCheck(not spam, "no_spam", "critical",
                                  "Spam language detected" if spam else "No spam language detected",
                                  "block" if spam else "allow"))
        if advice or spam:
            self._daily_blocks += 1
        too_long = len(response_text.split()) > 200
        checks.append(SafetyCheck(not too_long, "response_length_ok", "warning",
                                  "Response exceeds 200 words" if too_long else "Response length acceptable",
                                  "flag" if too_long else "allow"))
        if too_long:
            self._daily_flags += 1
        invalid_urls = []
        for raw in re.findall(r"https?://[^\s)\]]+", response_text):
            host = (urlparse(raw.rstrip(".,")).hostname or "").lower()
            if host not in self.ALLOWED_HOSTS:
                invalid_urls.append(raw)
        checks.append(SafetyCheck(not invalid_urls, "urls_allowed", "critical",
                                  f"Unauthorized URL: {invalid_urls[0]}" if invalid_urls else "All URLs are authorized",
                                  "block" if invalid_urls else "allow"))
        if invalid_urls:
            self._daily_blocks += 1
        return checks

    @staticmethod
    def can_proceed(checks: List[SafetyCheck]) -> Tuple[bool, List[str]]:
        blocking = [check for check in checks if not check.passed and check.action == "block"]
        return not blocking, [f"{check.rule_name}: {check.message}" for check in blocking]

    def get_stats(self) -> Dict[str, Any]:
        self._check_reset()
        return {"daily_blocks": self._daily_blocks, "daily_flags": self._daily_flags,
                "blocked_subreddits": sorted(self.BLOCKED_SUBREDDITS),
                "min_account_age_days": self.settings.min_account_age_days,
                "florida_bar_compliant": self.settings.florida_bar_compliant}


_guardrails: Optional[SafetyGuardrails] = None


def get_guardrails() -> SafetyGuardrails:
    global _guardrails
    if _guardrails is None:
        _guardrails = SafetyGuardrails()
    return _guardrails