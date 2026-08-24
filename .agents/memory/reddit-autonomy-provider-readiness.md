---
name: Reddit autonomy provider readiness
description: Required provider capability rule for enabling autonomous Reddit outreach.
---

Autonomous Reddit outreach can be enabled only when the active provider exposes and can check both the public-comment and direct-message actions. The current Composio Reddit action catalog includes public comments but no direct-message action, so the control plane must remain OFF.

**Why:** A comments-only mode violates the product requirement that public comments and direct messages are one combined autonomous system. Reporting ON without a viable DM action would be misleading and could create unbounded partial behavior.

**How to apply:** Recheck the active provider capability catalog at enablement and before each dispatch. Treat missing, revoked, or unverifiable write capability for either channel as a fail-closed condition for both. Do not enable an alternate transport for autonomy unless it has an equivalent live authorization check.
