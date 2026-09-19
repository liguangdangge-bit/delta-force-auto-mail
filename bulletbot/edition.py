"""Presentation policy for packaged releases and source previews."""

import os
import sys


def is_agent_edition() -> bool:
    return getattr(sys, "_bulletbot_edition", "normal") == "agent"


def is_price_drop_demo_enabled() -> bool:
    # Frozen releases use their embedded runtime hook, never the build shell's
    # environment. Keep the environment switch for source previews only.
    enabled = getattr(sys, "_bulletbot_price_drop_demo_enabled", True)
    if getattr(sys, "frozen", False):
        return enabled
    return enabled and os.environ.get("BULLETBOT_DISABLE_PRICE_DROP_DEMO") != "1"


def configure_startup_edition() -> None:
    """Consume the preview flag before Qt imports; frozen builds keep their edition."""
    agent_requested = "-AgentEdition" in sys.argv[1:]
    if not getattr(sys, "frozen", False):
        sys._bulletbot_edition = "agent" if agent_requested else "normal"
    sys.argv[:] = [sys.argv[0], *(arg for arg in sys.argv[1:] if arg != "-AgentEdition")]
