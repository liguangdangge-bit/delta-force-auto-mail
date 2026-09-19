"""Mail-storage automation integrated into the main application."""

from .models import MailWorkflowOutcome, MailWorkflowResult
from .service import MailProfile, MailStorageWorkflow

__all__ = [
    "MailProfile",
    "MailStorageWorkflow",
    "MailWorkflowOutcome",
    "MailWorkflowResult",
]
