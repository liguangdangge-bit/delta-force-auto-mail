"""Coordinate trading and mail-storage workflows in one process."""

from .checkpoint import SessionCheckpointStore
from .configuration import (
    MailIntegrationSettings,
    MailIntegrationSettingsStore,
    load_mail_profile,
)
from .coordinator import OrchestrationError, TradingMailOrchestrator
from .models import (
    MailTriggerKind,
    OrchestrationState,
    PurchaseCountUpdate,
    SessionCheckpoint,
)
from .runtime import MailIntegrationRuntime, MailIntegrationSnapshot

__all__ = [
    "MailTriggerKind",
    "MailIntegrationSettings",
    "MailIntegrationSettingsStore",
    "MailIntegrationRuntime",
    "MailIntegrationSnapshot",
    "OrchestrationError",
    "OrchestrationState",
    "PurchaseCountUpdate",
    "SessionCheckpoint",
    "SessionCheckpointStore",
    "TradingMailOrchestrator",
    "load_mail_profile",
]
