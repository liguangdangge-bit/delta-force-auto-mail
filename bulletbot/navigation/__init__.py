"""Shared page semantics for trading and mail-storage workflows."""

from .actions import (
    MappedNavigationActionExecutor,
    NavigationActionError,
    NavigationActionExecutor,
)
from .catalog import PAGE_CATALOG, format_page, page_descriptor, page_label
from .diagnostics import NavigationJournal
from .graph import (
    DEFAULT_NAVIGATION_EDGES,
    DEFAULT_NAVIGATION_GRAPH,
    NavigationGraph,
    NavigationGraphError,
)
from .models import (
    BusinessAction,
    GamePageId,
    NavigationAction,
    NavigationBudgets,
    NavigationEdge,
    NavigationIntent,
    NavigationResult,
    NavigationRisk,
    NavigationStatus,
    NavigationTraceEntry,
    PageDescriptor,
    PageKind,
    PageSurface,
    RebindStatus,
    UnifiedPageObservation,
    UnknownPageRecoveryPolicy,
    WorkflowKind,
    WindowRebindResult,
)
from .navigator import UnifiedPageNavigator
from .recognition import PageStabilityTracker, resolve_page_observations

__all__ = [
    "BusinessAction",
    "DEFAULT_NAVIGATION_EDGES",
    "DEFAULT_NAVIGATION_GRAPH",
    "GamePageId",
    "MappedNavigationActionExecutor",
    "NavigationAction",
    "NavigationActionError",
    "NavigationActionExecutor",
    "NavigationBudgets",
    "NavigationEdge",
    "NavigationGraph",
    "NavigationGraphError",
    "NavigationIntent",
    "NavigationJournal",
    "NavigationResult",
    "NavigationRisk",
    "NavigationStatus",
    "NavigationTraceEntry",
    "PAGE_CATALOG",
    "PageDescriptor",
    "PageKind",
    "PageStabilityTracker",
    "PageSurface",
    "RebindStatus",
    "UnifiedPageNavigator",
    "UnifiedPageObservation",
    "UnknownPageRecoveryPolicy",
    "WorkflowKind",
    "WindowRebindResult",
    "format_page",
    "page_descriptor",
    "page_label",
    "resolve_page_observations",
]
