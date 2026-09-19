from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GamePageId(str, Enum):
    UNKNOWN = "unknown"
    INVALID_FRAME = "invalid_frame"
    TRANSITION = "transition"

    FAVORITES = "favorites"
    MARKET_DETAIL = "market_detail"
    WAREHOUSE = "warehouse"
    WAREHOUSE_SELL_DIALOG = "warehouse_sell_dialog"
    MARKET_SELL = "market_sell"
    MARKET_CLAIM_COMPLETE = "market_claim_complete"
    LISTING_EDITOR = "listing_editor"
    MARKET_OTHER = "market_other"

    GAME_HOME_PREPARE = "game_home_prepare"
    GAME_HOME_READY = "game_home_ready"
    MATCHING = "matching"
    MAP_OVERVIEW = "map_overview"
    MAP_SELECTION = "map_selection"
    MAP_OTHER = "map_other"
    MAP_LONGBOW_START = "map_longbow_start"
    MAP_LONGBOW_DOWNLOAD = "map_longbow_download"
    LOADOUT = "loadout"
    AGENT_SELECT = "agent_select"
    ENTRY_WARNING = "entry_warning"
    GAME_STARTUP = "game_startup"
    MISSING_RESOURCE = "missing_resource"
    INITIAL_MODE_SELECTION = "initial_mode_selection"
    GAME_TRANSITION = "game_transition"
    ACTIVITY_REMINDER = "activity_reminder"
    MAP_ZERO_DAM_START = "map_zero_dam_start"
    GENERIC_CONFIRM = "generic_confirm"
    LOADOUT_PURCHASE_CONFIRM = "loadout_purchase_confirm"
    LOADOUT_PRICE_CHANGE = "loadout_price_change"
    LOADOUT_SCHEMES = "loadout_schemes"
    MAIL_SCHEME_SELECTED = "mail_scheme_selected"
    LOADOUT_SAVE_DIALOG = "loadout_save_dialog"
    LOADOUT_SAVE_OVERWRITE = "loadout_save_overwrite"
    MODE_MENU = "mode_menu"
    RECONNECT_PROMPT = "reconnect_prompt"
    ABANDON_PROMPT = "abandon_prompt"
    GAME_LOADING = "game_loading"
    SETTLEMENT = "settlement"
    WARFARE_HOME = "warfare_home"
    MAIL_INBOX = "mail_inbox"
    MAIL_CLAIM_COMPLETE = "mail_claim_complete"

    LAUNCHER_HOME = "launcher_home"
    LAUNCHER_RESOURCES = "launcher_resources"
    LAUNCHER_DELETE_CONFIRM = "launcher_delete_confirm"


class PageSurface(str, Enum):
    GAME = "game"
    LAUNCHER = "launcher"
    LAUNCHER_SETTINGS = "launcher_settings"


class PageKind(str, Enum):
    STABLE = "stable"
    MODAL = "modal"
    TRANSITION = "transition"
    META = "meta"


class WorkflowKind(str, Enum):
    TRADING = "trading"
    MAIL_STORAGE = "mail_storage"


class BusinessAction(str, Enum):
    PURCHASE = "purchase"
    CONFIRM_LISTING = "confirm_listing"
    MOVE_PAK = "move_pak"
    DELETE_PAK = "delete_pak"
    CONFIRM_ABANDON = "confirm_abandon"
    CLAIM_MAIL = "claim_mail"
    APPLY_LOADOUT_SCHEME = "apply_loadout_scheme"
    BUY_LOADOUT_SCHEME = "buy_loadout_scheme"


class NavigationAction(str, Enum):
    ESCAPE = "escape"
    SPACE = "space"
    TAB = "tab"
    OPEN_GAME_HOME = "open_game_home"
    OPEN_WAREHOUSE = "open_warehouse"
    OPEN_MARKET = "open_market"
    OPEN_MARKET_BUY = "open_market_buy"
    OPEN_MARKET_SELL = "open_market_sell"
    OPEN_GAME_MAP = "open_game_map"
    EXPAND_MAP = "expand_map"
    SELECT_LONGBOW = "select_longbow"


class NavigationRisk(str, Enum):
    SAFE = "safe"
    CONTEXTUAL = "contextual"


class NavigationStatus(str, Enum):
    ARRIVED = "arrived"
    RECOVERED = "recovered"
    RETRYABLE = "retryable"
    MANUAL_BINDING_REQUIRED = "manual_binding_required"
    NO_SAFE_PATH = "no_safe_path"
    FAILED = "failed"
    STOPPED = "stopped"


class RebindStatus(str, Enum):
    REBOUND = "rebound"
    MANUAL_REQUIRED = "manual_required"
    NOT_FOUND = "not_found"
    FAILED = "failed"


@dataclass(frozen=True)
class PageDescriptor:
    page_id: GamePageId
    chinese_name: str
    default_surface: PageSurface
    kind: PageKind = PageKind.STABLE


@dataclass(frozen=True)
class UnifiedPageObservation:
    page_id: GamePageId
    confidence: float
    chinese_name: str
    surface: PageSurface
    evidence: tuple[str, ...] = ()
    stable: bool = False
    captured_at: float = 0.0
    window_generation: int = 0
    screenshot_path: str | None = None


@dataclass(frozen=True)
class NavigationIntent:
    workflow: WorkflowKind
    target_page: GamePageId
    reason: str
    allowed_business_actions: frozenset[BusinessAction] = frozenset()
    acceptable_pages: frozenset[GamePageId] = frozenset()

    @property
    def target_pages(self) -> frozenset[GamePageId]:
        return frozenset({self.target_page, *self.acceptable_pages})


@dataclass(frozen=True)
class NavigationEdge:
    source: GamePageId
    target: GamePageId
    action: NavigationAction
    expected_pages: frozenset[GamePageId]
    repeatable: bool = True
    timeout_seconds: float = 5.0
    settle_seconds: float = 0.35
    risk: NavigationRisk = NavigationRisk.SAFE


@dataclass(frozen=True)
class NavigationBudgets:
    recognition_retries: int = 1
    action_retries: int = 1
    window_rebinds: int = 1
    max_steps: int = 12
    action_retry_delay_seconds: float = 0.0
    delayed_retry_actions: frozenset[NavigationAction] = frozenset()

    def __post_init__(self) -> None:
        values = (
            self.recognition_retries,
            self.action_retries,
            self.window_rebinds,
            self.max_steps,
        )
        if any(value < 0 for value in values) or self.max_steps < 1:
            raise ValueError("导航重试预算必须为非负数，最大步数必须至少为 1")
        if self.action_retry_delay_seconds < 0:
            raise ValueError("导航动作重试等待时间不能为负数")


@dataclass(frozen=True)
class UnknownPageRecoveryPolicy:
    wait_seconds: float = 3.0
    action_wait_seconds: float = 3.0
    actions: tuple[NavigationAction, ...] = (
        NavigationAction.ESCAPE,
        NavigationAction.SPACE,
        NavigationAction.TAB,
    )
    max_rounds: int = 3

    def __post_init__(self) -> None:
        if self.wait_seconds < 0 or self.action_wait_seconds < 0:
            raise ValueError("未知页面恢复等待时间不能为负数")
        if self.max_rounds < 1:
            raise ValueError("未知页面恢复轮数必须至少为 1")
        allowed = {NavigationAction.ESCAPE, NavigationAction.SPACE, NavigationAction.TAB}
        if any(action not in allowed for action in self.actions):
            raise ValueError("未知页面只能使用 Esc 或 Space 进行有限恢复")


@dataclass(frozen=True)
class WindowRebindResult:
    status: RebindStatus
    generation: int = 0
    message: str = ""


@dataclass(frozen=True)
class NavigationTraceEntry:
    sequence: int
    event: str
    message: str
    page_id: GamePageId | None = None
    action: NavigationAction | None = None
    attempt: int = 0
    window_generation: int = 0


@dataclass(frozen=True)
class NavigationResult:
    status: NavigationStatus
    target_page: GamePageId
    current_page: GamePageId | None
    message: str
    history: tuple[NavigationTraceEntry, ...] = ()
    recognition_retries_used: int = 0
    action_retries_used: int = 0
    window_rebinds_used: int = 0
    failure_screenshot_path: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status in {
            NavigationStatus.ARRIVED,
            NavigationStatus.RECOVERED,
        }
