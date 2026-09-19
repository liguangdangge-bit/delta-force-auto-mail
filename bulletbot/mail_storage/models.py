from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from bulletbot.navigation.catalog import page_label as unified_page_label
from bulletbot.navigation.models import GamePageId


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def valid(self, minimum: tuple[int, int] = (100, 100)) -> bool:
        return self.width >= minimum[0] and self.height >= minimum[1]


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    class_name: str
    pid: int
    process_name: str
    process_path: str
    client_rect: Rect
    minimized: bool = False

    @property
    def label(self) -> str:
        title = self.title.strip() or "(无标题)"
        process = self.process_name or f"PID {self.pid}"
        return f"{title}  [{process}, HWND 0x{self.hwnd:X}]"


@dataclass
class WindowFingerprint:
    process_name: str = ""
    process_path: str = ""
    title: str = ""
    class_name: str = ""
    minimum_client_size: tuple[int, int] = (800, 500)

    @classmethod
    def from_window(cls, window: WindowInfo) -> "WindowFingerprint":
        return cls(
            process_name=window.process_name,
            process_path=window.process_path,
            title=window.title,
            class_name=window.class_name,
            minimum_client_size=(
                min(window.client_rect.width, 960),
                min(window.client_rect.height, 540),
            ),
        )

    def configured(self) -> bool:
        return bool(self.process_name or self.process_path or self.title or self.class_name)

    def score(self, window: WindowInfo) -> int:
        score = 0
        expected_path = self.process_path.casefold()
        actual_path = window.process_path.casefold()
        expected_process = self.process_name.casefold()
        actual_process = window.process_name.casefold()

        if expected_path and actual_path and expected_path == actual_path:
            score += 60
        if expected_process and actual_process and expected_process == actual_process:
            score += 35
        if self.title and window.title:
            if self.title == window.title:
                score += 25
            elif self.title.casefold() in window.title.casefold() or window.title.casefold() in self.title.casefold():
                score += 12
        if self.class_name and self.class_name == window.class_name:
            score += 15
        if window.client_rect.valid(self.minimum_client_size):
            score += 5
        if window.minimized:
            score -= 20
        return score

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["minimum_client_size"] = list(self.minimum_client_size)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "WindowFingerprint":
        if not value:
            return cls()
        minimum = value.get("minimum_client_size", [800, 500])
        return cls(
            process_name=str(value.get("process_name", "")),
            process_path=str(value.get("process_path", "")),
            title=str(value.get("title", "")),
            class_name=str(value.get("class_name", "")),
            minimum_client_size=(int(minimum[0]), int(minimum[1])),
        )


class PageType(StrEnum):
    UNKNOWN = "unknown"
    INVALID_FRAME = "invalid_frame"
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
    LAUNCHER_HOME = "launcher_home"
    LAUNCHER_RESOURCES = "launcher_resources"
    LAUNCHER_DELETE_CONFIRM = "launcher_delete_confirm"
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
    WAREHOUSE_SELL_DIALOG = "warehouse_sell_dialog"
    MODE_MENU = "mode_menu"
    RECONNECT_PROMPT = "reconnect_prompt"
    ABANDON_PROMPT = "abandon_prompt"
    GAME_LOADING = "game_loading"
    SETTLEMENT = "settlement"
    WARFARE_HOME = "warfare_home"
    MAIL_INBOX = "mail_inbox"
    MAIL_CLAIM_COMPLETE = "mail_claim_complete"


def page_type_label(page_type: PageType) -> str:
    return unified_page_label(GamePageId(page_type.value))


@dataclass(frozen=True)
class TemplateMatch:
    name: str
    score: float
    location: tuple[int, int]
    size: tuple[int, int]
    click_normalized: tuple[float, float] | None = None


@dataclass(frozen=True)
class OcrTextRegion:
    text: str
    confidence: float
    bounds: tuple[int, int, int, int]


@dataclass
class PageObservation:
    page_type: PageType
    confidence: float
    matches: dict[str, TemplateMatch] = field(default_factory=dict)
    ocr_texts: list[str] = field(default_factory=list)
    ocr_regions: list[OcrTextRegion] = field(default_factory=list)
    frame_mean: float = 0.0
    frame_std: float = 0.0
    timestamp: float = 0.0

    def summary(self) -> str:
        return f"{page_type_label(self.page_type)}；置信度 {self.confidence:.0%}"


class WorkflowState(StrEnum):
    IDLE = "idle"
    DEMONSTRATION = "demonstration"
    PRECHECK = "precheck"
    WAIT_GAME = "wait_game"
    OPEN_MAP = "open_map"
    SELECT_LONGBOW = "select_longbow"
    START_ACTION = "start_action"
    DEPART = "depart"
    WAIT_AGENT = "wait_agent"
    CLOSE_GAME = "close_game"
    WAIT_GAME_EXIT = "wait_game_exit"
    STAGE_PAK = "stage_pak"
    WAIT_LAUNCHER = "wait_launcher"
    REMOVE_RESOURCE = "remove_resource"
    CONFIRM_DELETE = "confirm_delete"
    START_GAME = "start_game"
    REBIND_GAME = "rebind_game"
    SELECT_FIRESTORM = "select_firestorm"
    RESTORE_PAK = "restore_pak"
    DISMISS_DIALOGS = "dismiss_dialogs"
    DOWNLOAD_MAP = "download_map"
    PREPARE_ZERO_DAM = "prepare_zero_dam"
    APPLY_LOADOUT_SCHEME = "apply_loadout_scheme"
    LOADOUT_PURCHASE_PREPARE = "loadout_purchase_prepare"
    LOADOUT_PURCHASE_SCAN = "loadout_purchase_scan"
    LOADOUT_PURCHASE_BUY = "loadout_purchase_buy"
    LOADOUT_PURCHASE_CLEAR = "loadout_purchase_clear"
    LOADOUT_WAREHOUSE_SELL = "loadout_warehouse_sell"
    LOADOUT_SELL_SLOT_WAIT = "loadout_sell_slot_wait"
    LOADOUT_SELL_UNLIST = "loadout_sell_unlist"
    AMMUNITION_SALE = "ammunition_sale"
    WAIT_DOWNLOAD = "wait_download"
    SWITCH_MODE = "switch_mode"
    CANCEL_RECONNECT = "cancel_reconnect"
    WAIT_SETTLEMENT = "wait_settlement"
    RETURN_FIRESTORM = "return_firestorm"
    COLLECT_MAIL = "collect_mail"
    COMPLETE = "complete"
    STOPPED = "stopped"
    ERROR = "error"


class MailWorkflowOutcome(StrEnum):
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


WORKFLOW_STATE_LABELS: dict[WorkflowState, str] = {
    WorkflowState.IDLE: "空闲/观察",
    WorkflowState.DEMONSTRATION: "手动演示采集",
    WorkflowState.PRECHECK: "启动检查",
    WorkflowState.WAIT_GAME: "等待游戏",
    WorkflowState.OPEN_MAP: "打开地图",
    WorkflowState.SELECT_LONGBOW: "选择长弓溪谷",
    WorkflowState.START_ACTION: "开始行动",
    WorkflowState.DEPART: "配装出发",
    WorkflowState.WAIT_AGENT: "等待干员选择",
    WorkflowState.CLOSE_GAME: "关闭游戏",
    WorkflowState.WAIT_GAME_EXIT: "等待游戏退出",
    WorkflowState.STAGE_PAK: "暂存 PAK",
    WorkflowState.WAIT_LAUNCHER: "等待启动器",
    WorkflowState.REMOVE_RESOURCE: "删除地图资源",
    WorkflowState.CONFIRM_DELETE: "确认删除地图资源",
    WorkflowState.START_GAME: "重新启动游戏",
    WorkflowState.REBIND_GAME: "重新绑定游戏窗口",
    WorkflowState.SELECT_FIRESTORM: "选择烽火地带",
    WorkflowState.RESTORE_PAK: "恢复 PAK",
    WorkflowState.DISMISS_DIALOGS: "关闭提示",
    WorkflowState.DOWNLOAD_MAP: "下载长弓溪谷",
    WorkflowState.PREPARE_ZERO_DAM: "零号大坝配装准备",
    WorkflowState.APPLY_LOADOUT_SCHEME: "应用卡邮件方案",
    WorkflowState.LOADOUT_PURCHASE_PREPARE: "准备配装买入",
    WorkflowState.LOADOUT_PURCHASE_SCAN: "轮询配装价格",
    WorkflowState.LOADOUT_PURCHASE_BUY: "购买配装方案",
    WorkflowState.LOADOUT_PURCHASE_CLEAR: "切换空白方案",
    WorkflowState.LOADOUT_WAREHOUSE_SELL: "交易行直售子弹",
    WorkflowState.LOADOUT_SELL_SLOT_WAIT: "等待空闲售位",
    WorkflowState.LOADOUT_SELL_UNLIST: "超时下架挂单",
    WorkflowState.AMMUNITION_SALE: "独立出售子弹",
    WorkflowState.WAIT_DOWNLOAD: "等待地图下载",
    WorkflowState.SWITCH_MODE: "切换全面战场",
    WorkflowState.CANCEL_RECONNECT: "取消重连",
    WorkflowState.WAIT_SETTLEMENT: "等待并跳过结算",
    WorkflowState.RETURN_FIRESTORM: "返回烽火地带",
    WorkflowState.COLLECT_MAIL: "领取邮件装备",
    WorkflowState.COMPLETE: "流程完成",
    WorkflowState.STOPPED: "已停止",
    WorkflowState.ERROR: "发生错误",
}


def workflow_state_label(state: WorkflowState) -> str:
    return WORKFLOW_STATE_LABELS.get(state, "未知状态")


@dataclass(frozen=True)
class WorkflowEvent:
    state: WorkflowState
    message: str
    level: str = "info"
    event_type: str = "message"
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MailWorkflowResult:
    outcome: MailWorkflowOutcome
    completed_rounds: int
    pak_restored: bool
    final_page: str | None
    message: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "completed_rounds": self.completed_rounds,
            "pak_restored": self.pak_restored,
            "final_page": self.final_page,
            "message": self.message,
            "error": self.error,
        }
