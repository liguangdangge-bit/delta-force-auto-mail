from __future__ import annotations

from types import MappingProxyType

from .models import GamePageId, PageDescriptor, PageKind, PageSurface


def _page(
    page_id: GamePageId,
    chinese_name: str,
    *,
    surface: PageSurface = PageSurface.GAME,
    kind: PageKind = PageKind.STABLE,
) -> PageDescriptor:
    return PageDescriptor(page_id, chinese_name, surface, kind)


_PAGE_DESCRIPTORS = (
    _page(GamePageId.UNKNOWN, "未识别页面", kind=PageKind.META),
    _page(GamePageId.INVALID_FRAME, "画面无效或黑屏", kind=PageKind.META),
    _page(GamePageId.TRANSITION, "页面切换中", kind=PageKind.TRANSITION),
    _page(GamePageId.FAVORITES, "交易行收藏页"),
    _page(GamePageId.MARKET_DETAIL, "商品详情页"),
    _page(GamePageId.WAREHOUSE, "仓库"),
    _page(
        GamePageId.WAREHOUSE_SELL_DIALOG,
        "仓库出售选择页",
        kind=PageKind.MODAL,
    ),
    _page(GamePageId.MARKET_SELL, "交易行出售页"),
    _page(
        GamePageId.MARKET_CLAIM_COMPLETE,
        "交易行货款领取完成",
        kind=PageKind.MODAL,
    ),
    _page(GamePageId.LISTING_EDITOR, "上架编辑器"),
    _page(GamePageId.MARKET_OTHER, "交易行其他页面"),
    _page(GamePageId.GAME_HOME_PREPARE, "游戏主界面（行前备战）"),
    _page(GamePageId.GAME_HOME_READY, "游戏主界面（配装出发）"),
    _page(GamePageId.MATCHING, "正在匹配", kind=PageKind.TRANSITION),
    _page(GamePageId.MAP_OVERVIEW, "地图选择总览"),
    _page(GamePageId.MAP_SELECTION, "地图全览（选择地图）"),
    _page(GamePageId.MAP_OTHER, "其他地图详情"),
    _page(GamePageId.MAP_LONGBOW_START, "长弓溪谷（开始行动）"),
    _page(GamePageId.MAP_LONGBOW_DOWNLOAD, "长弓溪谷（需要下载）"),
    _page(GamePageId.LOADOUT, "配装界面"),
    _page(GamePageId.AGENT_SELECT, "干员选择界面"),
    _page(GamePageId.ENTRY_WARNING, "入局提醒明细", kind=PageKind.MODAL),
    _page(GamePageId.GAME_STARTUP, "游戏启动画面", kind=PageKind.TRANSITION),
    _page(GamePageId.MISSING_RESOURCE, "资源缺失提示", kind=PageKind.MODAL),
    _page(GamePageId.INITIAL_MODE_SELECTION, "初始模式选择"),
    _page(GamePageId.GAME_TRANSITION, "游戏过渡（按 Tab）", kind=PageKind.TRANSITION),
    _page(GamePageId.ACTIVITY_REMINDER, "活动提醒（按空格）", kind=PageKind.MODAL),
    _page(GamePageId.MAP_ZERO_DAM_START, "零号大坝（开始行动）"),
    _page(GamePageId.GENERIC_CONFIRM, "系统确认提示", kind=PageKind.MODAL),
    _page(GamePageId.LOADOUT_PURCHASE_CONFIRM, "确认实现方案", kind=PageKind.MODAL),
    _page(GamePageId.LOADOUT_PRICE_CHANGE, "价格变动提醒", kind=PageKind.MODAL),
    _page(GamePageId.LOADOUT_SCHEMES, "配装方案列表"),
    _page(GamePageId.MAIL_SCHEME_SELECTED, "卡邮件方案（可使用）"),
    _page(GamePageId.LOADOUT_SAVE_DIALOG, "保存配装方案", kind=PageKind.MODAL),
    _page(GamePageId.LOADOUT_SAVE_OVERWRITE, "覆盖配装方案", kind=PageKind.MODAL),
    _page(GamePageId.MODE_MENU, "模式选择菜单"),
    _page(GamePageId.RECONNECT_PROMPT, "取消重连提示", kind=PageKind.MODAL),
    _page(GamePageId.ABANDON_PROMPT, "放弃对局确认", kind=PageKind.MODAL),
    _page(GamePageId.GAME_LOADING, "游戏加载过渡", kind=PageKind.TRANSITION),
    _page(GamePageId.SETTLEMENT, "游戏结算界面"),
    _page(GamePageId.WARFARE_HOME, "全面战场主界面"),
    _page(GamePageId.MAIL_INBOX, "邮件附件界面"),
    _page(GamePageId.MAIL_CLAIM_COMPLETE, "邮件领取完成"),
    _page(
        GamePageId.LAUNCHER_HOME,
        "启动器首页",
        surface=PageSurface.LAUNCHER,
    ),
    _page(
        GamePageId.LAUNCHER_RESOURCES,
        "启动器资源管理",
        surface=PageSurface.LAUNCHER_SETTINGS,
    ),
    _page(
        GamePageId.LAUNCHER_DELETE_CONFIRM,
        "启动器删除资源确认",
        surface=PageSurface.LAUNCHER_SETTINGS,
        kind=PageKind.MODAL,
    ),
)

PAGE_CATALOG = MappingProxyType(
    {descriptor.page_id: descriptor for descriptor in _PAGE_DESCRIPTORS}
)

if set(PAGE_CATALOG) != set(GamePageId):
    missing = set(GamePageId) - set(PAGE_CATALOG)
    extra = set(PAGE_CATALOG) - set(GamePageId)
    raise RuntimeError(f"统一页面目录不完整：missing={missing}, extra={extra}")


def page_descriptor(page_id: GamePageId) -> PageDescriptor:
    return PAGE_CATALOG[page_id]


def page_label(page_id: GamePageId) -> str:
    return page_descriptor(page_id).chinese_name


def format_page(page_id: GamePageId) -> str:
    return f"{page_label(page_id)} ({page_id.value})"
