from __future__ import annotations

from collections.abc import Callable

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .widgets import NoWheelComboBox


class RefreshingWindowComboBox(NoWheelComboBox):
    """Refresh native window choices immediately before opening the popup."""

    def __init__(self, refresh_choices: Callable[[], None]) -> None:
        super().__init__()
        self._refresh_choices = refresh_choices
        self._refreshing_choices = False

    def showPopup(self) -> None:
        if not self._refreshing_choices:
            self._refreshing_choices = True
            try:
                self._refresh_choices()
            finally:
                self._refreshing_choices = False
        super().showPopup()


class PakPathsEdit(QWidget):
    """Two visible PAK file rows with a picker button on each row."""

    textChanged = pyqtSignal()

    def __init__(self, browse_path: Callable[[int], None]) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._rows = []
        self._browse_buttons = []
        for index, placeholder in enumerate((
            "第一个 PAK 文件路径",
            "第二个 PAK 文件路径（可选）",
        )):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            row = QLineEdit()
            row.setPlaceholderText(placeholder)
            row.setMinimumWidth(0)
            row.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            row.textChanged.connect(lambda _text: self.textChanged.emit())
            row_layout.addWidget(row, 1)
            browse = QToolButton()
            browse.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
            browse.setToolTip(f"选择第 {index + 1} 个原始 PAK 文件")
            browse.setAccessibleName(f"选择第 {index + 1} 个原始 PAK 文件")
            browse.clicked.connect(
                lambda _checked=False, row_index=index: browse_path(row_index)
            )
            row_layout.addWidget(browse)
            layout.addWidget(row_widget)
            self._rows.append(row)
            self._browse_buttons.append(browse)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        row_height = max(
            28,
            self._rows[0].sizeHint().height(),
            self._browse_buttons[0].sizeHint().height(),
        )
        self.setFixedHeight(row_height * 2 + layout.spacing())

    def text(self) -> str:
        return "\n".join(row.text().strip() for row in self._rows if row.text().strip())

    def setText(self, value: str) -> None:
        values = [line.strip() for line in value.splitlines() if line.strip()]
        for index, row in enumerate(self._rows):
            row.setText(values[index] if index < len(values) else "")

    def rowText(self, index: int) -> str:
        return self._rows[index].text().strip()

    def setRowText(self, index: int, value: str) -> None:
        self._rows[index].setText(value)


def build_mail_workspace_page(window) -> QWidget:
    """Build the first-class Mail workspace without duplicating its runtime."""

    scroll = QScrollArea()
    scroll.setObjectName("mailWorkspacePage")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)

    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(0, 0, 8, 0)
    layout.setSpacing(10)

    header = QHBoxLayout()
    title = QLabel("自动卡邮件")
    title.setObjectName("workspaceTitle")
    header.addWidget(title)
    header.addStretch(1)
    window._standalone_mail_button = QPushButton("独立运行卡邮件")
    window._standalone_mail_button.setObjectName("primaryAction")
    window._standalone_mail_button.setIcon(
        window.style().standardIcon(QStyle.SP_MediaPlay)
    )
    window._standalone_mail_button.clicked.connect(window._toggle_standalone_mail)
    header.addWidget(window._standalone_mail_button)
    layout.addLayout(header)

    status = QFrame()
    status.setObjectName("mailRunStatus")
    status_layout = QVBoxLayout(status)
    status_layout.setContentsMargins(10, 8, 10, 8)
    status_layout.setSpacing(5)
    summary = QHBoxLayout()
    window._mail_page_state_label = QLabel("未启用")
    window._mail_page_state_label.setObjectName("mailIntegrationState")
    summary.addWidget(window._mail_page_state_label)
    window._mail_page_progress_label = QLabel("--")
    window._mail_page_progress_label.setObjectName("mailIntegrationProgress")
    summary.addWidget(window._mail_page_progress_label)
    window._mail_page_result_label = QLabel("Mail 尚未运行")
    window._mail_page_result_label.setObjectName("mailIntegrationResult")
    summary.addWidget(window._mail_page_result_label)
    summary.addStretch(1)
    window._mail_acknowledge_button = QPushButton("确认环境已恢复")
    window._mail_acknowledge_button.setObjectName("warningAction")
    window._mail_acknowledge_button.clicked.connect(
        window._acknowledge_mail_integration_error
    )
    window._mail_acknowledge_button.hide()
    summary.addWidget(window._mail_acknowledge_button)
    status_layout.addLayout(summary)
    window._mail_page_detail_label = QLabel()
    window._mail_page_detail_label.setObjectName("mailIntegrationDetail")
    window._mail_page_detail_label.setWordWrap(True)
    window._mail_page_detail_label.setSizePolicy(
        QSizePolicy.Ignored,
        QSizePolicy.Preferred,
    )
    window._mail_page_detail_label.setMinimumWidth(0)
    window._mail_page_detail_label.hide()
    status_layout.addWidget(window._mail_page_detail_label)
    layout.addWidget(status)

    environment_heading = QLabel("运行环境")
    environment_heading.setObjectName("sectionHeading")
    layout.addWidget(environment_heading)
    form = QFormLayout()
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    form.setRowWrapPolicy(QFormLayout.WrapLongRows)

    profile_widget = QWidget()
    profile_layout = QHBoxLayout(profile_widget)
    profile_layout.setContentsMargins(0, 0, 0, 0)
    profile_layout.setSpacing(6)
    window._mail_profile_path = QLineEdit()
    window._mail_profile_path.setPlaceholderText("config/mail_storage_profile.json")
    window._mail_profile_path.setMinimumWidth(0)
    window._mail_profile_path.setSizePolicy(
        QSizePolicy.Ignored,
        QSizePolicy.Fixed,
    )
    window._mail_profile_path.editingFinished.connect(
        window._reload_mail_profile_from_path
    )
    profile_layout.addWidget(window._mail_profile_path, 1)
    window._mail_profile_browse = QToolButton()
    window._mail_profile_browse.setIcon(
        window.style().standardIcon(QStyle.SP_DialogOpenButton)
    )
    window._mail_profile_browse.setToolTip("选择 Mail 配置文件")
    window._mail_profile_browse.setAccessibleName("选择 Mail 配置文件")
    window._mail_profile_browse.clicked.connect(window._browse_mail_profile)
    profile_layout.addWidget(window._mail_profile_browse)
    form.addRow("配置文件", profile_widget)

    profile_tools = QWidget()
    profile_tools_layout = QHBoxLayout(profile_tools)
    profile_tools_layout.setContentsMargins(0, 0, 0, 0)
    profile_tools_layout.setSpacing(6)
    window._mail_profile_status = QLabel()
    window._mail_profile_status.setObjectName("settingHint")
    window._mail_profile_status.setWordWrap(True)
    window._mail_profile_status.setSizePolicy(
        QSizePolicy.Ignored,
        QSizePolicy.Preferred,
    )
    window._mail_profile_status.setMinimumWidth(0)
    profile_tools_layout.addWidget(window._mail_profile_status, 1)
    window._mail_windows_refresh = QToolButton()
    window._mail_windows_refresh.setIcon(
        window.style().standardIcon(QStyle.SP_BrowserReload)
    )
    window._mail_windows_refresh.setToolTip("刷新并自动匹配三个目标窗口")
    window._mail_windows_refresh.setAccessibleName("刷新卡邮件目标窗口")
    window._mail_windows_refresh.clicked.connect(
        lambda _checked=False: window._refresh_mail_profile_windows(
            show_candidates=True,
        )
    )
    profile_tools_layout.addWidget(window._mail_windows_refresh)
    form.addRow("配置状态", profile_tools)

    window._mail_scheme_index = NoWheelComboBox()
    for scheme_index in range(1, 4):
        window._mail_scheme_index.addItem(
            f"配装方案 {scheme_index}",
            scheme_index,
        )
    form.addRow("卡邮件配装方案", window._mail_scheme_index)

    _add_mail_window_row(window, form, "游戏窗口", "game")
    _add_mail_window_row(window, form, "启动器主窗口", "launcher")
    _add_mail_window_row(
        window,
        form,
        "启动器设置窗口",
        "launcher_settings",
    )

    window._mail_pak_source = PakPathsEdit(window._browse_mail_pak_source)
    window._mail_pak_source.textChanged.connect(window._refresh_mail_pak_status)
    window._mail_pak_source_browse = window._mail_pak_source._browse_buttons[0]
    window._mail_pak_optional_source_browse = window._mail_pak_source._browse_buttons[1]
    form.addRow("PAK 原文件（最多 2 个）", window._mail_pak_source)

    pak_stage_widget = QWidget()
    pak_stage_layout = QHBoxLayout(pak_stage_widget)
    pak_stage_layout.setContentsMargins(0, 0, 0, 0)
    pak_stage_layout.setSpacing(6)
    window._mail_pak_stage_directory = _path_editor()
    window._mail_pak_stage_directory.textChanged.connect(
        window._refresh_mail_pak_status
    )
    pak_stage_layout.addWidget(window._mail_pak_stage_directory, 1)
    window._mail_pak_stage_browse = QToolButton()
    window._mail_pak_stage_browse.setIcon(
        window.style().standardIcon(QStyle.SP_DirOpenIcon)
    )
    window._mail_pak_stage_browse.setToolTip("选择 PAK 暂存文件夹")
    window._mail_pak_stage_browse.setAccessibleName("选择 PAK 暂存文件夹")
    window._mail_pak_stage_browse.clicked.connect(window._browse_mail_pak_stage)
    pak_stage_layout.addWidget(window._mail_pak_stage_browse)
    form.addRow("PAK 暂存文件夹", pak_stage_widget)

    window._mail_pak_status = QLabel("路径尚未配置")
    window._mail_pak_status.setObjectName("mailPakStatus")
    window._mail_pak_status.setWordWrap(True)
    form.addRow("PAK 当前状态", window._mail_pak_status)

    # These runtime safeguards remain available to the existing workflow API,
    # but users no longer need to manage them as visible settings.
    window._mail_allow_input = QCheckBox(content)
    window._mail_allow_input.setChecked(True)
    window._mail_allow_input.hide()
    window._mail_allow_file_operations = QCheckBox(content)
    window._mail_allow_file_operations.setChecked(True)
    window._mail_allow_file_operations.hide()
    layout.addLayout(form)

    environment_actions = QHBoxLayout()
    environment_actions.addStretch(1)
    window._mail_save_environment_button = QPushButton("保存卡邮件环境")
    window._mail_save_environment_button.setIcon(
        window.style().standardIcon(QStyle.SP_DialogSaveButton)
    )
    window._mail_save_environment_button.clicked.connect(
        window._save_mail_environment_settings
    )
    environment_actions.addWidget(window._mail_save_environment_button)
    layout.addLayout(environment_actions)

    layout.addStretch(1)

    scroll.setWidget(content)
    return scroll


def _add_mail_window_row(
    window,
    form: QFormLayout,
    label: str,
    target: str,
) -> None:
    row_widget = QWidget()
    row_layout = QHBoxLayout(row_widget)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(8)
    combo = RefreshingWindowComboBox(window._refresh_mail_profile_windows)
    combo.setMinimumWidth(260)
    combo.setSizeAdjustPolicy(
        NoWheelComboBox.AdjustToMinimumContentsLengthWithIcon
    )
    combo.setMinimumContentsLength(20)
    combo.currentIndexChanged.connect(
        lambda index, binding_target=target: window._bind_mail_profile_window(
            binding_target,
            index,
        )
    )
    row_layout.addWidget(combo, 1)
    status = QLabel("未配置")
    status.setObjectName("mailBindingStatus")
    status.setMinimumWidth(125)
    row_layout.addWidget(status)
    setattr(window, f"_mail_{target}_window", combo)
    setattr(window, f"_mail_{target}_status", status)
    form.addRow(label, row_widget)


def _path_editor() -> QLineEdit:
    editor = QLineEdit()
    editor.setMinimumWidth(0)
    editor.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    return editor
