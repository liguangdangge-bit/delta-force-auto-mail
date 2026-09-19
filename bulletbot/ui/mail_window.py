"""Mail-only Qt shell. No trading pages or trading UI controllers are created."""
from datetime import datetime
import os
from pathlib import Path
import threading

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QTextEdit, QMessageBox, QSizePolicy,
)

from bulletbot.hotkeys import GlobalHotkeys
from bulletbot.mail_storage.models import MailWorkflowOutcome, workflow_state_label
from bulletbot.orchestration import MailIntegrationSettings
from bulletbot.ui.diagnostic_export import DiagnosticExportMixin
from bulletbot.ui.mail_integration_settings import MailIntegrationSettingsMixin
from bulletbot.ui.mail_page_view import build_mail_workspace_page
from bulletbot.ui.theme import APP_NAME


class MailWindow(MailIntegrationSettingsMixin, DiagnosticExportMixin, QMainWindow):
    event_received = pyqtSignal(object)
    runtime_message = pyqtSignal(str, str)

    def __init__(self, runtime, recorder, *, interactive=True):
        super().__init__()
        self._mail_runtime = runtime
        self._run_recorder = recorder
        self._mail_orchestrator = None
        self._mail_profile_editor = None
        self._close_when_idle = False
        self._closing = False
        self._preview_enabled = True
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._last_pixmap = None
        self._hotkeys = GlobalHotkeys()
        self._interactive = interactive
        self.setWindowTitle(APP_NAME)
        self.resize(1320, 850)
        self._build_layout()
        self.event_received.connect(self._on_event)
        self.runtime_message.connect(self._append_log)
        runtime.set_workflow_observers(on_event=self.event_received.emit, on_frame=self._receive_frame)
        self._mail_profile_path.setText(runtime.settings.mail_profile_path)
        self._reload_mail_profile_from_path()
        self._refresh_mail_integration_status()
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._hotkeys.stop_requested.connect(self._stop_all_activity)
        self._hotkeys.pause_toggled.connect(self._toggle_pause)
        self._hotkeys.toggle_window_requested.connect(self._toggle_window)
        if interactive:
            error = self._hotkeys.start()
            self._append_log(error or '快捷键：Ctrl+Space 暂停/继续，Ctrl+Esc 安全停止，Ctrl+Alt+2 显示/隐藏。')
        if runtime.startup_error:
            self._set_status(runtime.startup_error, error=True)
            self._append_log(runtime.startup_error, 'ERROR')

    def _build_layout(self):
        shell = QWidget()
        shell.setObjectName('appRoot')
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(12, 12, 12, 12)
        title = QLabel(APP_NAME)
        title.setObjectName('brandTitle')
        header = QHBoxLayout()
        header.addWidget(title)
        header.addStretch()
        top = QPushButton('窗口置顶')
        top.setCheckable(True)
        top.toggled.connect(self._set_on_top)
        header.addWidget(top)
        self._pause_button = QPushButton('安全暂停')
        self._pause_button.clicked.connect(self._toggle_pause)
        header.addWidget(self._pause_button)
        stop = QPushButton('安全停止')
        stop.clicked.connect(self._stop_all_activity)
        header.addWidget(stop)
        layout.addLayout(header)
        self._author_notice = QLabel('抖音号：86599533954  ｜  源码公开，仅限非商业用途，禁止商用')
        self._author_notice.setObjectName('authorNotice')
        self._author_notice.setWordWrap(True)
        self._author_notice.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._author_notice.setStyleSheet(
            'color: #f3c861; background: #101b21; border-left: 3px solid #29c99a; padding: 8px 10px;'
        )
        layout.addWidget(self._author_notice)
        self._status = QLabel('确认配装方案和运行环境后，启动独立卡邮件。')
        self._status.setWordWrap(True)
        self._status.setObjectName('statusMessage')
        self._runtime_state = QLabel('待启动')
        self._runtime_state.setObjectName('runtimeState')
        self._runtime_state.setWordWrap(True)
        for label in (self._status, self._runtime_state):
            label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            layout.addWidget(label)
        splitter = QSplitter(Qt.Horizontal)
        self._main_splitter = splitter
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)
        splitter.addWidget(self._build_preview_logs())
        self._mail_page = build_mail_workspace_page(self)
        self._mail_page.setMinimumWidth(560)
        splitter.addWidget(self._mail_page)
        splitter.setSizes([470, 810])
        layout.addWidget(splitter, 1)
        self.setCentralWidget(shell)

    def _build_preview_logs(self):
        split = QSplitter(Qt.Vertical)
        split.setMinimumWidth(350)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(10)
        preview = QWidget()
        pl = QVBoxLayout(preview)
        header = QHBoxLayout()
        heading = QLabel('游戏实时预览')
        heading.setObjectName('sectionHeading')
        header.addWidget(heading, 1)
        self._preview_toggle = QPushButton('关闭')
        self._preview_toggle.clicked.connect(self._toggle_preview)
        header.addWidget(self._preview_toggle)
        pl.addLayout(header)
        self._preview = QLabel('启动卡邮件后显示当前识别画面。')
        self._preview.setObjectName('livePreview')
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumSize(320, 180)
        self._preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        pl.addWidget(self._preview, 1)
        split.addWidget(preview)
        logs = QWidget()
        ll = QVBoxLayout(logs)
        heading = QLabel('运行日志')
        heading.setObjectName('sectionHeading')
        ll.addWidget(heading)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.document().setMaximumBlockCount(1500)
        ll.addWidget(self._log, 1)
        actions = QHBoxLayout()
        for text, action in [('运行报告', self._open_run_report), ('日志目录', self._open_run_directory)]:
            b = QPushButton(text)
            b.clicked.connect(action)
            actions.addWidget(b)
        ll.addLayout(actions)
        self._export_diagnostics_button = QPushButton('导出本次运行日志压缩包')
        self._export_diagnostics_button.clicked.connect(self._export_run_diagnostics)
        ll.addWidget(self._export_diagnostics_button)
        split.addWidget(logs)
        split.setSizes([310, 360])
        return split

    def _mail_integration_settings_payload(self):
        return MailIntegrationSettings(enabled=False, mail_profile_path=self._mail_profile_path.text().strip())

    def _toggle_mail_settings_controls(self, _enabled=False):
        active = self._mail_runtime.snapshot().standalone_active
        self._mail_profile_path.setEnabled(not active)
        self._mail_profile_browse.setEnabled(not active)
        ready = self._mail_profile_editor_instance() is not None
        for control in self._mail_profile_controls():
            control.setEnabled(ready and not active)

    def _toggle_standalone_mail(self):
        if self._closing or self._close_when_idle:
            return False
        runtime = self._mail_runtime
        snapshot = runtime.snapshot()
        if snapshot.standalone_active:
            self._stop_all_activity()
            return False
        if snapshot.standalone_requires_recovery:
            self._set_status('上次卡邮件未安全完成，请先确认环境已恢复。', error=True)
            return False
        answer = QMessageBox.question(self, '确认独立运行卡邮件',
            '运行前请脱下身上的胸挂和背包，并确认方案5已保存为空白配装。\n'
            '请确认原软件没有在操作同一个游戏或PAK文件。',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return False
        try:
            from bulletbot.resource_check import require_local_resources
            require_local_resources()
            profile = self._persist_current_mail_environment(runtime)
            runtime.start_standalone_mail()
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            self._set_status(str(error), error=True)
            self._append_log(str(error), 'ERROR')
            self._refresh_mail_integration_status()
            return False
        self._append_log(f'独立卡邮件已启动，配装方案 {profile.mail_scheme_index}。')
        self._set_status('独立卡邮件运行中。')
        self._refresh_mail_integration_status()
        return True

    def _stop_all_activity(self):
        if self._mail_runtime.request_stop_standalone_mail():
            self._set_status('正在安全停止，等待当前流程退出并恢复 PAK。')
            self._append_log('已请求安全停止，等待 PAK 恢复。')
        self._refresh_mail_integration_status()

    def _toggle_pause(self):
        runtime = self._mail_runtime
        requested, paused = runtime.active_workflow_pause_state()
        result = runtime.request_resume_active_workflow() if requested or paused else runtime.request_pause_active_workflow()
        if result:
            self._append_log('已请求继续运行。' if requested or paused else '已请求在安全位置暂停。')
        self._refresh_mail_integration_status()

    def _tick(self):
        runtime = self._mail_runtime
        if runtime.snapshot().standalone_active:
            try:
                result = runtime.poll_standalone_mail()
            except (OSError, ValueError, RuntimeError) as error:
                self._set_status(str(error), error=True)
                self._append_log(str(error), 'ERROR')
            else:
                if result is not None:
                    failed = result.outcome == MailWorkflowOutcome.FAILED or not result.pak_restored
                    self._set_status(result.message, error=failed)
                    self._append_log(result.message, 'ERROR' if failed else 'INFO')
        self._refresh_mail_integration_status()
        with self._frame_lock:
            frame, self._latest_frame = self._latest_frame, None
        if frame is not None and self._preview_enabled:
            rgb = np.ascontiguousarray(frame[:, :, :3][:, :, ::-1])
            image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
            self._last_pixmap = QPixmap.fromImage(image)
            self._draw_preview()
        if self._close_when_idle and not runtime.snapshot().standalone_active:
            self.close()

    def _refresh_mail_integration_status(self):
        s = self._mail_runtime.snapshot()
        requested, paused = self._mail_runtime.active_workflow_pause_state()
        state = ('正在安全停止' if s.standalone_stop_requested else '已暂停' if paused else
                 '等待安全暂停' if requested else '运行中') if s.standalone_active else (
                 '需要确认恢复' if s.standalone_requires_recovery else '待启动')
        result = s.standalone_result
        self._mail_page_state_label.setText(state)
        self._mail_page_progress_label.setText(f'已完成 {result.completed_rounds} 轮' if result else '独立模式')
        self._mail_page_result_label.setText(('PAK 已恢复' if result.pak_restored else 'PAK 未恢复') if result else '')
        self._mail_page_detail_label.setText(s.standalone_error or '')
        self._mail_page_detail_label.setVisible(bool(s.standalone_error))
        self._mail_acknowledge_button.setVisible(s.standalone_requires_recovery)
        self._standalone_mail_button.setText('正在安全停止…' if s.standalone_stop_requested else
                                           '停止独立卡邮件' if s.standalone_active else '独立运行卡邮件')
        self._standalone_mail_button.setEnabled(not s.standalone_stop_requested and not s.standalone_requires_recovery and not self._close_when_idle)
        self._pause_button.setText('继续运行' if requested or paused else '安全暂停')
        self._pause_button.setEnabled(s.standalone_active and not s.standalone_stop_requested)
        self._toggle_mail_settings_controls()

    def _acknowledge_mail_integration_error(self):
        if QMessageBox.question(self, '确认环境已恢复', '已人工确认 PAK 在原路径，且游戏与启动器状态正常？',
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            self._mail_runtime.acknowledge_standalone_recovery()
        except (OSError, ValueError, RuntimeError) as error:
            self._set_status(str(error), error=True)
            return
        self._append_log('已人工确认独立卡邮件环境恢复。')
        self._refresh_mail_integration_status()

    def _on_event(self, event):
        self._runtime_state.setText(f'{workflow_state_label(event.state)} | {event.message}')
        self._append_log(event.message, event.level.upper())

    def _receive_frame(self, frame, observation, target):
        if self._preview_enabled:
            with self._frame_lock:
                self._latest_frame = frame.copy()

    def _draw_preview(self):
        if self._last_pixmap is not None and self._preview_enabled:
            self._preview.setPixmap(self._last_pixmap.scaled(self._preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _toggle_preview(self):
        self._preview_enabled = not self._preview_enabled
        self._preview_toggle.setText('关闭' if self._preview_enabled else '开启')
        self._preview.clear()
        self._preview.setText('等待新的识别画面。' if self._preview_enabled else '实时预览已关闭，识别不受影响。')
        self._draw_preview()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_last_pixmap'):
            self._draw_preview()

    def _set_status(self, message, error=False):
        self._status.setText(message)
        self._status.setStyleSheet('color: #ff8279;' if error else 'color: #54e9bb;')

    def _append_log(self, message, level='INFO'):
        self._log.moveCursor(QTextCursor.End)
        self._log.insertPlainText(f'[{datetime.now():%H:%M:%S}] [{level}] {message}\n')
        if self._run_recorder:
            self._run_recorder.log(message, level)

    def report_runtime_message(self, message, level='INFO'):
        self.runtime_message.emit(message, level)

    def _open_run_report(self):
        try:
            os.startfile(self._run_recorder.generate_report())
        except (OSError, RuntimeError) as error:
            self._set_status(str(error), error=True)

    def _open_run_directory(self):
        os.startfile(self._run_recorder.directory)

    def _set_on_top(self, enabled):
        self.setWindowFlag(Qt.WindowStaysOnTopHint, enabled)
        self.show()

    def _toggle_window(self):
        if self.isVisible() and not self.isMinimized():
            self.hide()
        else:
            self.showNormal()
            self.raise_()
            self.activateWindow()

    def closeEvent(self, event):
        if self._closing:
            event.accept()
            return
        self._close_when_idle = True
        if self._mail_runtime.snapshot().standalone_active:
            self._stop_all_activity()
            event.ignore()
            return
        worker = getattr(self, '_diagnostic_export_worker', None)
        if worker is not None:
            self._set_status('等待日志导出结束后关闭。')
            event.ignore()
            return
        self._closing = True
        self._timer.stop()
        self._hotkeys.stop()
        self._run_recorder.close()
        event.accept()
