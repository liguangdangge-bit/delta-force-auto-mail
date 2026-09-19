import sys
from pathlib import Path

from bulletbot.ocr.favorites_ocr import preload_onnx_runtime

preload_onnx_runtime()

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QLockFile
from bulletbot.mail_storage.paths import user_data_dir
from bulletbot.orchestration import MailIntegrationRuntime
from bulletbot.platform.game_window_session import SharedGameWindowSession
from bulletbot.reporting.run_recorder import RunRecorder
from bulletbot.ui.mail_window import MailWindow
from bulletbot.ui.theme import APP_NAME, DARK_GREEN_THEME
from bulletbot.window.service import WindowService
from bulletbot.ocr.runtime_backend import configure_ocr_runtime_reporter


def run():
    root = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[1]
    WindowService.enable_dpi_awareness()
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName(APP_NAME)
    app.setStyleSheet(DARK_GREEN_THEME)
    lock = QLockFile(str(user_data_dir() / 'application.lock'))
    if not lock.tryLock(0):
        QMessageBox.information(None, APP_NAME, '独立卡邮件已经运行，请使用已打开的窗口。')
        return
    recorder = RunRecorder(root / 'logs/runs')
    original_hook = sys.excepthook
    def record_exception(kind, value, tb):
        recorder.record_exception(kind, value, tb)
        original_hook(kind, value, tb)
    sys.excepthook = record_exception
    runtime = MailIntegrationRuntime(root, run_directory=recorder.directory,
                                     game_window_session=SharedGameWindowSession())
    window = MailWindow(runtime, recorder)
    configure_ocr_runtime_reporter(window.report_runtime_message)
    window.show()
    try:
        app.exec_()
    finally:
        lock.unlock()
