from pathlib import Path

from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from bulletbot.reporting.diagnostic_export import DiagnosticPackage


class DiagnosticExportWorker(QThread):
    def __init__(self, recorder, parent):
        super().__init__(parent)
        self.recorder = recorder
        self.package = None
        self.error = None

    def run(self):
        try:
            self.package = DiagnosticPackage(self.recorder)
        except Exception as exc:
            self.error = str(exc)


class DiagnosticExportMixin:
    def _export_run_diagnostics(self):
        if self._run_recorder is None or getattr(self, "_diagnostic_export_worker", None):
            return
        self._export_diagnostics_button.setEnabled(False)
        self._set_status("正在生成并检查本次运行诊断包…")
        worker = DiagnosticExportWorker(self._run_recorder, self)
        self._diagnostic_export_worker = worker
        worker.finished.connect(self._finish_diagnostic_export)
        worker.start()

    def _finish_diagnostic_export(self):
        worker = self._diagnostic_export_worker
        try:
            if worker.error:
                raise RuntimeError(worker.error)
            package = worker.package
            while True:
                filename, _ = QFileDialog.getSaveFileName(
                    self, "保存本次运行日志压缩包", str(Path.home() / package.path.name),
                    "ZIP 压缩包 (*.zip)",
                )
                if not filename:
                    self._set_status("已取消导出诊断包。")
                    return
                try:
                    package.save(Path(filename))
                except OSError as exc:
                    QMessageBox.warning(self, "保存失败", f"{exc}\n请重新选择保存位置。")
                    continue
                self._set_status(f"诊断包已保存：{filename}")
                return
        except Exception as exc:
            self._set_status(f"导出诊断包失败：{exc}", error=True)
            QMessageBox.warning(self, "导出诊断包失败", str(exc))
        finally:
            if worker.package is not None:
                worker.package.cleanup()
            self._diagnostic_export_worker = None
            self._export_diagnostics_button.setEnabled(True)
            worker.deleteLater()
