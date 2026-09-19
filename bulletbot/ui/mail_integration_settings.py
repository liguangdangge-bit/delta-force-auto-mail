from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QFileDialog, QMessageBox, QStyle

from bulletbot.mail_storage.models import MailWorkflowOutcome
from bulletbot.mail_storage.pak_transaction import PakTransaction
from bulletbot.mail_storage.profile_editor import MailProfileEditor
from bulletbot.mail_storage.settings import MAX_PAK_FILES, AppSettings
from bulletbot.orchestration import (
    MailIntegrationRuntime,
    MailIntegrationSettings,
    OrchestrationError,
    OrchestrationState,
)


class MailIntegrationSettingsMixin:
    """Bind mail integration configuration and recovery controls to the Qt shell."""




    def _browse_mail_profile(self) -> None:
        runtime = self._mail_integration_runtime()
        project_root = runtime.project_root if runtime is not None else Path.cwd()
        raw_path = self._mail_profile_path.text().strip()
        initial = Path(raw_path).expanduser() if raw_path else project_root / "config"
        if not initial.is_absolute():
            initial = project_root / initial
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "选择或新建 Mail 配置文件",
            str(initial),
            "JSON 配置 (*.json);;所有文件 (*)",
        )
        if not filename:
            return
        path = Path(filename)
        try:
            path = path.relative_to(project_root)
        except ValueError:
            pass
        self._mail_profile_path.setText(str(path))
        self._reload_mail_profile_from_path()

    def _reload_mail_profile_from_path(self) -> None:
        runtime = self._mail_integration_runtime()
        if runtime is None:
            self._mail_profile_editor = None
            self._mail_profile_status.setText("运行时不可用")
            self._clear_mail_window_controls()
            self._toggle_mail_settings_controls(False)
            return
        raw_path = self._mail_profile_path.text().strip()
        if not raw_path:
            raw_path = "config/mail_storage_profile.json"
            self._mail_profile_path.setText(raw_path)
        try:
            editor = runtime.create_profile_editor(raw_path)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            self._mail_profile_editor = None
            self._mail_profile_status.setText("配置读取失败")
            self._mail_profile_status.setToolTip(str(exc))
            self._clear_mail_window_controls()
            self._toggle_mail_settings_controls(False)
            return
        self._mail_profile_editor = editor
        profile = editor.settings
        self._mail_pak_source.setText("\n".join(profile.effective_pak_sources()))
        stage_directory = str(Path(profile.pak_stage).parent) if profile.pak_stage else ""
        self._mail_pak_stage_directory.setText(stage_directory)
        self._mail_allow_input.setChecked(True)
        self._mail_allow_file_operations.setChecked(True)
        scheme_index = self._mail_scheme_index.findData(profile.mail_scheme_index)
        self._mail_scheme_index.setCurrentIndex(max(0, scheme_index))
        if editor.imported_from is not None:
            self._mail_profile_status.setText("已载入旧 Mail 配置；保存后迁移到主项目")
            self._mail_profile_status.setToolTip(
                f"来源：{editor.imported_from}\n目标：{editor.path}"
            )
        elif editor.path.is_file():
            self._mail_profile_status.setText("已加载 Mail 配置")
            self._mail_profile_status.setToolTip(str(editor.path))
        else:
            self._mail_profile_status.setText("保存设置时创建新 Mail 配置")
            self._mail_profile_status.setToolTip(str(editor.path))
        self._refresh_mail_profile_windows()
        self._refresh_mail_pak_status()
        self._toggle_mail_settings_controls(False)

    def _refresh_mail_profile_windows(self, *, show_candidates: bool = False) -> None:
        editor = self._mail_profile_editor_instance()
        if editor is None:
            self._clear_mail_window_controls()
            return
        try:
            states = editor.refresh()
        except Exception as exc:
            self._mail_profile_status.setText("窗口刷新失败")
            self._mail_profile_status.setToolTip(str(exc))
            return
        candidate_counts: dict[str, int] = {}
        first_unbound_combo = None
        fallback_combos: dict[str, object] = {}
        for target in MailProfileEditor.TARGETS:
            combo = getattr(self, f"_mail_{target}_window")
            candidates = editor.candidate_windows(target)
            candidate_counts[target] = len(candidates)
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(
                "请选择窗口" if candidates else "未发现可绑定窗口，请打开后重试",
                None,
            )
            for window in candidates:
                combo.addItem(window.label, window.hwnd)
            state = states[target]
            if state.bound is not None:
                index = combo.findData(state.bound.hwnd)
                combo.setCurrentIndex(index if index >= 0 else 0)
                combo.setToolTip(state.bound.label)
            else:
                combo.setCurrentIndex(0)
                combo.setToolTip(
                    "展开列表时会自动刷新当前窗口"
                    if candidates
                    else "没有发现符合该类型的窗口；请先打开目标窗口，再展开重试"
                )
            combo.blockSignals(False)
            status = getattr(self, f"_mail_{target}_status")
            if state.bound is not None:
                status.setText(f"已绑定 0x{state.bound.hwnd:X}")
            elif state.ambiguous_count:
                status.setText(f"发现 {len(candidates)} 个候选，请选择")
                if first_unbound_combo is None and candidates:
                    first_unbound_combo = combo
            elif candidates:
                status.setText(f"发现 {len(candidates)} 个候选，请选择")
                if first_unbound_combo is None:
                    first_unbound_combo = combo
            elif state.configured:
                status.setText("已记住，等待窗口出现")
            else:
                status.setText(
                    "未配置" if candidates else "未发现窗口"
                )
            if candidates:
                fallback_combos[target] = combo

        if not show_candidates:
            return
        labels = {
            "game": "游戏",
            "launcher": "启动器",
            "launcher_settings": "设置",
        }
        summary = "，".join(
            f"{labels[target]} {candidate_counts[target]}"
            for target in MailProfileEditor.TARGETS
        )
        self._mail_profile_status.setText(f"窗口刷新完成：{summary}")
        self._mail_profile_status.setToolTip("下拉列表已更新")
        popup_combo = first_unbound_combo
        if popup_combo is None:
            for target in ("launcher", "launcher_settings", "game"):
                popup_combo = fallback_combos.get(target)
                if popup_combo is not None:
                    break
        if popup_combo is not None:
            popup_combo.setFocus()
            popup_combo.showPopup()

    def _bind_mail_profile_window(self, target: str, index: int) -> None:
        editor = self._mail_profile_editor_instance()
        if editor is None or index <= 0:
            return
        combo = getattr(self, f"_mail_{target}_window")
        hwnd = combo.itemData(index)
        if not isinstance(hwnd, int):
            return
        try:
            state = editor.bind(target, hwnd)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "窗口绑定失败", str(exc))
            self._refresh_mail_profile_windows()
            return
        labels = {
            "game": "游戏窗口",
            "launcher": "启动器主窗口",
            "launcher_settings": "启动器设置窗口",
        }
        self._append_log(
            f"{labels[target]}已保存窗口指纹：{state.bound.label if state.bound else '--'}"
        )
        self._mail_profile_status.setText("窗口指纹已保存")
        self._mail_profile_status.setToolTip(str(editor.path))
        self._refresh_mail_profile_windows()

    def _clear_mail_window_controls(self) -> None:
        for target in MailProfileEditor.TARGETS:
            combo = getattr(self, f"_mail_{target}_window")
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("请选择窗口", None)
            combo.blockSignals(False)
            getattr(self, f"_mail_{target}_status").setText("未配置")

    def _browse_mail_pak_source(self, row_index: int = 0) -> None:
        current = self._mail_pak_source.rowText(row_index)
        initial = Path(current).expanduser().parent if current else Path.cwd()
        filename, _ = QFileDialog.getOpenFileName(
            self,
            f"选择第 {row_index + 1} 个原始 PAK 文件",
            str(initial),
            "PAK 文件 (*.pak);;所有文件 (*)",
        )
        if not filename:
            return
        selected = str(Path(filename).expanduser().resolve(strict=False)).casefold()
        other_paths = {
            str(
                Path(self._mail_pak_source.rowText(index))
                .expanduser()
                .resolve(strict=False)
            ).casefold()
            for index in range(MAX_PAK_FILES)
            if index != row_index and self._mail_pak_source.rowText(index)
        }
        if selected in other_paths:
            QMessageBox.warning(
                self,
                "PAK 文件重复",
                "两个 PAK 原文件不能选择同一个路径，请重新选择。",
            )
            return
        self._mail_pak_source.setRowText(row_index, filename)

    def _browse_mail_pak_stage(self) -> None:
        current = self._mail_pak_stage_directory.text().strip()
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择 PAK 暂存文件夹",
            current or str(Path.cwd()),
        )
        if directory:
            self._mail_pak_stage_directory.setText(directory)

    def _mail_staged_pak_path(self) -> str:
        staged_paths = self._mail_staged_pak_paths()
        return str(staged_paths[0]) if staged_paths else ""

    def _mail_pak_sources(self) -> tuple[Path, ...]:
        return tuple(
            Path(line.strip()).expanduser()
            for line in self._mail_pak_source.text().splitlines()
            if line.strip()
        )

    def _mail_staged_pak_paths(self) -> tuple[Path, ...]:
        sources = self._mail_pak_sources()
        directory = self._mail_pak_stage_directory.text().strip()
        if not sources or not directory or any(not source.name for source in sources):
            return ()
        stage_directory = Path(directory).expanduser()
        return tuple(stage_directory / source.name for source in sources)

    def _refresh_mail_pak_status(self) -> None:
        sources = self._mail_pak_sources()
        staged_paths = self._mail_staged_pak_paths()
        if (
            not sources
            or len(sources) > MAX_PAK_FILES
            or len(sources) != len(staged_paths)
            or len(set(sources)) != len(sources)
            or len({source.name.casefold() for source in sources}) != len(sources)
            or any(source == staged for source, staged in zip(sources, staged_paths))
        ):
            self._mail_pak_status.setText("路径无效")
            return
        status = PakTransaction(
            sources,
            staged_paths,
            enabled=False,
        ).inspect()
        count_label = f"{len(sources)} 个 PAK"
        self._mail_pak_status.setText(
            {
                "ready": f"{count_label} 原文件均在位",
                "staged": f"{count_label} 均已暂存",
                "partial": f"{count_label} 状态不一致，运行前将尝试恢复",
                "conflict": f"{count_label} 冲突：原路径和暂存路径都有文件",
                "missing": f"{count_label} 存在缺失文件",
            }[status]
        )


    def _mail_profile_from_controls(self) -> AppSettings | None:
        editor = self._mail_profile_editor_instance()
        if editor is None:
            return None
        sources = self._mail_pak_sources()
        editor.update_environment(
            pak_source=str(sources[0]) if sources else "",
            pak_stage=self._mail_staged_pak_path(),
            allow_input=True,
            allow_file_operations=True,
            mail_scheme_index=int(self._mail_scheme_index.currentData() or 1),
            pak_sources=[str(source) for source in sources],
        )
        return editor.settings

    def _persist_current_mail_environment(
        self,
        runtime: MailIntegrationRuntime,
    ) -> AppSettings:
        settings = self._mail_integration_settings_payload()
        profile = self._mail_profile_from_controls()
        if profile is None:
            raise ValueError("卡邮件配置文件尚未加载")
        self._mail_orchestrator = runtime.configure(settings, profile)
        return profile



    def _save_mail_environment_settings(self) -> None:
        runtime = self._mail_integration_runtime()
        if runtime is None:
            self._set_status("当前应用未初始化卡邮件运行时。", error=True)
            return
        try:
            profile = self._persist_current_mail_environment(runtime)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "卡邮件环境保存失败", str(exc))
            self._set_status(f"卡邮件环境保存失败：{exc}", error=True)
            return
        editor = self._mail_profile_editor_instance()
        if editor is not None:
            editor.imported_from = None
            self._mail_profile_status.setText("卡邮件环境已保存")
            self._mail_profile_status.setToolTip(str(editor.path))
        self._append_log(
            f"卡邮件窗口、PAK 路径和方案 {profile.mail_scheme_index} 已保存；"
            "输入及 PAK 操作已自动启用。"
        )
        self._set_status("卡邮件环境已保存。")
        self._refresh_mail_integration_status()





    def _mail_integration_runtime(self) -> MailIntegrationRuntime | None:
        runtime = getattr(self, "_mail_runtime", None)
        return runtime if isinstance(runtime, MailIntegrationRuntime) else None

    def _mail_profile_editor_instance(self) -> MailProfileEditor | None:
        editor = getattr(self, "_mail_profile_editor", None)
        return editor if isinstance(editor, MailProfileEditor) else None

    def _mail_profile_controls(self) -> tuple[object, ...]:
        return (
            self._mail_windows_refresh,
            self._mail_game_window,
            self._mail_launcher_window,
            self._mail_launcher_settings_window,
            self._mail_scheme_index,
            self._mail_pak_source,
            self._mail_pak_source_browse,
            self._mail_pak_optional_source_browse,
            self._mail_pak_stage_directory,
            self._mail_pak_stage_browse,
            self._mail_allow_input,
            self._mail_allow_file_operations,
            self._mail_save_environment_button,
        )
