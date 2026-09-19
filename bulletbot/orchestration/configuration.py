from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bulletbot.mail_storage.settings import AppSettings


@dataclass(frozen=True)
class MailIntegrationSettings:
    enabled: bool = False
    product_name: str = ""
    quantity_threshold: int = 10_000
    mail_profile_path: str = "config/mail_storage_profile.json"

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.quantity_threshold <= 0:
            raise ValueError("卡邮件触发数量必须大于零")
        if not self.mail_profile_path.strip():
            raise ValueError("卡邮件联动缺少 Mail 配置文件路径")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MailIntegrationSettings":
        settings = cls(
            enabled=bool(value.get("enabled", False)),
            product_name=str(value.get("product_name", "")),
            quantity_threshold=int(value.get("quantity_threshold", 10_000)),
            mail_profile_path=str(
                value.get("mail_profile_path", "config/mail_storage_profile.json")
            ),
        )
        settings.validate()
        return settings


class MailIntegrationSettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> MailIntegrationSettings:
        if not self.path.exists():
            return MailIntegrationSettings()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取卡邮件联动配置：{exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("卡邮件联动配置根节点必须是对象")
        return MailIntegrationSettings.from_dict(value)

    def save(self, settings: MailIntegrationSettings) -> None:
        settings.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(settings.to_dict(), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise


def load_mail_profile(path: Path) -> AppSettings:
    if not path.is_file():
        raise ValueError(f"Mail 配置文件不存在：{path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 Mail 配置文件：{exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Mail 配置文件根节点必须是对象")
    return AppSettings.from_dict(value)
