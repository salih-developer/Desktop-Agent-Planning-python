from __future__ import annotations

import customtkinter as ctk

from core.executor import TaskStatus

_STATUS_ICONS = {
    TaskStatus.PENDING: ("○", "#6B7280"),
    TaskStatus.RUNNING: ("◉", "#F59E0B"),
    TaskStatus.DONE: ("✓", "#10B981"),
    TaskStatus.FAILED: ("✗", "#EF4444"),
    TaskStatus.SKIPPED: ("⊘", "#6B7280"),
}


class TaskItem(ctk.CTkFrame):
    def __init__(self, parent, task_id: str, tool: str, description: str, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self._icon_var = ctk.StringVar(value="○")
        self._color_var = "#6B7280"

        self._icon_label = ctk.CTkLabel(
            self, textvariable=self._icon_var,
            font=ctk.CTkFont(size=14),
            text_color=self._color_var,
            width=20,
        )
        self._icon_label.pack(side="left", padx=(4, 2))

        tool_badge = ctk.CTkLabel(
            self,
            text=tool,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=("#1D4ED8", "#1E3A5F"),
            corner_radius=6,
            padx=6,
        )
        tool_badge.pack(side="left", padx=4)

        desc_label = ctk.CTkLabel(
            self,
            text=description[:60],
            font=ctk.CTkFont(size=12),
            anchor="w",
        )
        desc_label.pack(side="left", padx=4, fill="x", expand=True)

    def set_status(self, status: TaskStatus) -> None:
        icon, color = _STATUS_ICONS.get(status, ("?", "#6B7280"))
        self._icon_var.set(icon)
        self._icon_label.configure(text_color=color)
