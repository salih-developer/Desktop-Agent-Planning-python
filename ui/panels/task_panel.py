from __future__ import annotations

from typing import Callable

import customtkinter as ctk

from core.executor import TaskExecution, TaskStatus
from core.planner import TaskList
from ui.widgets.task_item import TaskItem


class TaskPanel(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._items: dict[str, TaskItem] = {}
        self._on_approve: Callable | None = None
        self._on_reject: Callable | None = None
        self._build()

    def _build(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(
            self, text="GÖREV PLANI",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#9CA3AF", "#6B7280"),
        )
        header.grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))

        self._scroll = ctk.CTkScrollableFrame(self, label_text="")
        self._scroll.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._scroll.grid_columnconfigure(0, weight=1)

        # Approval buttons (hidden by default)
        self._approval_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._approval_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        self._approval_frame.grid_columnconfigure(0, weight=1)
        self._approval_frame.grid_columnconfigure(1, weight=1)

        self._approve_btn = ctk.CTkButton(
            self._approval_frame,
            text="✓ Onayla",
            fg_color=("#16A34A", "#15803D"),
            hover_color=("#15803D", "#166534"),
            command=self._on_approve_click,
        )
        self._approve_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self._reject_btn = ctk.CTkButton(
            self._approval_frame,
            text="✗ İptal",
            fg_color=("#DC2626", "#B91C1C"),
            hover_color=("#B91C1C", "#991B1B"),
            command=self._on_reject_click,
        )
        self._reject_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self._approval_frame.grid_remove()  # hidden initially

    def set_approval_callbacks(self, on_approve: Callable, on_reject: Callable) -> None:
        self._on_approve = on_approve
        self._on_reject = on_reject

    def show_approval(self) -> None:
        self._approval_frame.grid()

    def hide_approval(self) -> None:
        self._approval_frame.grid_remove()

    def _on_approve_click(self) -> None:
        self.hide_approval()
        if self._on_approve:
            self._on_approve()

    def _on_reject_click(self) -> None:
        self.hide_approval()
        if self._on_reject:
            self._on_reject()

    def load_plan(self, task_list: TaskList) -> None:
        for w in self._scroll.winfo_children():
            w.destroy()
        self._items.clear()

        for task in task_list.tasks:
            item = TaskItem(
                self._scroll,
                task_id=task.id,
                tool=task.tool,
                description=task.description,
            )
            item.pack(fill="x", pady=2, padx=4)
            self._items[task.id] = item

        self.show_approval()

    def update_task(self, execution: TaskExecution) -> None:
        item = self._items.get(execution.task.id)
        if item is None:
            # ReAct mode: tasks arrive dynamically — create the item on first update
            item = TaskItem(
                self._scroll,
                task_id=execution.task.id,
                tool=execution.task.tool,
                description=execution.task.description,
            )
            item.pack(fill="x", pady=2, padx=4)
            self._items[execution.task.id] = item
        item.set_status(execution.status, elapsed=execution.elapsed)

    def clear(self) -> None:
        for w in self._scroll.winfo_children():
            w.destroy()
        self._items.clear()
        self.hide_approval()
