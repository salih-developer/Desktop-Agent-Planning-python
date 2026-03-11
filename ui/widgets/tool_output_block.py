from __future__ import annotations

import customtkinter as ctk

_INJECTION_MARKER = "[⚠ INJECTION WARNING"
_MAX_HEIGHT = 220
_LINE_H = 15


class ToolOutputBlock(ctk.CTkFrame):
    """Compact inline block that shows tool calls and their output inside the chat stream."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            fg_color=("#1A1A2E", "#0D0D1A"),
            corner_radius=8,
            border_width=1,
            border_color=("#2D2D44", "#1A1A2E"),
            **kwargs,
        )

        ctk.CTkLabel(
            self,
            text="⚙  araç çağrıları",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("#6B7280", "#4B5563"),
        ).pack(anchor="w", padx=10, pady=(6, 2))

        self._tb = ctk.CTkTextbox(
            self,
            font=ctk.CTkFont(family="Consolas", size=11),
            height=40,
            state="disabled",
            wrap="word",
            fg_color=("#1A1A2E", "#0D0D1A"),
            text_color=("#94A3B8", "#64748B"),
            border_width=0,
            activate_scrollbars=False,
        )
        self._tb.pack(fill="x", padx=8, pady=(0, 8))
        self._tb._textbox.tag_configure(
            "inj", foreground="#F97316", background="#431407"
        )

    def append(self, text: str) -> None:
        self._tb.configure(state="normal")
        tb = self._tb._textbox
        if _INJECTION_MARKER in text:
            parts = text.split(_INJECTION_MARKER, 1)
            if parts[0]:
                tb.insert("end", parts[0])
            start = tb.index("end-1c")
            tb.insert("end", _INJECTION_MARKER + parts[1])
            tb.tag_add("inj", start, tb.index("end-1c"))
        else:
            tb.insert("end", text)
        self._tb.configure(state="disabled")
        lines = int(tb.index("end-1c").split(".")[0])
        self._tb.configure(height=min(max(40, lines * _LINE_H), _MAX_HEIGHT))
