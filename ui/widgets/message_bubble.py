from __future__ import annotations

import tkinter as tk
import customtkinter as ctk


class MessageBubble(ctk.CTkFrame):
    """A single chat message bubble with selectable/copyable text."""

    def __init__(self, parent, role: str, content: str, **kwargs):
        super().__init__(parent, **kwargs)
        self._role = role

        is_user = role == "user"
        bg_color = ("#3B82F6", "#2563EB") if is_user else ("#374151", "#1F2937")
        text_color = ("#FFFFFF", "#F9FAFB")
        anchor = "e" if is_user else "w"

        self.configure(fg_color="transparent")

        bubble = ctk.CTkFrame(self, fg_color=bg_color, corner_radius=12)

        role_label = ctk.CTkLabel(
            bubble,
            text="You" if is_user else "Agent",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("#BFDBFE", "#93C5FD") if is_user else ("#9CA3AF", "#6B7280"),
        )
        role_label.pack(anchor="w", padx=10, pady=(8, 0))

        # CTkTextbox: read-only, seçilebilir, kopyalanabilir
        self._textbox = ctk.CTkTextbox(
            bubble,
            font=ctk.CTkFont(size=13),
            wrap="word",
            activate_scrollbars=False,
            height=1,           # _resize() ile dinamik boyutlandırılır
            border_width=0,
            fg_color=bg_color,
            text_color=text_color,
        )
        self._textbox.pack(anchor="w", padx=8, pady=(2, 8), fill="x", expand=True)

        # İçeriği yerleştir, sonra read-only yap
        if content:
            self._textbox.insert("1.0", content)
        self._textbox.configure(state="disabled")
        self._textbox.bind("<Configure>", self._resize)

        bubble.pack(anchor=anchor, padx=8, pady=4, fill="x")

    def _resize(self, event=None) -> None:
        """Textbox yüksekliğini içeriğe göre ayarla."""
        if getattr(self, "_resizing", False):
            return
        self._resizing = True
        try:
            lines = int(self._textbox.index("end-1c").split(".")[0])
            self._textbox.configure(height=max(1, lines) * 20 + 4)
        except Exception:
            pass
        finally:
            self._resizing = False

    def append(self, text: str) -> None:
        self._textbox.configure(state="normal")
        self._textbox.insert("end", text)
        self._textbox.configure(state="disabled")
        self._resize()
