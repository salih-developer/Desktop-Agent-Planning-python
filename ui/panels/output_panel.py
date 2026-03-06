from __future__ import annotations

import customtkinter as ctk


class OutputPanel(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._build()

    def _build(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(
            self, text="ARAÇ ÇIKTISI",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#9CA3AF", "#6B7280"),
        )
        header.grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))

        self._textbox = ctk.CTkTextbox(
            self,
            font=ctk.CTkFont(family="Consolas", size=12),
            state="disabled",
            wrap="word",
        )
        self._textbox.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        clear_btn = ctk.CTkButton(
            self, text="Clear Output", width=100,
            fg_color="transparent", border_width=1,
            command=self.clear,
        )
        clear_btn.grid(row=2, column=0, sticky="e", padx=8, pady=(0, 8))

    def append(self, text: str) -> None:
        self._textbox.configure(state="normal")
        self._textbox.insert("end", text)
        self._textbox.configure(state="disabled")
        self._textbox.see("end")

    def clear(self) -> None:
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        self._textbox.configure(state="disabled")
