from __future__ import annotations

import re
import customtkinter as ctk

_INJECTION_MARKER = "[⚠ INJECTION WARNING"
_MAX_HEIGHT = 220
_LINE_H = 15

# Matches: "50.0%", "50%", "[download]  50.3% of", "Step 3/5", "#5 [3/7]"
_PCT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
_STEP_RE = re.compile(r"(?:step\s+|#\d+\s+\[)(\d+)/(\d+)", re.IGNORECASE)

# Keywords that trigger the progress bar (long-running operations)
_PROGRESS_KEYWORDS = (
    "[download]", "downloading", "transcrib", "building", "pushing",
    "pulling", "extracting", "uploading", "encoding", "converting",
    "indiriliyor", "yükleniyor", "oluşturuluyor",
)


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

        self._header = ctk.CTkLabel(
            self,
            text="⚙  araç çağrıları",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("#6B7280", "#4B5563"),
        )
        self._header.pack(anchor="w", padx=10, pady=(6, 2))

        # Progress row (hidden by default)
        self._prog_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._prog_bar = ctk.CTkProgressBar(
            self._prog_frame,
            width=200, height=8,
            progress_color=("#3B82F6", "#60A5FA"),
        )
        self._prog_bar.set(0)
        self._prog_bar.pack(side="left", padx=(0, 8))

        self._prog_label = ctk.CTkLabel(
            self._prog_frame,
            text="",
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color=("#94A3B8", "#64748B"),
        )
        self._prog_label.pack(side="left")
        # prog_frame stays hidden until progress detected

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

        self._spinning = False
        self._spin_after: str | None = None

    # ── Progress helpers ───────────────────────────────────────────────

    def _show_progress_bar(self) -> None:
        if not self._prog_frame.winfo_ismapped():
            self._prog_frame.pack(anchor="w", padx=10, pady=(0, 4))

    def _set_progress(self, pct: float, label: str = "") -> None:
        """Update deterministic progress bar (0.0 – 1.0)."""
        self._stop_spin()
        self._show_progress_bar()
        self._prog_bar.configure(mode="determinate")
        self._prog_bar.set(pct)
        self._prog_label.configure(text=label)

    def start_spin(self) -> None:
        """Start indeterminate spinner animation."""
        if self._spinning:
            return
        self._spinning = True
        self._show_progress_bar()
        self._prog_bar.configure(mode="indeterminate")
        self._prog_bar.start()
        self._prog_label.configure(text="çalışıyor…")

    def _stop_spin(self) -> None:
        if not self._spinning:
            return
        self._spinning = False
        self._prog_bar.stop()
        self._prog_bar.configure(mode="determinate")

    def finish(self) -> None:
        """Mark operation as complete."""
        self._stop_spin()
        if self._prog_frame.winfo_ismapped():
            self._prog_bar.set(1.0)
            self._prog_label.configure(text="✓ tamamlandı")

    # ── Text append ────────────────────────────────────────────────────

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

        # Auto-detect progress from text
        lower = text.lower()

        # Detect percentage (e.g. yt-dlp, docker)
        m = _PCT_RE.search(text)
        if m:
            pct = float(m.group(1)) / 100.0
            self._set_progress(min(pct, 1.0), m.group(0).strip())
            return

        # Detect step fraction (e.g. docker "#5 [3/7]", "Step 3/5")
        m2 = _STEP_RE.search(text)
        if m2:
            cur, total = int(m2.group(1)), int(m2.group(2))
            if total > 0:
                self._set_progress(cur / total, f"{cur}/{total}")
                return

        # Detect long-running keyword → start spinner if not already showing progress
        if any(kw in lower for kw in _PROGRESS_KEYWORDS):
            if not self._prog_frame.winfo_ismapped():
                self.start_spin()
