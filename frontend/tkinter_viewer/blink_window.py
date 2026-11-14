"""Separate Tkinter window that flips WCS-aligned frames (blink comparison).

It runs its own background thread and polling loop, so the main viewer only
has to call ``open_blink_window``.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spherex.blink import BlinkError, prepare_blink

DEFAULT_INTERVAL_MS = 700
MIN_INTERVAL_MS = 150
MAX_INTERVAL_MS = 2500
POLL_MS = 100


class BlinkWindow:
    """Shows aligned frames one after another, plus an optional difference image."""

    def __init__(
        self,
        root: Any,
        service: Any,
        observations: Sequence[Mapping[str, Any]],
        ra: float,
        dec: float,
        rendering: Mapping[str, Any],
        output_dir: Path,
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.window = tk.Toplevel(root)
        self.window.title(f"SPHEREx Blink ({len(observations)} frames)")
        self.window.geometry("760x780")
        self.window.minsize(560, 600)
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._images: list[Any] = []
        self._difference: Any = None
        self._result: dict[str, Any] | None = None
        self._index = 0
        self._playing = False
        self._timer: str | None = None

        self.status = tk.StringVar(value="Downloading / aligning frames on the sky grid...")
        ttk.Label(self.window, textvariable=self.status, anchor="w", wraplength=720).pack(fill="x", padx=10, pady=(8, 2))
        self.image_label = ttk.Label(self.window, anchor="center")
        self.image_label.pack(fill="both", expand=True, padx=8, pady=4)
        self.caption = tk.StringVar(value="")
        ttk.Label(self.window, textvariable=self.caption, justify="left", wraplength=720).pack(fill="x", padx=10)

        controls = ttk.Frame(self.window)
        controls.pack(fill="x", padx=10, pady=8)
        self.prev_button = ttk.Button(controls, text="◀ Prev", command=lambda: self._step(-1), state="disabled")
        self.prev_button.pack(side="left")
        self.play_button = ttk.Button(controls, text="▶ Play", command=self._toggle_play, state="disabled")
        self.play_button.pack(side="left", padx=6)
        self.next_button = ttk.Button(controls, text="Next ▶", command=lambda: self._step(1), state="disabled")
        self.next_button.pack(side="left")
        self.show_difference = tk.BooleanVar(value=False)
        self.difference_check = ttk.Checkbutton(
            controls, text="Difference (last − first)", variable=self.show_difference,
            command=self._show_current, state="disabled",
        )
        self.difference_check.pack(side="left", padx=12)
        # Slider value is "speed": right = faster. It is turned into a delay in _interval_ms.
        ttk.Label(controls, text="Slow").pack(side="left", padx=(8, 2))
        self.speed = tk.DoubleVar(value=MIN_INTERVAL_MS + MAX_INTERVAL_MS - DEFAULT_INTERVAL_MS)
        ttk.Scale(
            controls, from_=MIN_INTERVAL_MS, to=MAX_INTERVAL_MS, variable=self.speed, orient="horizontal", length=140,
        ).pack(side="left")
        ttk.Label(controls, text="Fast").pack(side="left", padx=(2, 0))

        self.warning = tk.StringVar(value="")
        ttk.Label(self.window, textvariable=self.warning, foreground="#b45309", wraplength=720, justify="left").pack(
            fill="x", padx=10, pady=(0, 8)
        )

        self.window.protocol("WM_DELETE_WINDOW", self._close)
        records = [dict(observation) for observation in observations]
        threading.Thread(
            target=self._work,
            args=(service, records, ra, dec, dict(rendering), output_dir),
            name="spherex-blink",
            daemon=True,
        ).start()
        self.window.after(POLL_MS, self._poll)

    def _work(self, service, records, ra, dec, rendering, output_dir) -> None:
        mode = str(rendering.get("mode", "")).casefold()
        colormap = "gray" if mode == "greyscale" else str(rendering.get("colormap") or "inferno")
        try:
            result = prepare_blink(
                service, records, ra, dec, output_dir,
                colormap=colormap,
                stretch=str(rendering.get("stretch") or "asinh"),
                show_grid=bool(rendering.get("show_grid", True)),
                show_target=bool(rendering.get("show_target", True)),
                progress=lambda message: self._queue.put(("progress", message)),
            )
            self._queue.put(("result", result))
        except (BlinkError, ValueError) as exc:
            self._queue.put(("error", str(exc)))
        except Exception as exc:
            self._queue.put(("error", f"Unexpected blink failure: {exc}"))

    def _poll(self) -> None:
        if not self.window.winfo_exists():
            return
        while True:
            try:
                kind, payload = self._queue.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                self.status.set(str(payload))
            elif kind == "result":
                self._load(payload)
                return
            else:
                self.status.set(f"Blink failed: {payload}")
                return
        self.window.after(POLL_MS, self._poll)

    def _load(self, result: dict[str, Any]) -> None:
        import tkinter as tk

        self._result = result
        try:
            self._images = [tk.PhotoImage(master=self.window, file=path) for path in result["frame_paths"]]
            self._difference = tk.PhotoImage(master=self.window, file=result["difference_path"])
        except tk.TclError as exc:
            self.status.set(f"Blink images could not be displayed: {exc}")
            return
        for widget in (self.prev_button, self.play_button, self.next_button, self.difference_check):
            widget.configure(state="normal")
        self.status.set(
            f"{len(self._images)} frames aligned to {result.get('reference_observation_id')} "
            f"(common sky {result.get('overlap_fraction', 0.0):.0%}). Same brightness scale for all frames; "
            "fixed stars should stay still."
        )
        self.warning.set(self._warning_text(result))
        self._index = 0
        self._show_current()
        self._toggle_play()

    @staticmethod
    def _warning_text(result: Mapping[str, Any]) -> str:
        wavelength = result.get("wavelength") or {}
        lines = []
        if wavelength.get("comparable") is not True:
            lines.append("⚠ " + str(wavelength.get("note") or "Wavelength match unknown."))
        for reason in result.get("skipped") or []:
            lines.append("⚠ Skipped " + str(reason))
        if len(wavelength.get("detectors") or []) > 1:
            lines.append("⚠ Frames come from different detectors: " + ", ".join(wavelength["detectors"]) + ".")
        return "\n".join(lines)

    def _show_current(self) -> None:
        if not self._images or self._result is None:
            return
        if self.show_difference.get():
            self.image_label.configure(image=self._difference)
            self.caption.set("Difference: last frame minus first. Fixed stars cancel; a moving object leaves a dark/bright pair.")
            return
        self.image_label.configure(image=self._images[self._index])
        meta = self._result["metadata"][self._index]
        wavelength = meta.get("wavelength_um")
        wavelength_text = f"{wavelength:.3f} µm" if isinstance(wavelength, (int, float)) else "wavelength unknown"
        self.caption.set(
            f"Frame {self._index + 1}/{len(self._images)}  |  {meta.get('observation_timestamp') or 'date unknown'}  |  "
            f"{meta.get('observation_id')}  |  D{meta.get('detector') or '-'}  |  {wavelength_text}"
        )

    def _step(self, delta: int) -> None:
        if not self._images:
            return
        self._index = (self._index + delta) % len(self._images)
        self._show_current()

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        self.play_button.configure(text="⏸ Pause" if self._playing else "▶ Play")
        if self._playing:
            self._tick()
        elif self._timer is not None:
            self.window.after_cancel(self._timer)
            self._timer = None

    def _tick(self) -> None:
        if not self._playing or not self.window.winfo_exists():
            return
        if not self.show_difference.get():
            self._step(1)
        self._timer = self.window.after(self._interval_ms(), self._tick)

    def _interval_ms(self) -> int:
        """Delay between frames: the further right the slider, the shorter the delay."""
        delay = MIN_INTERVAL_MS + MAX_INTERVAL_MS - float(self.speed.get())
        return int(min(MAX_INTERVAL_MS, max(MIN_INTERVAL_MS, delay)))

    def _close(self) -> None:
        self._playing = False
        if self._timer is not None:
            self.window.after_cancel(self._timer)
        self.window.destroy()


def open_blink_window(
    root: Any,
    service: Any,
    observations: Sequence[Mapping[str, Any]],
    ra: float,
    dec: float,
    rendering: Mapping[str, Any],
) -> BlinkWindow:
    """Open a blink window next to the main viewer; PNGs go beside the previews."""
    output_dir = Path(service.processor.preview_dir) / "blink"
    return BlinkWindow(root, service, observations, ra, dec, rendering, output_dir)
