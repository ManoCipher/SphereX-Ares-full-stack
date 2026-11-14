"""Tkinter viewer for SPHEREx metadata and selected local FITS cutouts."""

from __future__ import annotations

import json
import math
import queue
import shutil
import sys
import threading
import webbrowser
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

# Allow ``python -m frontend.tkinter_viewer`` from a fresh checkout while also
# supporting an editable installation of the backend package.
BACKEND_SRC = Path(__file__).resolve().parents[2] / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from spherex.observation_service import ObservationPreviewService
from spherex.spherex_client import SPHERExArchiveError, SPHERExClient, SPHERExObservation


def _display(value: Any) -> str:
    return "" if value is None else str(value)


def _wavelength_label(observation: SPHERExObservation) -> str:
    lower = observation["wavelength_min_um"]
    upper = observation["wavelength_max_um"]
    if lower is None and upper is None:
        return ""
    if lower is None:
        return f"\u2264 {upper:.3f}"
    if upper is None:
        return f"\u2265 {lower:.3f}"
    return f"{lower:.3f}\u2013{upper:.3f}"


def _preview_identity(record: Mapping[str, Any]) -> str:
    """Keep detector products distinct even when an exposure ID is shared."""
    return "|".join(
        str(record.get(key) or "")
        for key in ("observation_id", "detector", "fits_filename")
    )


def _size_label(size_bytes: int | None) -> str:
    if size_bytes is None:
        return ""
    size = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return ""


class SPHERExViewer:
    """Desktop archive browser with an asynchronous selected-image gallery."""

    COLUMNS = (
        "selected",
        "observation_timestamp",
        "observation_id",
        "detector",
        "band",
        "ra",
        "dec",
        "exposure_time_s",
        "download_status",
    )
    HEADINGS = {
        "selected": "Image",
        "observation_timestamp": "Observed (UTC)",
        "observation_id": "Observation ID",
        "detector": "Detector",
        "band": "Band",
        "ra": "RA (deg)",
        "dec": "Dec (deg)",
        "exposure_time_s": "Exposure (s)",
        "download_status": "FITS status",
    }

    def __init__(
        self,
        root: Any,
        client: SPHERExClient | None = None,
        preview_service: ObservationPreviewService | None = None,
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.client = client or SPHERExClient()
        self.preview_service = preview_service or ObservationPreviewService()
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._records: dict[str, SPHERExObservation] = {}
        self._checked: set[str] = set()
        self._sort_reverse: dict[str, bool] = {}
        self._gallery_images: list[Any] = []
        self._preview_results: dict[str, dict[str, Any]] = {}
        self._rgb_result: dict[str, Any] | None = None

        root.title("SPHEREx FITS Image Viewer")
        root.geometry("1400x920")
        root.minsize(980, 680)

        search = ttk.LabelFrame(root, text="IRSA coordinate search", padding=(6, 4))
        search.pack(fill="x", padx=8, pady=(6, 2))
        defaults = {
            "RA (deg)": "133.795",
            "Dec (deg)": "-7.245",
            "Radius (deg)": "0.05",
            "Max results": "50",
            "Start date": "",
            "End date": "",
        }
        self.entries: dict[str, Any] = {}
        for column, (label, default) in enumerate(defaults.items()):
            label_column = column * 2
            ttk.Label(search, text=label).grid(row=0, column=label_column, sticky="e", padx=(4, 2))
            entry = ttk.Entry(search, width=11)
            entry.insert(0, default)
            entry.grid(row=0, column=label_column + 1, sticky="ew", padx=(0, 5))
            self.entries[label] = entry
            search.columnconfigure(label_column + 1, weight=1)
        self.search_button = ttk.Button(search, text="Search IRSA", command=self.search)
        self.search_button.grid(row=0, column=len(defaults) * 2, padx=(8, 4), sticky="ew")
        root.bind("<Return>", lambda _event: self.search())

        display = ttk.LabelFrame(root, text="Image display", padding=(6, 4))
        display.pack(fill="x", padx=8, pady=2)
        self.display_mode = tk.StringVar(value="False Colour")
        self.colour_palette = tk.StringVar(value="Inferno")
        self.brightness_stretch = tk.StringVar(value="Asinh")
        self.show_grid = tk.BooleanVar(value=True)
        self.show_target = tk.BooleanVar(value=True)

        ttk.Label(display, text="Mode").grid(row=0, column=0, sticky="e", padx=(4, 2))
        mode_combo = ttk.Combobox(
            display,
            textvariable=self.display_mode,
            values=("Greyscale", "False Colour", "RGB Composite"),
            state="readonly",
            width=16,
        )
        mode_combo.grid(row=0, column=1, sticky="w", padx=(0, 8))
        mode_combo.bind("<<ComboboxSelected>>", self._display_mode_changed)
        ttk.Label(display, text="Palette").grid(row=0, column=2, sticky="e", padx=(4, 2))
        self.palette_combo = ttk.Combobox(
            display,
            textvariable=self.colour_palette,
            values=("Gray", "Inferno", "Viridis", "Magma", "Plasma"),
            state="readonly",
            width=13,
        )
        self.palette_combo.grid(row=0, column=3, sticky="w", padx=(0, 8))
        ttk.Label(display, text="Stretch").grid(row=0, column=4, sticky="e", padx=(4, 2))
        ttk.Combobox(
            display,
            textvariable=self.brightness_stretch,
            values=("Linear", "Asinh"),
            state="readonly",
            width=11,
        ).grid(row=0, column=5, sticky="w", padx=(0, 8))
        ttk.Checkbutton(display, text="Show RA/Dec grid", variable=self.show_grid).grid(
            row=0, column=6, sticky="w", padx=4
        )
        ttk.Checkbutton(display, text="Show target position", variable=self.show_target).grid(
            row=0, column=7, sticky="w", padx=4
        )
        self.mapping_note = ttk.Label(
            display,
            text="RGB: wavelength-order mapping when available; otherwise selected table order maps to R, G, B.",
        )
        self.mapping_note.grid(row=0, column=8, sticky="e", padx=(12, 4))
        display.columnconfigure(8, weight=1)

        status_row = ttk.Frame(root)
        status_row.pack(fill="x", padx=10, pady=2)
        self.status = tk.StringVar(value="Search a sky position, then check observations to preview.")
        ttk.Label(status_row, textvariable=self.status, anchor="w").pack(side="left", fill="x", expand=True)
        self.preview_button = ttk.Button(
            status_row, text="Show Selected Images", command=self.show_selected_images, state="disabled"
        )
        self.preview_button.pack(side="right", padx=(8, 0))
        self.details_button = ttk.Button(
            status_row, text="View Selected Details", command=self.show_selected_details, state="disabled"
        )
        self.details_button.pack(side="right", padx=(8, 0))

        split = ttk.Panedwindow(root, orient="vertical")
        split.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        table_frame = ttk.LabelFrame(split, text="Observations (click the Image box to select up to six)", padding=5)
        split.add(table_frame, weight=1)
        self.table = ttk.Treeview(table_frame, columns=self.COLUMNS, show="headings", selectmode="browse")
        widths = (55, 165, 235, 70, 100, 95, 95, 90, 120)
        for column, width in zip(self.COLUMNS, widths, strict=True):
            self.table.heading(column, text=self.HEADINGS[column], command=lambda selected=column: self._sort(selected))
            self.table.column(column, width=width, minwidth=50, stretch=column not in {"selected", "detector"})
        table_y = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        table_x = ttk.Scrollbar(table_frame, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=table_y.set, xscrollcommand=table_x.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        table_y.grid(row=0, column=1, sticky="ns")
        table_x.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.table.bind("<Button-1>", self._table_click, add=True)

        gallery_box = ttk.LabelFrame(split, text="Selected target-centred previews", padding=6)
        split.add(gallery_box, weight=5)
        self.gallery_canvas = tk.Canvas(gallery_box, highlightthickness=0)
        gallery_scroll = ttk.Scrollbar(gallery_box, orient="vertical", command=self.gallery_canvas.yview)
        self.gallery_canvas.configure(yscrollcommand=gallery_scroll.set)
        self.gallery_canvas.pack(side="left", fill="both", expand=True)
        gallery_scroll.pack(side="right", fill="y")
        self.gallery = ttk.Frame(self.gallery_canvas)
        self._gallery_window = self.gallery_canvas.create_window((0, 0), window=self.gallery, anchor="nw")
        self.gallery.bind("<Configure>", lambda _e: self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all")))
        self.gallery_canvas.bind(
            "<Configure>", lambda event: self.gallery_canvas.itemconfigure(self._gallery_window, width=event.width)
        )
        ttk.Label(self.gallery, text="No previews generated yet.").grid(row=0, column=0, padx=20, pady=25)

        root.after(100, self._poll_results)

    def search(self) -> None:
        if str(self.search_button["state"]) == "disabled":
            return
        try:
            arguments = {
                "ra": float(self.entries["RA (deg)"].get()),
                "dec": float(self.entries["Dec (deg)"].get()),
                "radius": float(self.entries["Radius (deg)"].get()),
                "max_results": int(self.entries["Max results"].get()),
                "start_date": self.entries["Start date"].get().strip() or None,
                "end_date": self.entries["End date"].get().strip() or None,
            }
            self.client._validate_position(arguments["ra"], arguments["dec"], arguments["radius"])
            if arguments["max_results"] <= 0:
                raise ValueError("Max results must be positive")
        except ValueError as exc:
            self.status.set(f"Invalid search: {exc}")
            return
        self.search_button.configure(state="disabled")
        self.preview_button.configure(state="disabled")
        self.details_button.configure(state="disabled")
        self.status.set("Searching NASA/IPAC IRSA...")
        self._replace_records([])

        def run() -> None:
            try:
                self._queue.put(("search_results", self.client.search(**arguments)))
            except (SPHERExArchiveError, ValueError) as exc:
                self._queue.put(("search_error", str(exc)))
            except Exception as exc:
                self._queue.put(("search_error", f"Unexpected search failure: {exc}"))

        threading.Thread(target=run, name="spherex-irsa-search", daemon=True).start()

    def _display_mode_changed(self, _event: Any = None) -> None:
        rgb_mode = self.display_mode.get() == "RGB Composite"
        greyscale = self.display_mode.get() == "Greyscale"
        self.preview_button.configure(text="Generate RGB Composite" if rgb_mode else "Show Selected Images")
        self.palette_combo.configure(state="disabled" if rgb_mode or greyscale else "readonly")

    def _rendering_settings(self) -> dict[str, Any]:
        mode = self.display_mode.get()
        return {
            "mode": mode.casefold().replace(" ", "_"),
            "colormap": self.colour_palette.get().casefold(),
            "stretch": self.brightness_stretch.get().casefold(),
            "show_grid": bool(self.show_grid.get()),
            "show_target": bool(self.show_target.get()),
        }

    def show_selected_images(self) -> None:
        selected = [self._records[item] for item in self.table.get_children("") if item in self._checked]
        if not selected:
            self.status.set("Select at least one observation using its Image checkbox.")
            return
        rgb_mode = self.display_mode.get() == "RGB Composite"
        if rgb_mode and len(selected) != 3:
            self.status.set(f"RGB Composite requires exactly 3 observations; currently selected: {len(selected)}.")
            return
        if not rgb_mode and len(selected) > 6:
            self.status.set(f"{len(selected)} observations selected. Select no more than 6.")
            return
        try:
            ra = float(self.entries["RA (deg)"].get())
            dec = float(self.entries["Dec (deg)"].get())
            self.client._validate_position(ra, dec, 0.01)
        except ValueError as exc:
            self.status.set(f"Invalid target coordinates: {exc}")
            return
        self.preview_button.configure(state="disabled")
        action = "aligning 3 RGB channels" if rgb_mode else f"processing {len(selected)} selected observation(s)"
        self.status.set(f"Downloading missing FITS files and {action}...")
        for item in self._checked:
            if self.table.exists(item):
                self.table.set(item, "download_status", "Downloading / processing...")
        settings = self._rendering_settings()

        def run() -> None:
            try:
                if rgb_mode:
                    result = self.preview_service.create_rgb_composite(
                        selected, ra, dec, size=200, rendering=settings
                    )
                    self._queue.put(("rgb_result", result))
                else:
                    results = self.preview_service.prepare_observation_previews(
                        selected, ra, dec, size=200, rendering=settings
                    )
                    self._queue.put(("preview_results", results))
            except Exception as exc:
                self._queue.put(("preview_error", str(exc)))

        threading.Thread(target=run, name="spherex-fits-preview", daemon=True).start()

    def _poll_results(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "search_results":
                    self.search_button.configure(state="normal")
                    self._replace_records(payload)
                    self.status.set(f"Found {len(payload)} observation(s). Check rows to compare images.")
                elif kind == "search_error":
                    self.search_button.configure(state="normal")
                    self.status.set(f"Search failed: {payload}")
                elif kind == "preview_results":
                    self.preview_button.configure(state="normal")
                    self._render_gallery(payload)
                    successes = sum(result["status"] == "success" for result in payload)
                    self.status.set(
                        f"Prepared {successes}/{len(payload)} preview(s). Use View Selected Details for all metadata."
                    )
                elif kind == "rgb_result":
                    self.preview_button.configure(state="normal")
                    self._render_rgb_composite(payload)
                    self.status.set("RGB composite generated. Channel assignments are shown below the image.")
                elif kind == "preview_error":
                    self.preview_button.configure(state="normal")
                    self.status.set(f"Preview generation failed: {payload}")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_results)

    def _replace_records(self, observations: list[SPHERExObservation]) -> None:
        self.table.delete(*self.table.get_children())
        self._records.clear()
        self._checked.clear()
        self._preview_results.clear()
        self._rgb_result = None
        self.details_button.configure(state="disabled")
        self.preview_service.refresh_files()
        for index, observation in enumerate(observations):
            item_id = f"observation-{index}"
            self._records[item_id] = observation
            self.table.insert(
                "",
                "end",
                iid=item_id,
                values=(
                    "\u2610",
                    _display(observation["observation_timestamp"]),
                    _display(observation["observation_id"]),
                    _display(observation["detector"]),
                    _display(observation["band"] or _wavelength_label(observation)),
                    _display(observation["ra"]),
                    _display(observation["dec"]),
                    _display(observation["exposure_time_s"]),
                    self.preview_service.download_status(observation),
                ),
            )
        self.preview_button.configure(state="normal" if observations else "disabled")

    def _table_click(self, event: Any) -> None:
        if self.table.identify_region(event.x, event.y) != "cell" or self.table.identify_column(event.x) != "#1":
            return
        item = self.table.identify_row(event.y)
        if not item:
            return
        if item in self._checked:
            self._checked.remove(item)
            self.table.set(item, "selected", "\u2610")
        else:
            self._checked.add(item)
            self.table.set(item, "selected", "\u2611")
        count = len(self._checked)
        self.details_button.configure(state="normal" if count else "disabled")
        if count > 6:
            self.status.set(f"{count} observations selected; the preview limit is 6.")
        else:
            self.status.set(f"{count} observation(s) selected for preview.")

    def _render_gallery(self, results: list[dict[str, Any]]) -> None:
        import tkinter as tk
        from tkinter import ttk

        for child in self.gallery.winfo_children():
            child.destroy()
        self._gallery_images.clear()
        self._preview_results = {
            _preview_identity(result): result for result in results if result.get("observation_id")
        }
        for column in range(3):
            self.gallery.columnconfigure(column, weight=1)
        for index, result in enumerate(results):
            row, column = divmod(index, 3)
            card = ttk.LabelFrame(self.gallery, text=_display(result.get("observation_id")) or "Observation", padding=4)
            card.grid(row=row, column=column, padx=4, pady=3, sticky="nsew")
            image_path = result.get("preview_path") if result.get("status") == "success" else None
            if image_path:
                try:
                    image = self._load_photo_image(image_path, 300, 225)
                    self._gallery_images.append(image)
                    image_label = ttk.Label(card, image=image)
                    image_label.pack()
                except Exception as exc:
                    ttk.Label(card, text=f"Preview could not be displayed:\n{exc}", justify="center").pack(pady=25)
            else:
                ttk.Label(card, text=result.get("error") or result.get("status", "Failed"), wraplength=260, justify="center").pack(pady=35)
            summary = (
                f"{result.get('observation_date') or 'Date unavailable'}\n"
                f"D{result.get('detector') or '-'}  {result.get('band') or '-'}  |  "
                f"Target {result.get('target_ra'):.5f}, {result.get('target_dec'):.5f}\n"
                f"{(result.get('rendering') or {}).get('stretch', '-')} / "
                f"{(result.get('rendering') or {}).get('colormap', '-')}  |  {result.get('status')}"
            )
            label = ttk.Label(card, text=summary, justify="left")
            label.pack(anchor="w", pady=(3, 0))
            matching = next(
                (item for item, record in self._records.items() if record["observation_id"] == result.get("observation_id")),
                None,
            )
            if matching:
                self.table.set(matching, "download_status", "Preview ready" if result["status"] == "success" else result["status"])

    @staticmethod
    def _load_photo_image(path: str, max_width: int, max_height: int) -> Any:
        import tkinter as tk

        image = tk.PhotoImage(file=path)
        factor = max(
            1,
            math.ceil(image.width() / max_width),
            math.ceil(image.height() / max_height),
        )
        return image.subsample(factor) if factor > 1 else image

    def _render_rgb_composite(self, result: dict[str, Any]) -> None:
        from tkinter import ttk

        for child in self.gallery.winfo_children():
            child.destroy()
        self._gallery_images.clear()
        self._rgb_result = result
        card = ttk.LabelFrame(
            self.gallery,
            text="Representative false-colour SPHEREx infrared composite",
            padding=8,
        )
        card.grid(row=0, column=0, columnspan=3, padx=8, pady=8, sticky="nsew")
        try:
            image = self._load_photo_image(str(result["preview_path"]), 900, 600)
            self._gallery_images.append(image)
            ttk.Label(card, image=image).grid(row=0, column=0, columnspan=3, pady=(0, 8))
        except Exception as exc:
            ttk.Label(card, text=f"Composite could not be displayed:\n{exc}").grid(row=0, column=0)

        channel_lines: list[str] = []
        for channel in ("R", "G", "B"):
            metadata = result.get("channels", {}).get(channel, {})
            wavelength = metadata.get("wavelength_um")
            wavelength_label = f"{wavelength:.3f} um" if wavelength is not None else "wavelength unavailable"
            channel_lines.append(
                f"{channel}: {metadata.get('observation_id') or '-'} | detector {metadata.get('detector') or '-'} | "
                f"{wavelength_label} ({metadata.get('wavelength_source') or 'manual mapping'})"
            )
            matching = next(
                (
                    item
                    for item, record in self._records.items()
                    if record["observation_id"] == metadata.get("observation_id")
                ),
                None,
            )
            if matching:
                self.table.set(matching, "download_status", f"RGB channel {channel}")
        description = (
            "\n".join(channel_lines)
            + f"\nMapping: {result.get('mapping_note')}"
            + f"\nCommon valid footprint: {result.get('overlap_fraction', 0.0):.1%}"
            + f"\n{result.get('calibration_note')}"
        )
        ttk.Label(card, text=description, justify="left", wraplength=1000).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        ttk.Button(card, text="View Larger", command=self._show_large_composite).grid(
            row=2, column=1, sticky="e", padx=4
        )
        ttk.Button(card, text="Save PNG As...", command=self._save_rgb_composite).grid(
            row=2, column=2, sticky="e", padx=4
        )
        card.columnconfigure(0, weight=1)

    def _show_large_composite(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        if not self._rgb_result or not self._rgb_result.get("preview_path"):
            return
        window = tk.Toplevel(self.root)
        window.title("SPHEREx RGB Composite")
        window.geometry("1000x820")
        image = self._load_photo_image(str(self._rgb_result["preview_path"]), 960, 760)
        label = ttk.Label(window, image=image)
        label.pack(fill="both", expand=True, padx=8, pady=8)
        window._composite_image = image

    def _save_rgb_composite(self) -> None:
        from tkinter import filedialog

        if not self._rgb_result or not self._rgb_result.get("preview_path"):
            return
        destination = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save RGB composite",
            defaultextension=".png",
            filetypes=(("PNG image", "*.png"),),
        )
        if destination:
            shutil.copy2(str(self._rgb_result["preview_path"]), destination)
            self.status.set(f"Saved RGB composite to {destination}")

    def show_selected_details(self) -> None:
        """Open all checked observation and processed-image details together."""
        import tkinter as tk
        from tkinter import ttk

        selected = [
            self._records[item]
            for item in self.table.get_children("")
            if item in self._checked
        ]
        if not selected:
            self.status.set("Select at least one observation to view its details.")
            return

        window = tk.Toplevel(self.root)
        window.title(f"Selected Observation / Image Details ({len(selected)})")
        window.geometry("1150x800")
        window.minsize(760, 500)

        canvas = tk.Canvas(window, highlightthickness=0)
        scroll = ttk.Scrollbar(window, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        content = ttk.Frame(canvas, padding=8)
        content_window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(content_window, width=event.width))

        detail_images: list[Any] = []
        for index, observation in enumerate(selected):
            observation_id = str(observation.get("observation_id") or "Unknown observation")
            preview = next(
                (
                    result for result in self._preview_results.values()
                    if result.get("observation_id") == observation.get("observation_id")
                    and str(result.get("detector") or "") == str(observation.get("detector") or "")
                ),
                None,
            )
            card = ttk.LabelFrame(content, text=observation_id, padding=8)
            card.grid(row=index, column=0, padx=4, pady=6, sticky="ew")
            card.columnconfigure(1, weight=1)

            image_path = preview.get("preview_path") if preview and preview.get("status") == "success" else None
            if image_path:
                try:
                    image = self._load_photo_image(image_path, 300, 300)
                    detail_images.append(image)
                    ttk.Label(card, image=image).grid(row=0, column=0, rowspan=2, padx=(0, 10), sticky="n")
                except Exception as exc:
                    ttk.Label(card, text=f"Preview unavailable\n{exc}", justify="center").grid(
                        row=0, column=0, rowspan=2, padx=(0, 10), sticky="n"
                    )
            else:
                message = preview.get("error") if preview else "Preview has not been generated yet."
                ttk.Label(card, text=message, wraplength=220, justify="center").grid(
                    row=0, column=0, rowspan=2, padx=(0, 10), sticky="n"
                )

            payload = {
                "observation": observation,
                "image_processing": preview,
                "rgb_composite": self._rgb_result,
            }
            details = tk.Text(card, wrap="word", height=15, width=88)
            details.insert("1.0", json.dumps(payload, indent=2, sort_keys=True, default=str))
            details.configure(state="disabled")
            details.grid(row=0, column=1, sticky="nsew")
            text_scroll = ttk.Scrollbar(card, orient="vertical", command=details.yview)
            details.configure(yscrollcommand=text_scroll.set)
            text_scroll.grid(row=0, column=2, sticky="ns")

            url = observation.get("fits_url")
            if url:
                ttk.Button(
                    card,
                    text="Open FITS URL",
                    command=lambda value=url: webbrowser.open_new_tab(value),
                ).grid(row=1, column=1, sticky="e", pady=(6, 0))

        content.columnconfigure(0, weight=1)
        window._detail_images = detail_images

    def _sort(self, column: str) -> None:
        reverse = self._sort_reverse.get(column, False)
        rows = [(self.table.set(item, column), item) for item in self.table.get_children("")]

        def key(row: tuple[str, str]) -> tuple[bool, Any]:
            value = row[0]
            if not value:
                return True, ""
            try:
                return False, float(value)
            except ValueError:
                return False, value.casefold()

        rows.sort(key=key, reverse=reverse)
        for position, (_value, item) in enumerate(rows):
            self.table.move(item, "", position)
        self._sort_reverse[column] = not reverse


def main(client_factory: Callable[[], SPHERExClient] = SPHERExClient) -> None:
    import tkinter as tk

    root = tk.Tk()
    SPHERExViewer(root, client=client_factory())
    root.mainloop()


if __name__ == "__main__":
    main()
