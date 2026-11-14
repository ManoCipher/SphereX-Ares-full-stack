"""Tkinter viewer for SPHEREx metadata and selected local FITS cutouts."""

from __future__ import annotations

import json
import queue
import sys
import threading
import webbrowser
from collections.abc import Callable
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
        self._detail_image: Any = None

        root.title("SPHEREx FITS Image Viewer")
        root.geometry("1400x920")
        root.minsize(980, 680)

        search = ttk.LabelFrame(root, text="IRSA coordinate search", padding=10)
        search.pack(fill="x", padx=10, pady=(10, 5))
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
            ttk.Label(search, text=label).grid(row=0, column=column, sticky="w", padx=4)
            entry = ttk.Entry(search, width=15)
            entry.insert(0, default)
            entry.grid(row=1, column=column, sticky="ew", padx=4, pady=(2, 0))
            self.entries[label] = entry
            search.columnconfigure(column, weight=1)
        self.search_button = ttk.Button(search, text="Search IRSA", command=self.search)
        self.search_button.grid(row=1, column=len(defaults), padx=(12, 4), sticky="ew")
        root.bind("<Return>", lambda _event: self.search())

        status_row = ttk.Frame(root)
        status_row.pack(fill="x", padx=12, pady=3)
        self.status = tk.StringVar(value="Search a sky position, then check downloaded observations to preview.")
        ttk.Label(status_row, textvariable=self.status, anchor="w").pack(side="left", fill="x", expand=True)
        self.preview_button = ttk.Button(
            status_row, text="Show Selected Images", command=self.show_selected_images, state="disabled"
        )
        self.preview_button.pack(side="right", padx=(8, 0))

        split = ttk.Panedwindow(root, orient="vertical")
        split.pack(fill="both", expand=True, padx=10, pady=(3, 10))

        table_frame = ttk.LabelFrame(split, text="Observations (click the Image box to select up to six)", padding=5)
        split.add(table_frame, weight=3)
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
        self.table.bind("<<TreeviewSelect>>", self._show_observation)

        gallery_box = ttk.LabelFrame(split, text="Selected target-centred previews", padding=6)
        split.add(gallery_box, weight=4)
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

        detail_frame = ttk.LabelFrame(split, text="Observation / image detail", padding=6)
        split.add(detail_frame, weight=2)
        self.detail_preview = ttk.Label(detail_frame, anchor="center")
        self.detail_preview.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.details = tk.Text(detail_frame, wrap="none", height=10, state="disabled")
        detail_y = ttk.Scrollbar(detail_frame, orient="vertical", command=self.details.yview)
        detail_x = ttk.Scrollbar(detail_frame, orient="horizontal", command=self.details.xview)
        self.details.configure(yscrollcommand=detail_y.set, xscrollcommand=detail_x.set)
        self.details.grid(row=0, column=1, sticky="nsew")
        detail_y.grid(row=0, column=2, sticky="ns")
        detail_x.grid(row=1, column=1, sticky="ew")
        detail_frame.rowconfigure(0, weight=1)
        detail_frame.columnconfigure(1, weight=1)
        controls = ttk.Frame(detail_frame)
        controls.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.open_button = ttk.Button(
            controls, text="Open selected FITS URL", command=self._open_selected, state="disabled"
        )
        self.open_button.pack(side="right")

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

    def show_selected_images(self) -> None:
        selected = [self._records[item] for item in self.table.get_children("") if item in self._checked]
        if not selected:
            self.status.set("Select at least one observation using its Image checkbox.")
            return
        if len(selected) > 6:
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
        self.status.set(f"Processing 0/{len(selected)} selected observations...")
        for item in self._checked:
            if self.table.exists(item):
                self.table.set(item, "download_status", "Processing...")

        def run() -> None:
            try:
                results = self.preview_service.prepare_observation_previews(selected, ra, dec, size=200)
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
                    self.status.set(f"Found {len(payload)} observation(s). Check downloaded rows to compare images.")
                elif kind == "search_error":
                    self.search_button.configure(state="normal")
                    self.status.set(f"Search failed: {payload}")
                elif kind == "preview_results":
                    self.preview_button.configure(state="normal")
                    self._render_gallery(payload)
                    successes = sum(result["status"] == "success" for result in payload)
                    self.status.set(f"Prepared {successes}/{len(payload)} preview(s). Select a card for FITS details.")
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
        self._set_details("")
        self._set_detail_image(None)
        self.open_button.configure(state="disabled")
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
        if count > 6:
            self.status.set(f"{count} observations selected; the preview limit is 6.")
        else:
            self.status.set(f"{count} observation(s) selected for preview.")

    def _show_observation(self, _event: Any = None) -> None:
        selected = self.table.selection()
        if not selected:
            return
        observation = self._records[selected[0]]
        self._set_detail_image(None)
        self._set_details(json.dumps(observation, indent=2, sort_keys=True, default=str))
        self.open_button.configure(state="normal" if observation["fits_url"] else "disabled")

    def _render_gallery(self, results: list[dict[str, Any]]) -> None:
        import tkinter as tk
        from tkinter import ttk

        for child in self.gallery.winfo_children():
            child.destroy()
        self._gallery_images.clear()
        for column in range(3):
            self.gallery.columnconfigure(column, weight=1)
        for index, result in enumerate(results):
            row, column = divmod(index, 3)
            card = ttk.LabelFrame(self.gallery, text=_display(result.get("observation_id")) or "Observation", padding=7)
            card.grid(row=row, column=column, padx=6, pady=6, sticky="nsew")
            image_path = result.get("preview_path") if result.get("status") == "success" else None
            if image_path:
                try:
                    image = tk.PhotoImage(file=image_path)
                    self._gallery_images.append(image)
                    image_label = ttk.Label(card, image=image, cursor="hand2")
                    image_label.pack()
                    image_label.bind("<Button-1>", lambda _e, value=result: self._show_preview_detail(value))
                except Exception as exc:
                    ttk.Label(card, text=f"Preview could not be displayed:\n{exc}", justify="center").pack(pady=25)
            else:
                ttk.Label(card, text=result.get("error") or result.get("status", "Failed"), wraplength=260, justify="center").pack(pady=35)
            summary = (
                f"{result.get('observation_date') or 'Date unavailable'}\n"
                f"Detector: {result.get('detector') or '-'}  Band: {result.get('band') or '-'}\n"
                f"Target: {result.get('target_ra'):.6f}, {result.get('target_dec'):.6f}\n"
                f"Status: {result.get('status')}"
            )
            label = ttk.Label(card, text=summary, justify="left", cursor="hand2")
            label.pack(anchor="w", pady=(5, 0))
            label.bind("<Button-1>", lambda _e, value=result: self._show_preview_detail(value))
            matching = next(
                (item for item, record in self._records.items() if record["observation_id"] == result.get("observation_id")),
                None,
            )
            if matching:
                self.table.set(matching, "download_status", "Preview ready" if result["status"] == "success" else result["status"])

    def _show_preview_detail(self, result: dict[str, Any]) -> None:
        self._set_detail_image(result.get("preview_path"))
        self._set_details(json.dumps(result, indent=2, sort_keys=True, default=str))

    def _set_detail_image(self, path: str | None) -> None:
        import tkinter as tk

        self._detail_image = None
        self.detail_preview.configure(image="", text="")
        if path:
            try:
                self._detail_image = tk.PhotoImage(file=path)
                self.detail_preview.configure(image=self._detail_image)
            except Exception as exc:
                self.detail_preview.configure(text=f"Image unavailable\n{exc}")

    def _set_details(self, content: str) -> None:
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", content)
        self.details.configure(state="disabled")

    def _open_selected(self) -> None:
        selected = self.table.selection()
        if selected:
            url = self._records[selected[0]]["fits_url"]
            if url:
                webbrowser.open_new_tab(url)

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
