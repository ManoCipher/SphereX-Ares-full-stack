"""Tkinter desktop viewer for live SPHEREx archive metadata searches."""

from __future__ import annotations

import json
import queue
import threading
import webbrowser
from collections.abc import Callable
from typing import Any

from .spherex_client import SPHERExArchiveError, SPHERExClient, SPHERExObservation


def _display(value: Any) -> str:
    return "" if value is None else str(value)


def _wavelength_label(observation: SPHERExObservation) -> str:
    lower = observation["wavelength_min_um"]
    upper = observation["wavelength_max_um"]
    if lower is None and upper is None:
        return ""
    if lower is None:
        return f"≤ {upper:.3f}"
    if upper is None:
        return f"≥ {lower:.3f}"
    return f"{lower:.3f}–{upper:.3f}"


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
    """Interactive desktop table backed by :class:`SPHERExClient`."""

    COLUMNS = (
        "observation_timestamp",
        "observation_id",
        "collection",
        "detector",
        "wavelength",
        "exposure_time_s",
        "file_size",
    )

    HEADINGS = {
        "observation_timestamp": "Observed (UTC)",
        "observation_id": "Observation ID",
        "collection": "Collection",
        "detector": "Detector",
        "wavelength": "Wavelength (µm)",
        "exposure_time_s": "Exposure (s)",
        "file_size": "File size",
    }

    def __init__(self, root: Any, client: SPHERExClient | None = None) -> None:
        # Tkinter is imported lazily so importing the archive package remains
        # safe in headless pipelines and unit tests.
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.client = client or SPHERExClient()
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._records: dict[str, SPHERExObservation] = {}
        self._sort_reverse: dict[str, bool] = {}

        root.title("SPHEREx Archive Viewer")
        root.geometry("1280x760")
        root.minsize(900, 560)

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
            entry = ttk.Entry(search, width=16)
            entry.insert(0, default)
            entry.grid(row=1, column=column, sticky="ew", padx=4, pady=(2, 0))
            self.entries[label] = entry
            search.columnconfigure(column, weight=1)

        self.search_button = ttk.Button(search, text="Search IRSA", command=self.search)
        self.search_button.grid(row=1, column=len(defaults), padx=(12, 4), sticky="ew")
        root.bind("<Return>", lambda _event: self.search())

        self.status = tk.StringVar(value="Enter a sky position, then search.")
        ttk.Label(root, textvariable=self.status, anchor="w").pack(fill="x", padx=14, pady=3)

        split = ttk.Panedwindow(root, orient="vertical")
        split.pack(fill="both", expand=True, padx=10, pady=(3, 10))

        table_frame = ttk.Frame(split)
        split.add(table_frame, weight=3)
        self.table = ttk.Treeview(table_frame, columns=self.COLUMNS, show="headings", selectmode="browse")
        widths = (175, 280, 130, 75, 145, 100, 100)
        for column, width in zip(self.COLUMNS, widths, strict=True):
            self.table.heading(
                column,
                text=self.HEADINGS[column],
                command=lambda selected=column: self._sort(selected),
            )
            self.table.column(column, width=width, minwidth=60, stretch=True)
        table_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=table_scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        table_scroll.pack(side="right", fill="y")
        self.table.bind("<<TreeviewSelect>>", self._show_selected)

        detail_frame = ttk.LabelFrame(split, text="Selected observation metadata", padding=6)
        split.add(detail_frame, weight=2)
        self.details = tk.Text(detail_frame, wrap="none", height=12, state="disabled")
        detail_y = ttk.Scrollbar(detail_frame, orient="vertical", command=self.details.yview)
        detail_x = ttk.Scrollbar(detail_frame, orient="horizontal", command=self.details.xview)
        self.details.configure(yscrollcommand=detail_y.set, xscrollcommand=detail_x.set)
        self.details.grid(row=0, column=0, sticky="nsew")
        detail_y.grid(row=0, column=1, sticky="ns")
        detail_x.grid(row=1, column=0, sticky="ew")
        detail_frame.rowconfigure(0, weight=1)
        detail_frame.columnconfigure(0, weight=1)

        controls = ttk.Frame(detail_frame)
        controls.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.open_button = ttk.Button(
            controls, text="Open selected FITS URL", command=self._open_selected, state="disabled"
        )
        self.open_button.pack(side="right")

        root.after(100, self._poll_results)

    def search(self) -> None:
        """Start a live archive query on a worker thread."""
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
            # Validate synchronously so input mistakes appear immediately and do
            # not become indistinguishable from network errors.
            self.client._validate_position(arguments["ra"], arguments["dec"], arguments["radius"])
            if arguments["max_results"] <= 0:
                raise ValueError("Max results must be positive")
        except ValueError as exc:
            self.status.set(f"Invalid search: {exc}")
            return

        self.search_button.configure(state="disabled")
        self.status.set("Searching NASA/IPAC IRSA…")
        self._replace_records([])

        def run() -> None:
            try:
                self._queue.put(("results", self.client.search(**arguments)))
            except (SPHERExArchiveError, ValueError) as exc:
                self._queue.put(("error", str(exc)))
            except Exception as exc:  # keep unexpected worker failures visible
                self._queue.put(("error", f"Unexpected search failure: {exc}"))

        threading.Thread(target=run, name="spherex-irsa-search", daemon=True).start()

    def _poll_results(self) -> None:
        try:
            kind, payload = self._queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_results)
            return

        self.search_button.configure(state="normal")
        if kind == "results":
            self._replace_records(payload)
            self.status.set(f"Found {len(payload)} observation(s). Select a row for full metadata.")
        else:
            self.status.set(f"Search failed: {payload}")
        self.root.after(100, self._poll_results)

    def _replace_records(self, observations: list[SPHERExObservation]) -> None:
        self.table.delete(*self.table.get_children())
        self._records.clear()
        self._set_details("")
        self.open_button.configure(state="disabled")
        for index, observation in enumerate(observations):
            item_id = f"observation-{index}"
            self._records[item_id] = observation
            self.table.insert(
                "",
                "end",
                iid=item_id,
                values=(
                    _display(observation["observation_timestamp"]),
                    _display(observation["observation_id"]),
                    _display(observation["collection"]),
                    _display(observation["detector"]),
                    _wavelength_label(observation),
                    _display(observation["exposure_time_s"]),
                    _size_label(observation["file_size_bytes"]),
                ),
            )

    def _show_selected(self, _event: Any = None) -> None:
        selected = self.table.selection()
        if not selected:
            return
        observation = self._records[selected[0]]
        self._set_details(json.dumps(observation, indent=2, sort_keys=True, default=str))
        state = "normal" if observation["fits_url"] else "disabled"
        self.open_button.configure(state=state)

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
    """Launch the desktop viewer."""
    import tkinter as tk

    root = tk.Tk()
    SPHERExViewer(root, client=client_factory())
    root.mainloop()


if __name__ == "__main__":
    main()
