"""Unified UI for the METAbatt pipeline.

Four tabs, in pipeline order:
    1. Download           -> download/run_download.py
    2. Build BRONZE_CU    -> download/build_bronze_cu_with_ah.py
    3. Run Pipeline       -> main.py
    4. Monitor            -> monitor/aging_status.py
    5. Evaluation         -> evaluation/export_cap_pulse.py

Run from the src/ directory:
    python pipeline_ui.py
"""

import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk


SRC_DIR = Path(__file__).resolve().parent
UI_STATE_PATH = Path.home() / ".config" / "metabatt_ui.json"

DEFAULT_DOWNLOAD_CFG = {
    "project": "",
    "target_cell": [""],
    "cell_type": "",
    "testformat": " Format01",
    "ahjo_endpoint": "https://ahjo.isea.rwth-aachen.de",
    "ahjo_key": "",
    "minio_endpoint": "optimusprime.isea.rwth-aachen.de:9000",
    "access_key": "",
    "secret_key": "",
    "bucket_name": "",
    "export_type": "local",
    "export_path": "",
    "include_unfinished": False,
    "update_unfinished": True,
}


def _load_ui_state() -> dict:
    if UI_STATE_PATH.exists():
        try:
            return json.loads(UI_STATE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_ui_state(state: dict) -> None:
    try:
        UI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        UI_STATE_PATH.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


class ProcessRunner:
    """Spawns a subprocess and streams stdout+stderr into a queue."""

    def __init__(self, on_line, on_done):
        self._on_line = on_line
        self._on_done = on_done
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, argv: list[str], cwd: str | None = None) -> None:
        if self.is_running:
            raise RuntimeError("A process is already running")

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        self._proc = subprocess.Popen(
            argv,
            cwd=cwd or str(SRC_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self._on_line(f"$ {' '.join(shlex.quote(a) for a in argv)}\n")

        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        assert self._proc is not None
        try:
            for line in self._proc.stdout:  # type: ignore[union-attr]
                self._on_line(line)
        finally:
            rc = self._proc.wait() if self._proc else -1
            self._on_done(rc)

    def stop(self) -> None:
        if not self.is_running:
            return
        assert self._proc is not None
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        except Exception:
            pass


class _Section(ctk.CTkFrame):
    """A labeled section with grid-laid form rows."""

    def __init__(self, master, title: str):
        super().__init__(master)
        self.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            self, text=title, font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(8, 4))
        self._row = 1

    def add_entry(self, label: str, *, secret: bool = False, placeholder: str = "") -> ctk.CTkEntry:
        ctk.CTkLabel(self, text=label).grid(
            row=self._row, column=0, sticky="w", padx=(15, 8), pady=3
        )
        entry = ctk.CTkEntry(self, show="*" if secret else "", placeholder_text=placeholder)
        entry.grid(row=self._row, column=1, columnspan=2, sticky="ew", padx=(0, 10), pady=3)
        self._row += 1
        return entry

    def add_path(self, label: str, *, is_dir: bool = False) -> ctk.CTkEntry:
        ctk.CTkLabel(self, text=label).grid(
            row=self._row, column=0, sticky="w", padx=(15, 8), pady=3
        )
        entry = ctk.CTkEntry(self)
        entry.grid(row=self._row, column=1, sticky="ew", padx=(0, 5), pady=3)

        def browse():
            path = (
                filedialog.askdirectory()
                if is_dir
                else filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All", "*.*")])
            )
            if path:
                entry.delete(0, "end")
                entry.insert(0, path)

        ctk.CTkButton(self, text="Browse", width=70, command=browse).grid(
            row=self._row, column=2, padx=(0, 10), pady=3
        )
        self._row += 1
        return entry

    def add_checkbox(self, label: str) -> ctk.CTkCheckBox:
        cb = ctk.CTkCheckBox(self, text=label)
        cb.grid(row=self._row, column=1, sticky="w", padx=(0, 10), pady=3)
        self._row += 1
        return cb


class PipelineUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("METAbatt pipeline")
        self.geometry("1100x880")

        self._state = _load_ui_state()
        self._runner = ProcessRunner(self._on_runner_line, self._on_runner_done)
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._chain: list = []  # remaining build_argv callables when running full pipeline
        self._chain_label: str | None = None

        self._build_widgets()
        self._restore_state()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(80, self._drain_log_queue)

    # ------------------------------------------------------------------ UI build

    def _build_widgets(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Shared battery config row at the top
        shared = ctk.CTkFrame(self)
        shared.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        shared.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            shared, text="Battery config (shared by Build BRONZE_CU / Pipeline / Monitor):"
        ).grid(row=0, column=0, sticky="w", padx=(10, 8), pady=8)
        self.battery_cfg_entry = ctk.CTkEntry(shared)
        self.battery_cfg_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5), pady=8)

        def browse_battery_cfg():
            path = filedialog.askopenfilename(
                filetypes=[("JSON", "*.json"), ("All", "*.*")],
                title="Select battery config JSON",
            )
            if path:
                self.battery_cfg_entry.delete(0, "end")
                self.battery_cfg_entry.insert(0, path)

        ctk.CTkButton(shared, text="Browse", width=70, command=browse_battery_cfg).grid(
            row=0, column=2, padx=(0, 10), pady=8
        )

        # Tabs
        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        self.tabs.add("1. Download")
        self.tabs.add("2. Build BRONZE_CU")
        self.tabs.add("3. Run Pipeline")
        self.tabs.add("4. Monitor")
        self.tabs.add("5. Evaluation")

        self._build_download_tab(self.tabs.tab("1. Download"))
        self._build_bronze_tab(self.tabs.tab("2. Build BRONZE_CU"))
        self._build_pipeline_tab(self.tabs.tab("3. Run Pipeline"))
        self._build_monitor_tab(self.tabs.tab("4. Monitor"))
        self._build_evaluation_tab(self.tabs.tab("5. Evaluation"))

        # Console + bottom controls
        bottom = ctk.CTkFrame(self)
        bottom.grid(row=2, column=0, sticky="nsew", padx=10, pady=(4, 10))
        bottom.grid_rowconfigure(1, weight=1)
        bottom.grid_columnconfigure(0, weight=1)

        ctrl = ctk.CTkFrame(bottom, fg_color="transparent")
        ctrl.grid(row=0, column=0, sticky="ew", pady=(6, 6))
        self.status_label = ctk.CTkLabel(ctrl, text="idle", anchor="w")
        self.status_label.pack(side="left", padx=(8, 12))
        self.stop_btn = ctk.CTkButton(
            ctrl, text="■ Stop", width=90, fg_color="#a83232",
            hover_color="#7f2424", state="disabled", command=self._stop_current,
        )
        self.stop_btn.pack(side="right", padx=(0, 8))
        self.clear_btn = ctk.CTkButton(
            ctrl, text="Clear console", width=120, command=self._clear_console,
        )
        self.clear_btn.pack(side="right", padx=(0, 8))
        self.runall_btn = ctk.CTkButton(
            ctrl, text="⏩ Run all (1→2→3→4→5)", width=200,
            fg_color="#1f6f3d", hover_color="#155028", command=self._run_all,
        )
        self.runall_btn.pack(side="right", padx=(0, 8))

        self.console = ctk.CTkTextbox(bottom, font=("JetBrains Mono", 11), wrap="none")
        self.console.grid(row=1, column=0, sticky="nsew")
        self.console.configure(state="disabled")

    def _build_download_tab(self, parent):
        for i in range(2):
            parent.grid_columnconfigure(i, weight=1)

        proj = _Section(parent, "Project")
        proj.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=6)
        self.dl_project = proj.add_entry("Project:")
        self.dl_specimen = proj.add_entry(
            "Target cell:", placeholder="comma-separated, e.g. cell01,cell02"
        )
        self.dl_cell_type = proj.add_entry(
            "Cell type:", placeholder="required, e.g. VTC or JGNE (MinIO path segment)"
        )
        self.dl_format = proj.add_entry("Test format:", placeholder=" Format01")

        ahjo = _Section(parent, "AHJO")
        ahjo.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        self.dl_ahjo_endpoint = ahjo.add_entry(
            "Endpoint:", placeholder="https://ahjo.isea.rwth-aachen.de"
        )
        self.dl_ahjo_key = ahjo.add_entry("Key:", secret=True)

        minio = _Section(parent, "MinIO")
        minio.grid(row=1, column=1, sticky="ew", padx=8, pady=6)
        self.dl_minio_endpoint = minio.add_entry(
            "Endpoint:", placeholder="optimusprime.isea.rwth-aachen.de:9000"
        )
        self.dl_access_key = minio.add_entry("Access key:")
        self.dl_secret_key = minio.add_entry("Secret key:", secret=True)
        self.dl_bucket = minio.add_entry("Bucket:")

        export = _Section(parent, "Export")
        export.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=6)

        # Export type as a dropdown: local | minio | both
        ctk.CTkLabel(export, text="Export type:").grid(
            row=export._row, column=0, sticky="w", padx=(15, 8), pady=3
        )
        self.dl_export_type = ctk.CTkOptionMenu(
            export, values=["local", "minio", "both"], width=160,
        )
        self.dl_export_type.set("local")
        self.dl_export_type.grid(
            row=export._row, column=1, sticky="w", padx=(0, 10), pady=3
        )
        export._row += 1

        self.dl_export_path = export.add_path("Export path:", is_dir=True)
        self.dl_include_unfinished = export.add_checkbox(
            "Include unfinished tests"
        )
        self.dl_update_unfinished = export.add_checkbox(
            "Update unfinished tests (re-fetch previously unfinished ones)"
        )
        self.dl_include_unfinished.configure(command=self._on_include_unfinished_toggle)
        # Initial state: update_unfinished is locked until include_unfinished is checked.
        self._on_include_unfinished_toggle()

        # Buttons
        btns = ctk.CTkFrame(parent, fg_color="transparent")
        btns.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=10)
        ctk.CTkButton(btns, text="Load JSON", width=120, command=self._dl_load).pack(side="left", padx=4)
        ctk.CTkButton(btns, text="Save JSON", width=120, command=self._dl_save).pack(side="left", padx=4)
        self.dl_run_btn = ctk.CTkButton(
            btns, text="▶ Run download", width=160, command=self._run_download
        )
        self.dl_run_btn.pack(side="right", padx=4)

    def _build_bronze_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        s = _Section(parent, "Build BRONZE_CU")
        s.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        self.br_cells = s.add_entry(
            "Cells filter (--cells):",
            placeholder="optional, space-separated, e.g. VTC_cell01 VTC_cell02",
        )
        self.br_overwrite = s.add_checkbox("--overwrite (rebuild even if output exists)")

        ctk.CTkLabel(
            parent,
            text="Reads its parameters from the shared battery config above.",
            text_color="#888",
        ).grid(row=1, column=0, sticky="w", padx=12, pady=(0, 6))

        btns = ctk.CTkFrame(parent, fg_color="transparent")
        btns.grid(row=2, column=0, sticky="ew", padx=8, pady=10)
        self.br_run_btn = ctk.CTkButton(
            btns, text="▶ Run build_bronze_cu", width=200, command=self._run_bronze,
        )
        self.br_run_btn.pack(side="right", padx=4)

    def _build_pipeline_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        s = _Section(parent, "Run pipeline (main.py)")
        s.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        self.mn_cells = s.add_entry(
            "Cells filter (--cells):",
            placeholder="optional, space-separated, e.g. VTC_cell01 VTC_cell02",
        )
        self.mn_overwrite = s.add_checkbox("--overwrite (reprocess cells with existing GOLD)")

        ctk.CTkLabel(
            parent,
            text="Reads its parameters from the shared battery config above.",
            text_color="#888",
        ).grid(row=1, column=0, sticky="w", padx=12, pady=(0, 6))

        btns = ctk.CTkFrame(parent, fg_color="transparent")
        btns.grid(row=2, column=0, sticky="ew", padx=8, pady=10)
        self.mn_run_btn = ctk.CTkButton(
            btns, text="▶ Run main pipeline", width=180, command=self._run_pipeline,
        )
        self.mn_run_btn.pack(side="right", padx=4)

    def _build_monitor_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        s = _Section(parent, "Aging-status monitor")
        s.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        self.mo_output = s.add_path("Output HTML (optional):", is_dir=False)

        ctk.CTkLabel(
            parent,
            text="If left blank, the report goes to <working_path>/40_capacity_monitore/aging_status.html.",
            text_color="#888",
        ).grid(row=1, column=0, sticky="w", padx=12, pady=(0, 6))

        btns = ctk.CTkFrame(parent, fg_color="transparent")
        btns.grid(row=2, column=0, sticky="ew", padx=8, pady=10)
        self.mo_open_btn = ctk.CTkButton(
            btns, text="Open last report", width=150, command=self._open_monitor_report,
            state="disabled",
        )
        self.mo_open_btn.pack(side="left", padx=4)
        self.mo_run_btn = ctk.CTkButton(
            btns, text="▶ Run monitor", width=150, command=self._run_monitor,
        )
        self.mo_run_btn.pack(side="right", padx=4)

        self._last_monitor_html: str | None = None

    def _build_evaluation_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        s = _Section(parent, "Fleet-wide capacity aggregation")
        s.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        self.ev_output = s.add_path("Output CSV (optional):", is_dir=False)

        ctk.CTkLabel(
            parent,
            text=(
                "If left blank, the CSV goes to <working_path>/50_evaluation/capacity_results.csv. "
                "Pulse aggregation will be a separate stage."
            ),
            text_color="#888",
            wraplength=900,
            justify="left",
        ).grid(row=1, column=0, sticky="w", padx=12, pady=(0, 6))

        btns = ctk.CTkFrame(parent, fg_color="transparent")
        btns.grid(row=2, column=0, sticky="ew", padx=8, pady=10)
        self.ev_run_btn = ctk.CTkButton(
            btns, text="▶ Run evaluation", width=160, command=self._run_evaluation,
        )
        self.ev_run_btn.pack(side="right", padx=4)

    # ------------------------------------------------------------- state I/O

    def _restore_state(self) -> None:
        s = self._state
        self.battery_cfg_entry.insert(0, s.get("battery_cfg", ""))
        self.br_cells.insert(0, s.get("br_cells", ""))
        if s.get("br_overwrite"):
            self.br_overwrite.select()
        self.mn_cells.insert(0, s.get("mn_cells", ""))
        if s.get("mn_overwrite"):
            self.mn_overwrite.select()
        self.mo_output.insert(0, s.get("mo_output", ""))
        self.ev_output.insert(0, s.get("ev_output", ""))

        dl = {**DEFAULT_DOWNLOAD_CFG, **s.get("download_cfg", {})}
        self._apply_download_cfg(dl)

    def _persist_state(self) -> None:
        self._state.update({
            "battery_cfg": self.battery_cfg_entry.get(),
            "br_cells": self.br_cells.get(),
            "br_overwrite": bool(self.br_overwrite.get()),
            "mn_cells": self.mn_cells.get(),
            "mn_overwrite": bool(self.mn_overwrite.get()),
            "mo_output": self.mo_output.get(),
            "ev_output": self.ev_output.get(),
            "download_cfg": self._collect_download_cfg(),
        })
        _save_ui_state(self._state)

    def _on_close(self):
        if self._runner.is_running:
            if not messagebox.askyesno(
                "Confirm exit",
                "A pipeline stage is still running. Stop it and exit?",
            ):
                return
            self._runner.stop()
        self._persist_state()
        self.destroy()

    # ----------------------------------------------------- download config I/O

    def _collect_download_cfg(self) -> dict:
        specimen_text = self.dl_specimen.get().strip()
        specimen_list = [s.strip() for s in specimen_text.split(",")] if specimen_text else [""]
        return {
            "project": self.dl_project.get(),
            "target_cell": specimen_list,
            "cell_type": self.dl_cell_type.get().strip(),
            "testformat": self.dl_format.get(),
            "ahjo_endpoint": self.dl_ahjo_endpoint.get(),
            "ahjo_key": self.dl_ahjo_key.get(),
            "minio_endpoint": self.dl_minio_endpoint.get(),
            "access_key": self.dl_access_key.get(),
            "secret_key": self.dl_secret_key.get(),
            "bucket_name": self.dl_bucket.get(),
            "export_type": self.dl_export_type.get(),
            "export_path": self.dl_export_path.get(),
            "include_unfinished": bool(self.dl_include_unfinished.get()),
            "update_unfinished": bool(self.dl_update_unfinished.get()),
        }

    def _apply_download_cfg(self, cfg: dict) -> None:
        def _set(entry: ctk.CTkEntry, value: str) -> None:
            entry.delete(0, "end")
            entry.insert(0, value or "")

        _set(self.dl_project, cfg.get("project", ""))
        # Accept legacy "target_specimen" key for back-compat with older saved configs.
        target = cfg.get("target_cell", cfg.get("target_specimen", [""]))
        _set(self.dl_specimen, ", ".join(target) if isinstance(target, list) else str(target))
        _set(self.dl_cell_type, cfg.get("cell_type", ""))
        _set(self.dl_format, cfg.get("testformat", ""))
        _set(self.dl_ahjo_endpoint, cfg.get("ahjo_endpoint", ""))
        _set(self.dl_ahjo_key, cfg.get("ahjo_key", ""))
        _set(self.dl_minio_endpoint, cfg.get("minio_endpoint", ""))
        _set(self.dl_access_key, cfg.get("access_key", ""))
        _set(self.dl_secret_key, cfg.get("secret_key", ""))
        _set(self.dl_bucket, cfg.get("bucket_name", ""))

        # export_type: accept legacy "server" as alias for "minio".
        et = (cfg.get("export_type") or "local").lower().strip()
        if et == "server":
            et = "minio"
        if et not in ("local", "minio", "both"):
            et = "local"
        self.dl_export_type.set(et)

        _set(self.dl_export_path, cfg.get("export_path", ""))

        if cfg.get("include_unfinished", False):
            self.dl_include_unfinished.select()
        else:
            self.dl_include_unfinished.deselect()
        if cfg.get("update_unfinished", True):
            self.dl_update_unfinished.select()
        else:
            self.dl_update_unfinished.deselect()
        # Re-apply the dependency: update_unfinished is only enabled when
        # include_unfinished is checked.
        self._on_include_unfinished_toggle()

    def _on_include_unfinished_toggle(self) -> None:
        """Enable update_unfinished only while include_unfinished is checked."""
        if self.dl_include_unfinished.get():
            self.dl_update_unfinished.configure(state="normal")
        else:
            self.dl_update_unfinished.deselect()
            self.dl_update_unfinished.configure(state="disabled")

    def _dl_save(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save download config",
        )
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(self._collect_download_cfg(), indent=4))
            messagebox.showinfo("Saved", f"Download config saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _dl_load(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load download config",
        )
        if not path:
            return
        try:
            cfg = json.loads(Path(path).read_text())
            self._apply_download_cfg(cfg)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    # ------------------------------------------------------------ argv builders

    def _battery_cfg_or_warn(self) -> str | None:
        cfg = self.battery_cfg_entry.get().strip()
        if not cfg:
            messagebox.showerror("Missing config", "Set the battery config path at the top first.")
            return None
        if not os.path.exists(cfg):
            messagebox.showerror("Missing config", f"Battery config not found:\n{cfg}")
            return None
        return cfg

    def _build_download_argv(self) -> list[str] | None:
        cfg = self._collect_download_cfg()
        missing = [
            k
            for k in ("project", "cell_type", "ahjo_endpoint", "ahjo_key", "export_path")
            if not cfg.get(k)
        ]
        if missing:
            messagebox.showerror(
                "Incomplete download config",
                "Missing required fields: " + ", ".join(missing),
            )
            return None

        # Persist a temp config the subprocess can read.
        tmp_cfg = SRC_DIR.parent / ".metabatt_ui_download.json"
        tmp_cfg.write_text(json.dumps(cfg, indent=2))
        return [sys.executable, "download/run_download.py", str(tmp_cfg)]

    def _build_bronze_argv(self) -> list[str] | None:
        cfg = self._battery_cfg_or_warn()
        if not cfg:
            return None
        argv = [sys.executable, "download/build_bronze_cu_with_ah.py", cfg]
        cells = self.br_cells.get().strip().split()
        if cells:
            argv += ["--cells", *cells]
        if self.br_overwrite.get():
            argv.append("--overwrite")
        return argv

    def _build_pipeline_argv(self) -> list[str] | None:
        cfg = self._battery_cfg_or_warn()
        if not cfg:
            return None
        argv = [sys.executable, "main.py", cfg]
        cells = self.mn_cells.get().strip().split()
        if cells:
            argv += ["--cells", *cells]
        if self.mn_overwrite.get():
            argv.append("--overwrite")
        return argv

    def _build_monitor_argv(self) -> list[str] | None:
        cfg = self._battery_cfg_or_warn()
        if not cfg:
            return None
        argv = [sys.executable, "-m", "monitor.aging_status", cfg]
        out = self.mo_output.get().strip()
        if out:
            argv += ["-o", out]
            self._last_monitor_html = out
        else:
            # Derive from battery cfg working_path
            try:
                bcfg = json.loads(Path(cfg).read_text())
                wp = bcfg.get("working_path")
                if wp:
                    self._last_monitor_html = os.path.join(
                        wp, "40_capacity_monitore", "aging_status.html"
                    )
            except Exception:
                self._last_monitor_html = None
        return argv

    def _build_evaluation_argv(self) -> list[str] | None:
        cfg = self._battery_cfg_or_warn()
        if not cfg:
            return None
        argv = [sys.executable, "-m", "evaluation.export_cap_pulse", cfg]
        out = self.ev_output.get().strip()
        if out:
            argv += ["-o", out]
        return argv

    # --------------------------------------------------------------- run paths

    def _run_download(self):
        argv = self._build_download_argv()
        if argv:
            self._launch(argv, label="download")

    def _run_bronze(self):
        argv = self._build_bronze_argv()
        if argv:
            self._launch(argv, label="build_bronze_cu")

    def _run_pipeline(self):
        argv = self._build_pipeline_argv()
        if argv:
            self._launch(argv, label="main pipeline")

    def _run_monitor(self):
        argv = self._build_monitor_argv()
        if argv:
            self._launch(argv, label="monitor")

    def _run_evaluation(self):
        argv = self._build_evaluation_argv()
        if argv:
            self._launch(argv, label="evaluation")

    def _run_all(self):
        if self._runner.is_running:
            messagebox.showwarning("Busy", "A stage is already running.")
            return
        # Build all four argvs up front so we fail fast on missing config.
        builders = [
            ("download", self._build_download_argv),
            ("build_bronze_cu", self._build_bronze_argv),
            ("main pipeline", self._build_pipeline_argv),
            ("monitor", self._build_monitor_argv),
            ("evaluation", self._build_evaluation_argv),
        ]
        steps: list[tuple[str, list[str]]] = []
        for label, fn in builders:
            argv = fn()
            if argv is None:
                self._append_console(
                    f"[run-all aborted: failed to prepare stage '{label}']\n"
                )
                return
            steps.append((label, argv))
        self._append_console("=== Running all stages 1->2->3->4->5 ===\n")
        self._chain = steps[1:]
        first_label, first_argv = steps[0]
        self._launch(first_argv, label=f"{first_label} (1/5)")

    def _launch(self, argv: list[str], *, label: str) -> None:
        if self._runner.is_running:
            messagebox.showwarning("Busy", "A stage is already running.")
            return
        self._chain_label = label
        self._set_running_ui(True, label)
        try:
            self._runner.start(argv)
        except Exception as e:
            self._set_running_ui(False, "idle")
            messagebox.showerror("Failed to start", str(e))

    def _stop_current(self):
        if self._runner.is_running:
            self._append_console("[stop requested]\n")
            self._chain = []  # cancel any pending chain
            self._runner.stop()

    # ----------------------------------------------------- runner callbacks

    def _on_runner_line(self, line: str) -> None:
        # Called from background thread — marshal to UI thread.
        self._log_queue.put(line)

    def _on_runner_done(self, returncode: int) -> None:
        self._log_queue.put(f"\n[exit {returncode}]\n")
        self.after(0, self._after_done, returncode)

    def _after_done(self, returncode: int) -> None:
        label = self._chain_label or "stage"
        if returncode == 0:
            self.status_label.configure(text=f"{label} finished ok")
            if label.startswith("monitor") and self._last_monitor_html:
                self.mo_open_btn.configure(state="normal")
        else:
            self.status_label.configure(text=f"{label} failed (exit {returncode})")
            self._chain = []  # abort any chain on failure

        if self._chain and returncode == 0:
            next_label, next_argv = self._chain.pop(0)
            remaining_index = 5 - len(self._chain)  # 2,3,4,5
            self._launch(next_argv, label=f"{next_label} ({remaining_index}/5)")
            return

        self._set_running_ui(False, self.status_label.cget("text"))

    def _drain_log_queue(self) -> None:
        try:
            while True:
                line = self._log_queue.get_nowait()
                self._append_console(line)
        except queue.Empty:
            pass
        self.after(80, self._drain_log_queue)

    # ------------------------------------------------------ console helpers

    def _append_console(self, text: str) -> None:
        self.console.configure(state="normal")
        self.console.insert("end", text)
        self.console.see("end")
        self.console.configure(state="disabled")

    def _clear_console(self) -> None:
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def _set_running_ui(self, running: bool, status_text: str) -> None:
        self.status_label.configure(text=status_text if running else f"idle ({status_text})")
        new_state = "disabled" if running else "normal"
        for btn in (
            self.dl_run_btn, self.br_run_btn, self.mn_run_btn,
            self.mo_run_btn, self.ev_run_btn, self.runall_btn,
        ):
            btn.configure(state=new_state)
        self.stop_btn.configure(state="normal" if running else "disabled")

    def _open_monitor_report(self) -> None:
        path = self._last_monitor_html
        if path and os.path.exists(path):
            webbrowser.open(f"file://{os.path.abspath(path)}")
        else:
            messagebox.showinfo(
                "Not available",
                "No local monitor report has been generated yet, or the path "
                "is not local (e.g. MinIO-only upload).",
            )


def main():
    app = PipelineUI()
    app.mainloop()


if __name__ == "__main__":
    main()
