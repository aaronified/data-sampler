"""
Data Sampler — GUI front-end
All sampling logic lives in data_sampler.py and is imported directly.

Build into a standalone app:
    pyinstaller --onedir --windowed --noupx --name "DataSampler" data-sampler-gui.py
    # output: dist/DataSampler/  — zip the contents and share DataSampler.zip
    # recipients extract the zip and run DataSampler.exe from the extracted folder
"""

import ctypes
import io
import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import data_sampler as _s

# Fix blurry UI on high-DPI Windows displays
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# ── stdout redirect ───────────────────────────────────────────────────────────

class TextRedirect(io.StringIO):
    """Thread-safe redirect of sys.stdout to a tkinter Text widget."""
    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def write(self, s):
        self.widget.after(0, self._append, s)

    def _append(self, s):
        self.widget.configure(state="normal")
        self.widget.insert("end", s, ("normal",))
        self.widget.see("end")
        self.widget.configure(state="disabled")

    def flush(self):
        pass


# ── GUI ───────────────────────────────────────────────────────────────────────

EXCEL_EXTS = {".xlsx", ".xls"}

BG       = "#1e1e2e"
FG       = "#cdd6f4"
ACCENT   = "#89b4fa"
INPUT_BG = "#313244"
BTN_BG   = "#89b4fa"
BTN_FG   = "#1e1e2e"
SUCCESS  = "#a6e3a1"
ERROR    = "#f38ba8"
WARNING  = "#fab387"
MONO     = ("Consolas", 9)
SANS     = ("Segoe UI", 10)
TITLE    = ("Segoe UI Semibold", 11)


class DataSamplerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Data Sampler")
        self.resizable(True, True)
        self.configure(bg=BG)
        self.minsize(620, 560)
        self._last_outdir = None
        self._build_ui()
        self._center()

    def _center(self):
        self.update_idletasks()
        w, h = 700, 620
        x = (self.winfo_screenwidth()  - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self):
        # header
        hdr = tk.Frame(self, bg=ACCENT, height=48)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Data Sampler", font=("Segoe UI Semibold", 14),
                 bg=ACCENT, fg=BTN_FG).pack(side="left", padx=16, pady=10)

        # form
        form = tk.Frame(self, bg=BG)
        form.pack(fill="x", padx=16, pady=6)
        form.columnconfigure(1, weight=1)
        row = 0

        self._label(form, "Input file:", row)
        self.var_source = tk.StringVar()
        self._entry(form, self.var_source, row, col=1)
        self.var_source.trace_add("write", self._on_source_change)
        self._button(form, "Browse…", self._browse_source, row, col=2)
        row += 1

        self.lbl_sheet = self._label(form, "Sheet name:", row)
        self.var_sheet = tk.StringVar()
        self.ent_sheet = self._entry(form, self.var_sheet, row, col=1)
        row += 1
        self._set_sheet_visibility(False)

        self._label(form, "Sample count:", row)
        self.var_count = tk.StringVar(value="100")
        self._entry(form, self.var_count, row, col=1, width=12, sticky="w")
        row += 1

        self._label(form, "Output folder:", row)
        self.var_outdir = tk.StringVar()
        self._entry(form, self.var_outdir, row, col=1)
        self._button(form, "Browse…", self._browse_outdir, row, col=2)
        row += 1

        self._label(form, "Sampling mode:", row)
        self.var_mode = tk.StringVar(value="stratified")
        mode_frame = tk.Frame(form, bg=BG)
        mode_frame.grid(row=row, column=1, sticky="w", pady=4)
        for text, val in [("Stratified (auto)", "stratified"), ("Random", "random")]:
            tk.Radiobutton(mode_frame, text=text, variable=self.var_mode, value=val,
                           bg=BG, fg=FG, selectcolor=INPUT_BG,
                           activebackground=BG, activeforeground=FG,
                           font=SANS).pack(side="left", padx=(0, 16))
        row += 1

        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(fill="x", padx=16, pady=(4, 8))
        self.btn_run = tk.Button(
            btn_frame, text="▶  Run Sample",
            font=("Segoe UI Semibold", 11),
            bg=BTN_BG, fg=BTN_FG, activebackground="#74c7ec",
            relief="flat", cursor="hand2", padx=24, pady=8,
            command=self._run,
        )
        self.btn_run.pack(side="left")
        self.lbl_status = tk.Label(btn_frame, text="", font=SANS, bg=BG, fg=FG)
        self.lbl_status.pack(side="left", padx=16)

        log_frame = tk.Frame(self, bg=BG)
        log_frame.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        tk.Label(log_frame, text="Log", font=TITLE, bg=BG, fg=ACCENT).pack(anchor="w")

        # pack button row first so it anchors to bottom before text area expands
        log_btn_frame = tk.Frame(log_frame, bg=BG)
        log_btn_frame.pack(fill="x", side="bottom", pady=(4, 0))

        txt_frame = tk.Frame(log_frame, bg=INPUT_BG, bd=1, relief="flat")
        txt_frame.pack(fill="both", expand=True, pady=(4, 0))

        self.log = tk.Text(
            txt_frame, bg=INPUT_BG, fg=FG, font=MONO, relief="flat",
            state="disabled", wrap="none",
            selectbackground=ACCENT, selectforeground=BTN_FG, padx=8, pady=6,
        )
        self.log.tag_configure("error",   foreground=ERROR)
        self.log.tag_configure("success", foreground=SUCCESS)
        self.log.tag_configure("normal",  foreground=FG)

        vsb = ttk.Scrollbar(txt_frame, orient="vertical",   command=self.log.yview)
        hsb = ttk.Scrollbar(txt_frame, orient="horizontal", command=self.log.xview)
        self.log.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right",  fill="y")
        hsb.pack(side="bottom", fill="x")
        self.log.pack(fill="both", expand=True)

        self.btn_open_folder = tk.Button(
            log_btn_frame, text="📂  Open output folder", font=("Segoe UI", 9),
            bg=INPUT_BG, fg=FG, relief="flat", cursor="hand2",
            activebackground="#45475a", state="disabled",
            command=self._open_output_folder)
        self.btn_open_folder.pack(side="left")

        tk.Button(log_btn_frame, text="Copy log", font=("Segoe UI", 9),
                  bg=INPUT_BG, fg=FG, relief="flat", cursor="hand2",
                  activebackground="#45475a",
                  command=self._copy_log).pack(side="right", padx=(6, 0))

        tk.Button(log_btn_frame, text="Clear log", font=("Segoe UI", 9),
                  bg=INPUT_BG, fg=FG, relief="flat", cursor="hand2",
                  activebackground="#45475a",
                  command=self._clear_log).pack(side="right")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _label(self, parent, text, row):
        lbl = tk.Label(parent, text=text, font=SANS, bg=BG, fg=FG, anchor="e")
        lbl.grid(row=row, column=0, sticky="e", padx=(0, 8), pady=5)
        return lbl

    def _entry(self, parent, var, row, col=1, width=None, sticky="ew"):
        kw = dict(textvariable=var, bg=INPUT_BG, fg=FG, relief="flat",
                  insertbackground=FG, font=SANS, highlightthickness=1,
                  highlightcolor=ACCENT, highlightbackground="#45475a")
        if width:
            kw["width"] = width
        e = tk.Entry(parent, **kw)
        e.grid(row=row, column=col, sticky=sticky, pady=5, ipady=4)
        return e

    def _button(self, parent, text, cmd, row, col):
        tk.Button(parent, text=text, font=SANS, bg=INPUT_BG, fg=FG,
                  relief="flat", cursor="hand2", activebackground="#45475a",
                  command=cmd, padx=10).grid(row=row, column=col,
                                              sticky="w", padx=(6, 0), pady=5)

    def _set_sheet_visibility(self, visible):
        if visible:
            self.lbl_sheet.grid()
            self.ent_sheet.grid()
        else:
            self.lbl_sheet.grid_remove()
            self.ent_sheet.grid_remove()

    def _on_source_change(self, *_):
        path = self.var_source.get()
        ext = Path(path).suffix.lower() if path else ""
        self._set_sheet_visibility(ext in EXCEL_EXTS)
        if path and os.path.isfile(path) and not self.var_outdir.get():
            self.var_outdir.set(str(Path(path).parent))

    def _browse_source(self):
        path = filedialog.askopenfilename(
            title="Select source file",
            filetypes=[
                ("Data files", "*.csv *.tsv *.json *.xlsx *.xls *.parquet"),
                ("CSV",        "*.csv"),
                ("Excel",      "*.xlsx *.xls"),
                ("JSON",       "*.json"),
                ("Parquet",    "*.parquet"),
                ("All files",  "*.*"),
            ],
        )
        if path:
            self.var_source.set(path)

    def _browse_outdir(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.var_outdir.set(folder)

    def _append_log(self, text, tag="normal"):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", (tag,))
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _copy_log(self):
        text = self.log.get("1.0", "end").strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)

    def _open_output_folder(self):
        if self._last_outdir and os.path.isdir(self._last_outdir):
            os.startfile(self._last_outdir)

    def _set_running(self, running):
        self.btn_run.configure(
            state="disabled" if running else "normal",
            text="⏳  Running…" if running else "▶  Run Sample",
        )
        self.lbl_status.configure(
            text="Working…" if running else "",
            fg=WARNING if running else FG,
        )

    # ── run ───────────────────────────────────────────────────────────────────

    def _run(self):
        source  = self.var_source.get().strip().strip('"').strip("'")
        outdir  = self.var_outdir.get().strip()
        sheet   = self.var_sheet.get().strip()
        count_s = self.var_count.get().strip()
        use_rnd = self.var_mode.get() == "random"

        if not source:
            messagebox.showerror("Missing input", "Please select a source file.")
            return
        if not os.path.isfile(source):
            messagebox.showerror("File not found", f"Cannot find:\n{source}")
            return
        if not count_s.isdigit() or int(count_s) < 1:
            messagebox.showerror("Invalid count", "Sample count must be a positive integer.")
            return
        if outdir and not os.path.isdir(outdir):
            messagebox.showerror("Invalid folder", f"Output folder does not exist:\n{outdir}")
            return

        self._clear_log()
        self._set_running(True)
        threading.Thread(
            target=self._run_worker,
            args=(source, int(count_s), sheet or None, use_rnd, outdir or None),
            daemon=True,
        ).start()

    def _run_worker(self, source, count, sheet, use_rnd, outdir):
        old_stdout = sys.stdout
        sys.stdout = TextRedirect(self.log)
        try:
            print(f"Loading: {source}")
            df = _s.load_file(source, sheet=sheet)
            print(f"Loaded {len(df):,} rows × {len(df.columns)} columns\n")

            result = _s.sample(df, count, use_random=use_rnd)
            print(f"\nSampled {len(result):,} rows.")

            out_path = _s.save_output(result, source, count, output_folder=outdir)
            self._last_outdir = str(Path(out_path).parent)

            self.after(0, self._append_log, "\n✓ Done.", "success")
            self.after(0, self.lbl_status.configure, {"text": "Done!", "fg": SUCCESS})
            self.after(0, self.btn_open_folder.configure, {"state": "normal"})
        except Exception as exc:
            self.after(0, self._append_log, f"\nError: {exc}", "error")
            self.after(0, self.lbl_status.configure, {"text": "Error — see log", "fg": ERROR})
        finally:
            sys.stdout = old_stdout
            self.after(0, self._set_running, False)


if __name__ == "__main__":
    app = DataSamplerApp()
    app.mainloop()
