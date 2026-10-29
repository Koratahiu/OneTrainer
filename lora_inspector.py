import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import torch
from safetensors.torch import load_file
from safetensors import safe_open
import os
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class LoRAInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LoRA Weight Inspector Pro + Factor Analytics (‖A‖_F & σ(B)) + Global Totals")
        self.root.geometry("1650x940")
        
        # --- Dark Mode Colors ---
        self.bg_color = "#2d2d2d"
        self.fg_color = "#e0e0e0"
        self.entry_bg = "#3d3d3d"
        self.root.configure(bg=self.bg_color)

        self.mode_var = tk.StringVar(value="compare")
        self.file1_path = tk.StringVar()
        self.file2_path = tk.StringVar()
        self.global_diff_var = tk.StringVar(value="Ready.")
        self.file1_info_var = tk.StringVar(value="LoRA 1: No file loaded.")
        self.file2_info_var = tk.StringVar(value="LoRA 2: No file loaded.")
        
        # Global Summary Bar Text
        self.global_summary_text = tk.StringVar(value="No analysis run yet.")

        # Configuration
        self.auto_alpha_var = tk.BooleanVar(value=True)
        self.norm_view_var = tk.StringVar(value="Effective (Scaled)")

        # In-memory cached analysis results
        self.cached_r1 = {}
        self.cached_r2 = {}
        self.results_data = [] 
        self.global_stats = {}
        self.current_columns = []

        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background=self.entry_bg, foreground=self.fg_color, fieldbackground=self.entry_bg, borderwidth=0)
        style.configure("Treeview.Heading", background="#4d4d4d", foreground=self.fg_color, borderwidth=1)
        style.map("Treeview", background=[('selected', '#1a73e8')])
        
        style.configure("TNotebook", background=self.bg_color, borderwidth=0)
        style.configure("TNotebook.Tab", background="#4d4d4d", foreground=self.fg_color, padding=[10, 5])
        style.map("TNotebook.Tab", background=[('selected', '#1a73e8')])

        # --- Top Frame (Mode & Files) ---
        top_frame = tk.Frame(self.root, padx=10, pady=8, bg=self.bg_color)
        top_frame.pack(fill=tk.X)

        # Mode Selection & Options Toolbar
        mode_frame = tk.Frame(top_frame, bg=self.bg_color)
        mode_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        
        tk.Label(mode_frame, text="Mode:", bg=self.bg_color, fg=self.fg_color, font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Single LoRA", variable=self.mode_var, value="single", command=self.toggle_mode, bg=self.bg_color, fg=self.fg_color, selectcolor=self.entry_bg).pack(side=tk.LEFT, padx=6)
        tk.Radiobutton(mode_frame, text="Compare Two LoRAs", variable=self.mode_var, value="compare", command=self.toggle_mode, bg=self.bg_color, fg=self.fg_color, selectcolor=self.entry_bg).pack(side=tk.LEFT, padx=6)

        # Options Toolbar
        opt_frame = tk.Frame(mode_frame, bg=self.bg_color)
        opt_frame.pack(side=tk.RIGHT)

        # Norm Mode Selector (Raw vs Scaled by Alpha/r)
        tk.Label(opt_frame, text="Norm View:", bg=self.bg_color, fg="#80cbc4", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(8, 2))
        self.norm_mode_combo = ttk.Combobox(
            opt_frame,
            textvariable=self.norm_view_var,
            values=["Effective (Scaled)", "Raw (Unscaled)"],
            state="readonly",
            width=16,
            font=("Arial", 9)
        )
        self.norm_mode_combo.pack(side=tk.LEFT, padx=4)
        self.norm_mode_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_display())

        tk.Checkbutton(opt_frame, text="Auto-Detect Alpha", variable=self.auto_alpha_var, bg=self.bg_color, fg=self.fg_color, selectcolor=self.entry_bg).pack(side=tk.LEFT, padx=5)

        # File Inputs
        tk.Label(top_frame, text="LoRA 1 (Baseline):", bg=self.bg_color, fg=self.fg_color).grid(row=1, column=0, sticky="e", pady=2)
        tk.Entry(top_frame, textvariable=self.file1_path, width=82, bg=self.entry_bg, fg=self.fg_color, insertbackground="white").grid(row=1, column=1, padx=5)
        tk.Button(top_frame, text="Browse", command=lambda: self.browse_file(self.file1_path, self.file1_info_var), bg="#555555", fg="white").grid(row=1, column=2)

        self.lbl_file1_info = tk.Label(top_frame, textvariable=self.file1_info_var, font=("Consolas", 8), fg="#81c784", bg=self.bg_color)
        self.lbl_file1_info.grid(row=2, column=1, sticky="w", padx=5)

        self.lbl_file2 = tk.Label(top_frame, text="LoRA 2 (Target):", bg=self.bg_color, fg=self.fg_color)
        self.lbl_file2.grid(row=3, column=0, sticky="e", pady=2)
        self.entry_file2 = tk.Entry(top_frame, textvariable=self.file2_path, width=82, bg=self.entry_bg, fg=self.fg_color, insertbackground="white")
        self.entry_file2.grid(row=3, column=1, padx=5)
        self.btn_file2 = tk.Button(top_frame, text="Browse", command=lambda: self.browse_file(self.file2_path, self.file2_info_var), bg="#555555", fg="white")
        self.btn_file2.grid(row=3, column=2)

        self.lbl_file2_info = tk.Label(top_frame, textvariable=self.file2_info_var, font=("Consolas", 8), fg="#64b5f6", bg=self.bg_color)
        self.lbl_file2_info.grid(row=4, column=1, sticky="w", padx=5)

        # --- Button Bar & File Level Stats ---
        btn_frame = tk.Frame(self.root, pady=6, bg=self.bg_color)
        btn_frame.pack(fill=tk.X)
        
        tk.Button(btn_frame, text="1. Run Analysis", command=self.run_analysis, bg="#2e7d32", fg="white", width=18, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="2. Show Analytics", command=self.show_charts, bg="#1565c0", fg="white", width=18, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="3. Export CSV", command=self.export_csv, bg="#ef6c00", fg="white", width=18, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)

        tk.Label(btn_frame, textvariable=self.global_diff_var, font=("Arial", 10, "bold"), fg="#ffb74d", bg=self.bg_color).pack(side=tk.RIGHT, padx=20)

        # --- Dedicated Global Analytics Banner ---
        summary_frame = tk.Frame(self.root, bg="#1e222a", padx=12, pady=6, highlightbackground="#3d4450", highlightthickness=1)
        summary_frame.pack(fill=tk.X, padx=10, pady=(4, 6))

        tk.Label(summary_frame, text="GLOBAL TOTALS & AVERAGES:", bg="#1e222a", fg="#ffd54f", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(summary_frame, textvariable=self.global_summary_text, bg="#1e222a", fg="#ffffff", font=("Consolas", 9)).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- Table Frame ---
        bottom_frame = tk.Frame(self.root, padx=10, pady=2, bg=self.bg_color)
        bottom_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(bottom_frame, show="headings")
        
        y_scroll = ttk.Scrollbar(bottom_frame, orient=tk.VERTICAL, command=self.tree.yview)
        x_scroll = ttk.Scrollbar(bottom_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscroll=y_scroll.set, xscroll=x_scroll.set)
        
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Tree tags for highlights
        self.tree.tag_configure("summary", background="#1e3a5f", foreground="#ffd54f", font=("Arial", 9, "bold"))  # Pinned Total Row
        self.tree.tag_configure("collapsed", foreground="#ffb74d")    # Orange (Rank collapsed)
        self.tree.tag_configure("dead", foreground="#9e9e9e")         # Grey (Zero weights)
        self.tree.tag_configure("safe", foreground="#e0e0e0")         # Normal

        self.toggle_mode()

    def toggle_mode(self):
        if self.mode_var.get() == "single":
            self.lbl_file2.grid_remove()
            self.entry_file2.grid_remove()
            self.btn_file2.grid_remove()
            self.lbl_file2_info.grid_remove()
        else:
            self.lbl_file2.grid()
            self.entry_file2.grid()
            self.btn_file2.grid()
            self.lbl_file2_info.grid()

    def browse_file(self, path_var, info_var):
        path = filedialog.askopenfilename(filetypes=[("LoRA Weights", "*.safetensors *.bin *.pt")])
        if path:
            path_var.set(path)
            self.inspect_file_header(path, info_var)

    def inspect_file_header(self, path, info_var):
        if not path or not os.path.exists(path):
            info_var.set("File not found.")
            return
        try:
            metadata = {}
            if path.endswith(".safetensors"):
                try:
                    with safe_open(path, framework="pt") as f:
                        metadata = f.metadata() or {}
                except Exception:
                    pass

            sd = load_file(path) if path.endswith(".safetensors") else torch.load(path, map_location="cpu")
            pairs = self.find_lora_pairs(sd)
            ranks = [p['rank'] for p in pairs.values() if 'rank' in p]
            avg_rank = f"{sum(ranks)/len(ranks):.1f}" if ranks else "N/A"

            info_str = f"Pairs Detected: {len(pairs)} | Avg Rank: {avg_rank} | Metadata Keys: {len(metadata)}"
            info_var.set(info_str)
        except Exception as e:
            info_var.set(f"Could not read metadata: {e}")

    def find_lora_pairs(self, sd):
        pairs = {}
        for k in sd.keys():
            k_lower = k.lower()
            if any(term in k_lower for term in ['alpha', 'bias', 'magnitude_vector', 'dora_scale']):
                continue

            # Down / A weights
            if any(term in k for term in ['.lora_A.', '.lora_down.', '.down.', '.linear_1.']):
                base = k.replace('.lora_A.weight', '').replace('.lora_A.default.weight', '') \
                        .replace('.lora_down.weight', '').replace('.down.weight', '') \
                        .replace('.linear_1.weight', '')
                pairs.setdefault(base, {})['down'] = k
            
            # Up / B weights
            elif any(term in k for term in ['.lora_B.', '.lora_up.', '.up.', '.linear_2.']):
                base = k.replace('.lora_B.weight', '').replace('.lora_B.default.weight', '') \
                        .replace('.lora_up.weight', '').replace('.up.weight', '') \
                        .replace('.linear_2.weight', '')
                pairs.setdefault(base, {})['up'] = k

        valid_pairs = {}
        for base, p in pairs.items():
            if 'down' in p and 'up' in p:
                t_down = sd[p['down']]
                rank = t_down.shape[0] if len(t_down.shape) >= 2 else 0
                p['rank'] = rank
                valid_pairs[base] = p

        return valid_pairs

    @staticmethod
    def compute_lora_metrics(A_tensor: torch.Tensor, B_tensor: torch.Tensor, scale: float = 1.0):
        A = A_tensor.float()
        B = B_tensor.float()

        if A.dim() > 2:
            A = A.flatten(start_dim=1)
        if B.dim() > 2:
            B = B.flatten(start_dim=1)

        r, d_in = A.shape
        d_out, r_b = B.shape
        if r != r_b or r == 0:
            return {
                "spec_max_eff": 0.0, "spec_mean_eff": 0.0, "spec_max_raw": 0.0,
                "spec_mean_raw": 0.0, "fro_eff": 0.0, "fro_raw": 0.0,
                "a_l2": 0.0, "b_spec": 0.0, "stable_rank": 1.0
            }

        # 1. Direct Factor Metrics
        a_l2 = torch.linalg.norm(A).item()

        try:
            b_svs = torch.linalg.svdvals(B)
            b_spec = b_svs[0].item() if b_svs.numel() > 0 else 0.0
        except Exception:
            b_spec = 0.0

        # 2. Product Matrix ΔW Metrics via Reduced QR-Decomposition
        try:
            Q_A, R_A = torch.linalg.qr(A.t(), mode='reduced')
            Q_B, R_B = torch.linalg.qr(B, mode='reduced')
            
            M = torch.mm(R_B, R_A.t())
            svs = torch.linalg.svdvals(M)
            
            spec_max_raw = svs[0].item() if svs.numel() > 0 else 0.0
            spec_mean_raw = svs.mean().item() if svs.numel() > 0 else 0.0
            fro_raw = torch.linalg.norm(svs).item()

            spec_max_eff = spec_max_raw * scale
            spec_mean_eff = spec_mean_raw * scale
            fro_eff = fro_raw * scale

            stable_rank = (fro_raw**2) / (spec_max_raw**2 + 1e-12)
            stable_rank = min(max(stable_rank, 1.0), float(r))

        except Exception:
            spec_max_raw, spec_mean_raw, fro_raw = 0.0, 0.0, 0.0
            spec_max_eff, spec_mean_eff, fro_eff = 0.0, 0.0, 0.0
            stable_rank = 1.0

        return {
            "spec_max_eff": spec_max_eff,
            "spec_mean_eff": spec_mean_eff,
            "spec_max_raw": spec_max_raw,
            "spec_mean_raw": spec_mean_raw,
            "fro_eff": fro_eff,
            "fro_raw": fro_raw,
            "a_l2": a_l2,
            "b_spec": b_spec,
            "stable_rank": stable_rank
        }

    def get_metrics(self, path):
        if not path or not os.path.exists(path): return {}
        try:
            metadata = {}
            if path.endswith(".safetensors"):
                try:
                    with safe_open(path, framework="pt") as f:
                        metadata = f.metadata() or {}
                except Exception:
                    pass

            sd = load_file(path) if path.endswith(".safetensors") else torch.load(path, map_location="cpu")
            pairs = self.find_lora_pairs(sd)
            layers = {}

            global_alpha = None
            if "lora_alpha" in metadata:
                try: global_alpha = float(metadata["lora_alpha"])
                except Exception: pass

            alpha_dict = {}
            for k, v in sd.items():
                if 'alpha' in k:
                    base = k.replace('.alpha', '')
                    alpha_dict[base] = float(v.item() if hasattr(v, 'item') else v)

            auto_alpha = self.auto_alpha_var.get()

            for base, p in pairs.items():
                down_k, up_k = p['down'], p['up']
                A, B = sd[down_k], sd[up_k]
                r = p['rank']

                alpha = alpha_dict.get(base, None)
                if alpha is None and auto_alpha:
                    alpha = global_alpha if global_alpha is not None else float(r)
                elif alpha is None:
                    alpha = float(r)

                scale = (alpha / r) if r > 0 else 1.0
                calc_res = self.compute_lora_metrics(A, B, scale=scale)

                layers[base] = {
                    'rank': r,
                    'alpha': alpha,
                    'scale': scale,
                    'fro_raw': calc_res['fro_raw'],
                    'fro_eff': calc_res['fro_eff'],
                    'spec_max_raw': calc_res['spec_max_raw'],
                    'spec_mean_raw': calc_res['spec_mean_raw'],
                    'spec_max_eff': calc_res['spec_max_eff'],
                    'spec_mean_eff': calc_res['spec_mean_eff'],
                    'a_l2': calc_res['a_l2'],
                    'b_spec': calc_res['b_spec'],
                    'stable_rank': calc_res['stable_rank'],
                }
            return layers
        except Exception as e:
            messagebox.showerror("Error", f"Error loading {path}:\n{e}")
            return {}

    def calc_pct_diff(self, val1, val2):
        if val1 == 0 and val2 == 0: return 0.0
        if val1 == 0: return 100.0 
        return ((val2 - val1) / val1) * 100.0

    def format_tree_columns(self, cols, headers):
        self.tree["columns"] = cols
        self.current_columns = headers
        for col, head in zip(cols, headers):
            self.tree.heading(col, text=head)
            if col == "layer":
                width = 280
                anchor = tk.W
            elif col in ["status", "config"]:
                width = 160
                anchor = tk.CENTER
            else:
                width = 105
                anchor = tk.CENTER
            self.tree.column(col, width=width, anchor=anchor)

    def run_analysis(self):
        r1 = self.get_metrics(self.file1_path.get())
        if not r1: return
        self.cached_r1 = r1

        if self.mode_var.get() == "compare":
            r2 = self.get_metrics(self.file2_path.get())
            if not r2: return
            self.cached_r2 = r2
        else:
            self.cached_r2 = {}

        self.refresh_display()

    def refresh_display(self):
        if not self.cached_r1:
            return

        for item in self.tree.get_children(): 
            self.tree.delete(item)
        self.results_data = []

        is_raw = "Raw" in self.norm_view_var.get()
        mode = self.mode_var.get()
        norm_label = "Raw" if is_raw else "Eff"

        if mode == "single":
            cols = (
                "layer", "rank", "alpha", "scale", "fro_norm",
                "spec_max", "spec_mean", "a_l2", "b_spec", "stable_rank", "status"
            )
            headers = [
                "Layer Name", "Rank (r)", "Alpha (α)", "Scale (α/r)",
                f"‖ΔW‖_F ({norm_label})", f"σ_Max ({norm_label})", f"σ_Mean ({norm_label})",
                "‖A‖_F", "σ_Max(B)", "Stable Rank", "Health Status"
            ]
            self.format_tree_columns(cols, headers)

            collapsed_count = 0
            n_layers = len(self.cached_r1)

            # Accumulators for Global Totals & Averages
            tot_rank = 0
            tot_alpha = 0.0
            tot_scale = 0.0
            tot_fro = 0.0
            tot_fro_sq = 0.0
            tot_spec_max = 0.0
            peak_spec_max = 0.0
            tot_spec_mean = 0.0
            tot_a = 0.0
            tot_a_sq = 0.0
            tot_b = 0.0
            peak_b_spec = 0.0
            tot_srank = 0.0

            for k in sorted(self.cached_r1.keys()):
                d = self.cached_r1[k]
                r = d['rank']
                s_rank = d['stable_rank']
                cur_spec = d['spec_max_eff']

                if cur_spec < 1e-6:
                    status = "DEAD (Zero Update)"
                    tag = "dead"
                elif s_rank < 1.3 and r >= 4:
                    status = f"COLLAPSED ({s_rank:.2f})"
                    tag = "collapsed"
                    collapsed_count += 1
                else:
                    status = f"Optimal ({s_rank:.2f})"
                    tag = "safe"

                disp_fro = d['fro_raw'] if is_raw else d['fro_eff']
                disp_spec_max = d['spec_max_raw'] if is_raw else d['spec_max_eff']
                disp_spec_mean = d['spec_mean_raw'] if is_raw else d['spec_mean_eff']
                a_norm = d['a_l2']
                b_norm = d['b_spec']

                # Accumulate
                tot_rank += r
                tot_alpha += d['alpha']
                tot_scale += d['scale']
                tot_fro += disp_fro
                tot_fro_sq += (disp_fro ** 2)
                tot_spec_max += disp_spec_max
                peak_spec_max = max(peak_spec_max, disp_spec_max)
                tot_spec_mean += disp_spec_mean
                tot_a += a_norm
                tot_a_sq += (a_norm ** 2)
                tot_b += b_norm
                peak_b_spec = max(peak_b_spec, b_norm)
                tot_srank += s_rank

                row_dict = {
                    "layer": k,
                    "rank": r,
                    "fro_norm": disp_fro,
                    "spec_max": disp_spec_max,
                    "spec_mean": disp_spec_mean,
                    "a_l2": a_norm,
                    "b_spec": b_norm,
                    "stable_rank": s_rank,
                    "status": status,
                    "is_raw_view": is_raw
                }
                self.results_data.append(row_dict)

                self.tree.insert("", "end", values=(
                    k, r, f"{d['alpha']:.1f}", f"{d['scale']:.3f}",
                    f"{disp_fro:.4f}", f"{disp_spec_max:.4f}", f"{disp_spec_mean:.4f}",
                    f"{a_norm:.4f}", f"{b_norm:.4f}",
                    f"{s_rank:.2f}/{r}", status
                ), tags=(tag,))

            # Compute Final Global Metrics
            global_rms_fro = math.sqrt(tot_fro_sq)
            global_rms_a = math.sqrt(tot_a_sq)
            avg_r = tot_rank / n_layers if n_layers else 0
            avg_alpha = tot_alpha / n_layers if n_layers else 0
            avg_scale = tot_scale / n_layers if n_layers else 0
            avg_fro = tot_fro / n_layers if n_layers else 0
            avg_spec_max = tot_spec_max / n_layers if n_layers else 0
            avg_spec_mean = tot_spec_mean / n_layers if n_layers else 0
            avg_a = tot_a / n_layers if n_layers else 0
            avg_b = tot_b / n_layers if n_layers else 0
            avg_srank = tot_srank / n_layers if n_layers else 0

            self.global_stats = {
                "n_layers": n_layers, "tot_fro": tot_fro, "rms_fro": global_rms_fro, "avg_fro": avg_fro,
                "peak_spec_max": peak_spec_max, "avg_spec_max": avg_spec_max, "avg_spec_mean": avg_spec_mean,
                "tot_a": tot_a, "rms_a": global_rms_a, "avg_a": avg_a,
                "peak_b": peak_b_spec, "avg_b": avg_b, "avg_srank": avg_srank, "avg_r": avg_r
            }

            # Insert Pinned Summary Row at Index 0
            self.tree.insert("", 0, values=(
                "★ [GLOBAL TOTAL / MEAN]",
                f"μ={avg_r:.1f} (Σ={tot_rank})",
                f"μ={avg_alpha:.1f}",
                f"μ={avg_scale:.3f}",
                f"Σ={tot_fro:.3f} | RMS={global_rms_fro:.3f}",
                f"Max={peak_spec_max:.3f} | μ={avg_spec_max:.3f}",
                f"μ={avg_spec_mean:.3f}",
                f"Σ={tot_a:.3f} | RMS={global_rms_a:.3f}",
                f"Max={peak_b_spec:.3f} | μ={avg_b:.3f}",
                f"μ={avg_srank:.2f}",
                f"Collapsed: {collapsed_count}/{n_layers}"
            ), tags=("summary",))

            # Update Banner Text
            banner = (
                f"Layers: {n_layers}  |  "
                f"Total ‖ΔW‖: {tot_fro:.3f} (RMS: {global_rms_fro:.3f}, μ: {avg_fro:.3f})  |  "
                f"Peak σ_Max: {peak_spec_max:.3f} (μ: {avg_spec_max:.3f})  |  "
                f"Total ‖A‖: {tot_a:.3f}  |  "
                f"Peak σ(B): {peak_b_spec:.3f} (μ: {avg_b:.3f})  |  "
                f"Avg s_Rank: {avg_srank:.2f}  |  "
                f"Avg Rank: {avg_r:.1f}"
            )
            self.global_summary_text.set(banner)
            self.global_diff_var.set(f"Loaded {n_layers} layers | View: {self.norm_view_var.get()} | Collapsed: {collapsed_count}")

        else: # Compare Mode
            r1, r2 = self.cached_r1, self.cached_r2
            cols = (
                "layer", "fro_1", "fro_2", "diff_fro", "spec_1", "spec_2", "diff_spec",
                "a_l2_1", "a_l2_2", "diff_a", "b_spec_1", "b_spec_2", "diff_b",
                "srank_1", "srank_2", "diff_srank", "status"
            )
            headers = [
                "Layer Name",
                f"LoRA1 ‖ΔW‖({norm_label})", f"LoRA2 ‖ΔW‖({norm_label})", "Δ ‖ΔW‖%",
                f"LoRA1 σ_Max({norm_label})", f"LoRA2 σ_Max({norm_label})", "Δ σ_Max%",
                "LoRA1 ‖A‖_F", "LoRA2 ‖A‖_F", "Δ ‖A‖%",
                "LoRA1 σ(B)", "LoRA2 σ(B)", "Δ σ(B)%",
                "LoRA1 s_Rank", "LoRA2 s_Rank", "Δ s_Rank%",
                "Status (1|2)"
            ]
            self.format_tree_columns(cols, headers)

            all_keys = sorted(set(list(r1.keys()) + list(r2.keys())))
            n_layers = len(all_keys)
            
            # Totals accumulator dict
            g = {key: 0.0 for key in [
                "fro_1", "fro_2", "fro_sq_1", "fro_sq_2",
                "spec_1", "spec_2", "peak_spec_1", "peak_spec_2",
                "a_1", "a_2", "a_sq_1", "a_sq_2",
                "b_1", "b_2", "peak_b_1", "peak_b_2",
                "srank_1", "srank_2"
            ]}

            for k in all_keys:
                d1 = r1.get(k, {'rank':0, 'fro_raw':0, 'fro_eff':0, 'spec_max_raw':0, 'spec_max_eff':0, 'a_l2':0, 'b_spec':0, 'stable_rank':1.0})
                d2 = r2.get(k, {'rank':0, 'fro_raw':0, 'fro_eff':0, 'spec_max_raw':0, 'spec_max_eff':0, 'a_l2':0, 'b_spec':0, 'stable_rank':1.0})

                f1 = d1['fro_raw'] if is_raw else d1['fro_eff']
                f2 = d2['fro_raw'] if is_raw else d2['fro_eff']
                s1 = d1['spec_max_raw'] if is_raw else d1['spec_max_eff']
                s2 = d2['spec_max_raw'] if is_raw else d2['spec_max_eff']
                a1, a2 = d1['a_l2'], d2['a_l2']
                b1, b2 = d1['b_spec'], d2['b_spec']
                sr1, sr2 = d1['stable_rank'], d2['stable_rank']

                diff_fro = self.calc_pct_diff(f1, f2)
                diff_spec = self.calc_pct_diff(s1, s2)
                diff_a = self.calc_pct_diff(a1, a2)
                diff_b = self.calc_pct_diff(b1, b2)
                diff_srank = self.calc_pct_diff(sr1, sr2)

                s1_stat = "Collapsed" if (sr1 < 1.3 and d1['rank'] >= 4) else "OK"
                s2_stat = "Collapsed" if (sr2 < 1.3 and d2['rank'] >= 4) else "OK"
                status_str = f"{s1_stat} | {s2_stat}"

                tag = "collapsed" if ("Collapsed" in status_str) else "safe"

                row_dict = {
                    "layer": k,
                    "fro_1": f1, "fro_2": f2, "diff_fro": diff_fro,
                    "spec_1": s1, "spec_2": s2, "diff_spec": diff_spec,
                    "a_l2_1": a1, "a_l2_2": a2, "diff_a": diff_a,
                    "b_spec_1": b1, "b_spec_2": b2, "diff_b": diff_b,
                    "srank_1": sr1, "srank_2": sr2, "diff_srank": diff_srank,
                    "status": status_str,
                    "is_raw_view": is_raw
                }
                self.results_data.append(row_dict)

                self.tree.insert("", "end", values=(
                    k, 
                    f"{f1:.4f}", f"{f2:.4f}", f"{diff_fro:+.2f}%",
                    f"{s1:.4f}", f"{s2:.4f}", f"{diff_spec:+.2f}%",
                    f"{a1:.4f}", f"{a2:.4f}", f"{diff_a:+.2f}%",
                    f"{b1:.4f}", f"{b2:.4f}", f"{diff_b:+.2f}%",
                    f"{sr1:.2f}", f"{sr2:.2f}", f"{diff_srank:+.2f}%",
                    status_str
                ), tags=(tag,))

                g["fro_1"] += f1; g["fro_2"] += f2
                g["fro_sq_1"] += (f1 ** 2); g["fro_sq_2"] += (f2 ** 2)
                g["spec_1"] += s1; g["spec_2"] += s2
                g["peak_spec_1"] = max(g["peak_spec_1"], s1)
                g["peak_spec_2"] = max(g["peak_spec_2"], s2)
                g["a_1"] += a1; g["a_2"] += a2
                g["a_sq_1"] += (a1 ** 2); g["a_sq_2"] += (a2 ** 2)
                g["b_1"] += b1; g["b_2"] += b2
                g["peak_b_1"] = max(g["peak_b_1"], b1)
                g["peak_b_2"] = max(g["peak_b_2"], b2)
                g["srank_1"] += sr1; g["srank_2"] += sr2

            # Compute Global Percentage Drifts
            gd_fro = self.calc_pct_diff(g["fro_1"], g["fro_2"])
            gd_spec = self.calc_pct_diff(g["spec_1"], g["spec_2"])
            gd_peak_spec = self.calc_pct_diff(g["peak_spec_1"], g["peak_spec_2"])
            gd_a = self.calc_pct_diff(g["a_1"], g["a_2"])
            gd_b = self.calc_pct_diff(g["b_1"], g["b_2"])
            gd_peak_b = self.calc_pct_diff(g["peak_b_1"], g["peak_b_2"])
            gd_srank = self.calc_pct_diff(g["srank_1"], g["srank_2"])

            rms_fro_1, rms_fro_2 = math.sqrt(g["fro_sq_1"]), math.sqrt(g["fro_sq_2"])
            rms_a_1, rms_a_2 = math.sqrt(g["a_sq_1"]), math.sqrt(g["a_sq_2"])
            avg_sr1 = g["srank_1"] / n_layers if n_layers else 0
            avg_sr2 = g["srank_2"] / n_layers if n_layers else 0

            self.global_stats = {
                "n_layers": n_layers,
                "fro_1": g["fro_1"], "fro_2": g["fro_2"], "diff_fro": gd_fro,
                "rms_fro_1": rms_fro_1, "rms_fro_2": rms_fro_2,
                "peak_spec_1": g["peak_spec_1"], "peak_spec_2": g["peak_spec_2"], "diff_peak_spec": gd_peak_spec,
                "avg_spec_1": g["spec_1"]/n_layers, "avg_spec_2": g["spec_2"]/n_layers, "diff_spec": gd_spec,
                "a_1": g["a_1"], "a_2": g["a_2"], "diff_a": gd_a,
                "rms_a_1": rms_a_1, "rms_a_2": rms_a_2,
                "peak_b_1": g["peak_b_1"], "peak_b_2": g["peak_b_2"], "diff_peak_b": gd_peak_b,
                "avg_b_1": g["b_1"]/n_layers, "avg_b_2": g["b_2"]/n_layers, "diff_b": gd_b,
                "srank_1": avg_sr1, "srank_2": avg_sr2, "diff_srank": gd_srank
            }

            # Insert Pinned Summary Row at Index 0
            self.tree.insert("", 0, values=(
                "★ [GLOBAL TOTAL / MEAN]",
                f"Σ={g['fro_1']:.2f}", f"Σ={g['fro_2']:.2f}", f"{gd_fro:+.2f}%",
                f"Max={g['peak_spec_1']:.2f}", f"Max={g['peak_spec_2']:.2f}", f"{gd_peak_spec:+.2f}%",
                f"Σ={g['a_1']:.2f}", f"Σ={g['a_2']:.2f}", f"{gd_a:+.2f}%",
                f"Max={g['peak_b_1']:.2f}", f"Max={g['peak_b_2']:.2f}", f"{gd_peak_b:+.2f}%",
                f"μ={avg_sr1:.2f}", f"μ={avg_sr2:.2f}", f"{gd_srank:+.2f}%",
                "GLOBAL TOTAL"
            ), tags=("summary",))

            # Update Banner
            banner = (
                f"Total ‖ΔW‖: {g['fro_1']:.2f} → {g['fro_2']:.2f} ({gd_fro:+.2f}%)  |  "
                f"Peak σ_Max: {g['peak_spec_1']:.2f} → {g['peak_spec_2']:.2f} ({gd_peak_spec:+.2f}%)  |  "
                f"Total ‖A‖: {g['a_1']:.2f} → {g['a_2']:.2f} ({gd_a:+.2f}%)  |  "
                f"Peak σ(B): {g['peak_b_1']:.2f} → {g['peak_b_2']:.2f} ({gd_peak_b:+.2f}%)  |  "
                f"Mean s_Rank: {avg_sr1:.2f} → {avg_sr2:.2f} ({gd_srank:+.2f}%)"
            )
            self.global_summary_text.set(banner)
            self.global_diff_var.set(f"Global Drift ({norm_label}) → ‖ΔW‖: {gd_fro:+.2f}% | σ_Max: {gd_spec:+.2f}% | ‖A‖: {gd_a:+.2f}% | σ(B): {gd_b:+.2f}%")

    def export_csv(self):
        if not self.results_data: 
            messagebox.showwarning("No Data", "Please run analysis first before exporting.")
            return
            
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if path:
            try:
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.current_columns)
                    # Automatically exports the top summary row along with every layer row
                    for item in self.tree.get_children():
                        writer.writerow(self.tree.item(item)['values'])
                messagebox.showinfo("Success", "Data exported successfully with Global Totals included!")
            except PermissionError:
                messagebox.showerror("Error", "Permission Denied! Make sure the file isn't open in another program.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export CSV:\n{str(e)}")

    def show_charts(self):
        if not self.results_data: 
            messagebox.showwarning("No Data", "Please run analysis first.")
            return
        
        is_raw = "Raw" in self.norm_view_var.get()
        norm_desc = "Raw (Unscaled)" if is_raw else "Effective (Scaled)"
        
        chart_win = tk.Toplevel(self.root)
        chart_win.title(f"LoRA Spectral & Factor Analytics [{norm_desc}]")
        chart_win.geometry("1240x780")
        chart_win.configure(bg=self.bg_color)
        
        notebook = ttk.Notebook(chart_win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        plt.style.use('dark_background')
        layers = [r["layer"].split('.')[-1][:15] for r in self.results_data]
        x = np.arange(len(layers))

        if self.mode_var.get() == "single":
            gs = self.global_stats

            # Tab 0: Dedicated Global Totals Bar Summary
            f_tot = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax_tot = f_tot.add_subplot(111)
            metrics = ["Total ‖ΔW‖_F", "RMS ‖ΔW‖_F", "Peak σ_Max", "Total ‖A‖_F", "Peak σ_Max(B)", "Mean s_Rank"]
            vals = [gs.get("tot_fro", 0), gs.get("rms_fro", 0), gs.get("peak_spec_max", 0), gs.get("tot_a", 0), gs.get("peak_b", 0), gs.get("avg_srank", 0)]
            colors = ["#42a5f5", "#29b6f6", "#ff7043", "#ab47bc", "#26a69a", "#66bb6a"]
            bars = ax_tot.bar(metrics, vals, color=colors, width=0.55)
            for bar, val in zip(bars, vals):
                ax_tot.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05 * (max(vals)+1e-5), f"{val:.2f}", ha='center', va='bottom', fontsize=9, color="white")
            ax_tot.set_title(f"Global Totals & Peak Analytics Summary [{norm_desc}]")
            ax_tot.set_ylabel("Metric Value")
            ax_tot.grid(True, alpha=0.15, axis='y')
            self.add_tab(notebook, f_tot, "★ Global Totals Summary")

            # Tab 1: Spectral Norm Max & Mean
            f_spec = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax_spec = f_spec.add_subplot(111)
            ax_spec.plot(x, [r["spec_max"] for r in self.results_data], label=f'σ_Max ({norm_desc})', marker='o', color='#ff7043', linewidth=1.5)
            ax_spec.plot(x, [r["spec_mean"] for r in self.results_data], label=f'σ_Mean ({norm_desc})', linestyle='--', color='#42a5f5')
            ax_spec.axhline(gs.get("avg_spec_max", 0), color='#ffd54f', linestyle=':', label=f'Global Mean σ_Max ({gs.get("avg_spec_max", 0):.2f})')
            ax_spec.set_title(f"LoRA ΔW Spectral Norm (Operator Norm) per Layer [{norm_desc}]")
            ax_spec.set_xticks(x); ax_spec.set_xticklabels(layers, rotation=90, fontsize=8)
            ax_spec.grid(True, alpha=0.2)
            ax_spec.legend()
            self.add_tab(notebook, f_spec, f"ΔW Spectral Norm ({norm_desc})")

            # Tab 2: Factor A & Factor B Metrics
            f_factors = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax_factors = f_factors.add_subplot(111)
            ax_factors.plot(x, [r["a_l2"] for r in self.results_data], label='Factor A L2 Norm (‖A‖_F)', marker='o', color='#ab47bc', linewidth=1.5)
            ax_factors.plot(x, [r["b_spec"] for r in self.results_data], label='Factor B Spectral Norm (σ_Max(B))', marker='^', color='#26a69a', linewidth=1.5)
            ax_factors.set_title("LoRA Factor Decomposition: ‖A‖_F and σ_Max(B)")
            ax_factors.set_xticks(x); ax_factors.set_xticklabels(layers, rotation=90, fontsize=8)
            ax_factors.grid(True, alpha=0.2)
            ax_factors.legend()
            self.add_tab(notebook, f_factors, "Factor Metrics (A & B)")

            # Tab 3: Stable Rank
            f_rank = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax_rank = f_rank.add_subplot(111)
            ax_rank.plot(x, [r["stable_rank"] for r in self.results_data], label='Stable Rank (‖ΔW‖_F² / σ_max²)', marker='s', color='#66bb6a')
            ax_rank.plot(x, [r["rank"] for r in self.results_data], label='Configured Rank (r)', linestyle=':', color='#9e9e9e')
            ax_rank.axhline(gs.get("avg_srank", 0), color='#ffd54f', linestyle=':', label=f'Global Avg s_Rank ({gs.get("avg_srank", 0):.2f})')
            ax_rank.axhline(1.0, color='#f44336', linestyle='--', label='Complete Rank Collapse (1.0)', linewidth=1.2)
            ax_rank.set_title("LoRA Effective Rank Dimension Utilization (Stable Rank)")
            ax_rank.set_xticks(x); ax_rank.set_xticklabels(layers, rotation=90, fontsize=8)
            ax_rank.grid(True, alpha=0.2)
            ax_rank.legend()
            self.add_tab(notebook, f_rank, "Stable Rank")

            # Tab 4: Frobenius Norm
            f1 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax1 = f1.add_subplot(111)
            ax1.bar(x, [r["fro_norm"] for r in self.results_data], 0.5, label=f'‖ΔW‖_F [{norm_desc}]', color='#64b5f6')
            ax1.axhline(gs.get("avg_fro", 0), color='#ffd54f', linestyle=':', label=f'Global Mean ‖ΔW‖_F ({gs.get("avg_fro", 0):.2f})')
            ax1.set_title(f"LoRA Frobenius Weight Magnitude per Layer [{norm_desc}]")
            ax1.set_xticks(x); ax1.set_xticklabels(layers, rotation=90, fontsize=8)
            ax1.legend()
            self.add_tab(notebook, f1, f"Frobenius Norm ({norm_desc})")

        else: # Compare Mode
            gs = self.global_stats

            # Tab 0: Global Totals Comparison Chart
            f_tot_c = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax_tot_c = f_tot_c.add_subplot(111)
            comp_metrics = ["Total ‖ΔW‖", "Peak σ_Max", "Total ‖A‖", "Peak σ(B)", "Mean s_Rank"]
            v1 = [gs.get("fro_1", 0), gs.get("peak_spec_1", 0), gs.get("a_1", 0), gs.get("peak_b_1", 0), gs.get("srank_1", 0)]
            v2 = [gs.get("fro_2", 0), gs.get("peak_spec_2", 0), gs.get("a_2", 0), gs.get("peak_b_2", 0), gs.get("srank_2", 0)]
            
            x_ind = np.arange(len(comp_metrics))
            width = 0.35
            ax_tot_c.bar(x_ind - width/2, v1, width, label='LoRA 1 (Baseline)', color='#64b5f6')
            ax_tot_c.bar(x_ind + width/2, v2, width, label='LoRA 2 (Target)', color='#ff7043')
            
            diffs = [gs.get("diff_fro", 0), gs.get("diff_peak_spec", 0), gs.get("diff_a", 0), gs.get("diff_peak_b", 0), gs.get("diff_srank", 0)]
            for i, d in enumerate(diffs):
                top_val = max(v1[i], v2[i])
                ax_tot_c.text(x_ind[i], top_val + 0.05 * (top_val + 1e-4), f"{d:+.1f}%", ha='center', va='bottom', fontsize=9, color="#ffd54f", fontweight='bold')
                
            ax_tot_c.set_xticks(x_ind); ax_tot_c.set_xticklabels(comp_metrics)
            ax_tot_c.set_title(f"Global Totals & Averages Comparison [{norm_desc}]")
            ax_tot_c.grid(True, alpha=0.15, axis='y')
            ax_tot_c.legend()
            self.add_tab(notebook, f_tot_c, "★ Global Totals Comparison")

            # Tab 1: Spectral Norm Compare
            f_spec_c = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax_spec_c = f_spec_c.add_subplot(111)
            ax_spec_c.plot(x, [r["spec_1"] for r in self.results_data], label=f'LoRA 1 - σ_Max ({norm_desc})', alpha=0.85, color='#64b5f6', marker='o')
            ax_spec_c.plot(x, [r["spec_2"] for r in self.results_data], label=f'LoRA 2 - σ_Max ({norm_desc})', alpha=0.85, color='#ff7043', marker='s')
            ax_spec_c.fill_between(x, [r["spec_1"] for r in self.results_data], [r["spec_2"] for r in self.results_data], color='gray', alpha=0.2)
            ax_spec_c.set_title(f"Spectral Norm Comparison [{norm_desc}]")
            ax_spec_c.set_xticks(x); ax_spec_c.set_xticklabels(layers, rotation=90, fontsize=8)
            ax_spec_c.legend()
            self.add_tab(notebook, f_spec_c, f"Spectral Norm ({norm_desc})")

            # Tab 2: Factor A L2 Compare
            f_a = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax_a = f_a.add_subplot(111)
            ax_a.plot(x, [r["a_l2_1"] for r in self.results_data], label='LoRA 1 - ‖A‖_F', alpha=0.85, color='#ab47bc', marker='o')
            ax_a.plot(x, [r["a_l2_2"] for r in self.results_data], label='LoRA 2 - ‖A‖_F', alpha=0.85, color='#ffca28', marker='s')
            ax_a.fill_between(x, [r["a_l2_1"] for r in self.results_data], [r["a_l2_2"] for r in self.results_data], color='gray', alpha=0.2)
            ax_a.set_title("Factor A L2 Norm Comparison (‖A‖_F)")
            ax_a.set_xticks(x); ax_a.set_xticklabels(layers, rotation=90, fontsize=8)
            ax_a.legend()
            self.add_tab(notebook, f_a, "Factor A L2 Compare")

            # Tab 3: Factor B Spectral Compare
            f_b = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax_b = f_b.add_subplot(111)
            ax_b.plot(x, [r["b_spec_1"] for r in self.results_data], label='LoRA 1 - σ_Max(B)', alpha=0.85, color='#26a69a', marker='o')
            ax_b.plot(x, [r["b_spec_2"] for r in self.results_data], label='LoRA 2 - σ_Max(B)', alpha=0.85, color='#ef5350', marker='s')
            ax_b.fill_between(x, [r["b_spec_1"] for r in self.results_data], [r["b_spec_2"] for r in self.results_data], color='gray', alpha=0.2)
            ax_b.set_title("Factor B Spectral Norm Comparison (σ_Max(B))")
            ax_b.set_xticks(x); ax_b.set_xticklabels(layers, rotation=90, fontsize=8)
            ax_b.legend()
            self.add_tab(notebook, f_b, "Factor B Spectral Compare")

            # Tab 4: Stable Rank Compare
            f2 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax2 = f2.add_subplot(111)
            ax2.plot(x, [r["srank_1"] for r in self.results_data], label='LoRA 1 - Stable Rank', alpha=0.85, color='#f44336', marker='o')
            ax2.plot(x, [r["srank_2"] for r in self.results_data], label='LoRA 2 - Stable Rank', alpha=0.85, color='#4caf50', marker='s')
            ax2.fill_between(x, [r["srank_1"] for r in self.results_data], [r["srank_2"] for r in self.results_data], color='gray', alpha=0.2)
            ax2.set_title("Stable Rank Dimension Utilization Comparison")
            ax2.set_xticks(x); ax2.set_xticklabels(layers, rotation=90, fontsize=8)
            ax2.legend()
            self.add_tab(notebook, f2, "Stable Rank Compare")

            # Tab 5: Drift Scatter Plot (%)
            f3 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax3 = f3.add_subplot(111)
            ax3.scatter(x, [r["diff_fro"] for r in self.results_data], label=f'Δ ‖ΔW‖_F % [{norm_desc}]', color='#ba68c8')
            ax3.scatter(x, [r["diff_spec"] for r in self.results_data], label=f'Δ σ_Max % [{norm_desc}]', color='#ff7043', marker='x')
            ax3.scatter(x, [r["diff_a"] for r in self.results_data], label='Δ ‖A‖_F %', color='#ffca28', marker='^')
            ax3.scatter(x, [r["diff_b"] for r in self.results_data], label='Δ σ_Max(B) %', color='#26a69a', marker='v')
            ax3.axhline(0, color='white', linestyle='--', linewidth=1)
            ax3.set_title(f"Percentage Drift Across Layers [{norm_desc}]")
            ax3.set_xticks(x); ax3.set_xticklabels(layers, rotation=90, fontsize=8)
            ax3.legend()
            self.add_tab(notebook, f3, "Drift Analytics (%)")

    def add_tab(self, notebook, figure, title):
        frame = tk.Frame(notebook, bg=self.bg_color)
        figure.tight_layout()
        canvas = FigureCanvasTkAgg(figure, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        notebook.add(frame, text=title)

if __name__ == "__main__":
    root = tk.Tk()
    app = LoRAInspectorApp(root)
    root.mainloop()