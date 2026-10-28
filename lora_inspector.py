import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import torch
from safetensors.torch import load_file
from safetensors import safe_open
import os
import csv
import json
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class LoRAInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LoRA Weight Inspector Pro + Spectral & Rank Analytics")
        self.root.geometry("1580x920")
        
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
        
        # Configuration & Thresholds
        self.spectral_warn_var = tk.DoubleVar(value=5.0)
        self.auto_alpha_var = tk.BooleanVar(value=True)
        self.norm_view_var = tk.StringVar(value="Effective (Scaled)")

        # In-memory cached analysis results
        self.cached_r1 = {}
        self.cached_r2 = {}
        self.results_data = [] 
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

        # Spectral Warnings
        tk.Label(opt_frame, text="Warn Thresh (σ):", bg=self.bg_color, fg="#ffb74d", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(8, 2))
        tk.Entry(opt_frame, textvariable=self.spectral_warn_var, width=5, bg=self.entry_bg, fg=self.fg_color, insertbackground="white").pack(side=tk.LEFT, padx=3)
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

        # --- Button Bar & Global Stats ---
        btn_frame = tk.Frame(self.root, pady=8, bg=self.bg_color)
        btn_frame.pack(fill=tk.X)
        
        tk.Button(btn_frame, text="1. Run Analysis", command=self.run_analysis, bg="#2e7d32", fg="white", width=18, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="2. Show Analytics", command=self.show_charts, bg="#1565c0", fg="white", width=18, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="3. Export CSV", command=self.export_csv, bg="#ef6c00", fg="white", width=18, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)

        tk.Label(btn_frame, textvariable=self.global_diff_var, font=("Arial", 10, "bold"), fg="#ffb74d", bg=self.bg_color).pack(side=tk.RIGHT, padx=20)

        # --- Table Frame ---
        bottom_frame = tk.Frame(self.root, padx=10, pady=5, bg=self.bg_color)
        bottom_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(bottom_frame, show="headings")
        
        y_scroll = ttk.Scrollbar(bottom_frame, orient=tk.VERTICAL, command=self.tree.yview)
        x_scroll = ttk.Scrollbar(bottom_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscroll=y_scroll.set, xscroll=x_scroll.set)
        
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Tree tags for status color highlights
        self.tree.tag_configure("high_energy", foreground="#ff5252")  # Red
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
            has_dora = any('magnitude' in k or 'dora_scale' in k for k in sd.keys())
            
            # Count paired layers
            pairs = self.find_lora_pairs(sd)
            ranks = [p['rank'] for p in pairs.values() if 'rank' in p]
            avg_rank = f"{sum(ranks)/len(ranks):.1f}" if ranks else "N/A"

            info_str = f"Pairs Detected: {len(pairs)} | Avg Rank: {avg_rank} | DoRA: {has_dora} | Metadata Keys: {len(metadata)}"
            info_var.set(info_str)
        except Exception as e:
            info_var.set(f"Could not read metadata: {e}")

    def find_lora_pairs(self, sd):
        """
        Pairs A/down and B/up weight tensors across PEFT, Kohya, Diffusers, and LyCORIS formats.
        """
        pairs = {}
        for k in sd.keys():
            k_lower = k.lower()
            if any(term in k_lower for term in ['alpha', 'magnitude_vector', 'dora_scale', 'bias']):
                continue

            # Case 1: Down / A weights
            if any(term in k for term in ['.lora_A.', '.lora_down.', '.down.', '.linear_1.']):
                base = k.replace('.lora_A.weight', '').replace('.lora_A.default.weight', '') \
                        .replace('.lora_down.weight', '').replace('.down.weight', '') \
                        .replace('.linear_1.weight', '')
                pairs.setdefault(base, {})['down'] = k
            
            # Case 2: Up / B weights
            elif any(term in k for term in ['.lora_B.', '.lora_up.', '.up.', '.linear_2.']):
                base = k.replace('.lora_B.weight', '').replace('.lora_B.default.weight', '') \
                        .replace('.lora_up.weight', '').replace('.up.weight', '') \
                        .replace('.linear_2.weight', '')
                pairs.setdefault(base, {})['up'] = k

        # Validate complete pairs and determine rank
        valid_pairs = {}
        for base, p in pairs.items():
            if 'down' in p and 'up' in p:
                t_down = sd[p['down']]
                rank = t_down.shape[0] if len(t_down.shape) >= 2 else 0
                p['rank'] = rank
                valid_pairs[base] = p

        return valid_pairs

    @staticmethod
    def compute_fast_lora_spectral_metrics(A_tensor: torch.Tensor, B_tensor: torch.Tensor, scale: float = 1.0):
        """
        Fast Exact SVD for LoRA via Reduced QR-Decomposition:
        Since rank r << d_in, d_out, non-zero singular values of B @ A
        match the singular values of the tiny (r x r) matrix: R_B @ R_A.T
        """
        A = A_tensor.float()
        B = B_tensor.float()

        # Handle Conv2d LoRA (flatten spatial dimensions if present)
        if A.dim() > 2:
            A = A.flatten(start_dim=1)
        if B.dim() > 2:
            B = B.flatten(start_dim=1)

        r, d_in = A.shape
        d_out, r_b = B.shape
        if r != r_b or r == 0:
            return {"spec_max_eff": 0.0, "spec_mean_eff": 0.0, "spec_max_raw": 0.0,
                    "spec_mean_raw": 0.0, "fro_eff": 0.0, "fro_raw": 0.0, "stable_rank": 1.0}

        # Reduced QR decompositions
        try:
            Q_A, R_A = torch.linalg.qr(A.t(), mode='reduced')  # Q_A: (d_in, r), R_A: (r, r)
            Q_B, R_B = torch.linalg.qr(B, mode='reduced')      # Q_B: (d_out, r), R_B: (r, r)
            
            # Tiny r x r intermediate core
            M = torch.mm(R_B, R_A.t())
            svs = torch.linalg.svdvals(M) # Exact singular values of (B @ A)
            
            spec_max_raw = svs[0].item() if svs.numel() > 0 else 0.0
            spec_mean_raw = svs.mean().item() if svs.numel() > 0 else 0.0
            fro_raw = torch.linalg.norm(svs).item()

            spec_max_eff = spec_max_raw * scale
            spec_mean_eff = spec_mean_raw * scale
            fro_eff = fro_raw * scale

            # Stable Rank = ||W||_F^2 / sigma_max^2 (Ranges from 1.0 to r)
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

            # Read global alpha metadata if present
            global_alpha = None
            if "lora_alpha" in metadata:
                try: global_alpha = float(metadata["lora_alpha"])
                except Exception: pass

            # Pre-extract DoRA multipliers and Alphas
            dora_mults = {}
            alpha_dict = {}

            for k, v in sd.items():
                if 'magnitude_vector' in k or 'dora_scale' in k:
                    base = k.replace('.lora_magnitude_vector', '').replace('.dora_scale', '')
                    dora_mults[base] = v.float()
                elif 'alpha' in k:
                    base = k.replace('.alpha', '')
                    alpha_dict[base] = float(v.item() if hasattr(v, 'item') else v)

            auto_alpha = self.auto_alpha_var.get()

            for base, p in pairs.items():
                down_k, up_k = p['down'], p['up']
                A, B = sd[down_k], sd[up_k]
                r = p['rank']

                # Locate Alpha
                alpha = alpha_dict.get(base, None)
                if alpha is None and auto_alpha:
                    alpha = global_alpha if global_alpha is not None else float(r)
                elif alpha is None:
                    alpha = float(r)

                scale = (alpha / r) if r > 0 else 1.0

                calc_res = self.compute_fast_lora_spectral_metrics(A, B, scale=scale)

                # DoRA Magnitude metrics
                has_dora = base in dora_mults
                if has_dora and dora_mults[base].numel() > 0:
                    dora_t = dora_mults[base]
                    dora_mean = torch.mean(dora_t).item()
                    dora_min = torch.min(dora_t).item()
                    dora_max = torch.max(dora_t).item()
                else:
                    dora_mean = 1.0; dora_min = 1.0; dora_max = 1.0

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
                    'stable_rank': calc_res['stable_rank'],
                    'has_dora': has_dora,
                    'dora_mean': dora_mean,
                    'dora_min': dora_min,
                    'dora_max': dora_max,
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
                width = 300
                anchor = tk.W
            elif col in ["status", "config"]:
                width = 170
                anchor = tk.CENTER
            elif col in ["dora_1", "dora_2"]:
                width = 175
                anchor = tk.CENTER
            elif col in ["dora_mean", "dora_min", "dora_max"]:
                width = 95
                anchor = tk.CENTER
            else:
                width = 110
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
        warn_thresh = self.spectral_warn_var.get()

        if mode == "single":
            cols = (
                "layer", "rank", "alpha", "scale", "fro_norm",
                "spec_max", "spec_mean", "stable_rank", "status",
                "dora_mean", "dora_min", "dora_max", "config"
            )
            headers = [
                "Layer Name", "Rank (r)", "Alpha (α)", "Scale (α/r)",
                f"‖ΔW‖_F ({norm_label})", f"σ_Max ({norm_label})", f"σ_Mean ({norm_label})",
                "Stable Rank", "Health Status",
                "DoRA Mean", "DoRA Min", "DoRA Max", "Config"
            ]
            self.format_tree_columns(cols, headers)

            high_energy_count = 0
            collapsed_count = 0

            for k in sorted(self.cached_r1.keys()):
                d = self.cached_r1[k]
                r = d['rank']
                s_rank = d['stable_rank']
                cur_spec = d['spec_max_eff']

                # Health / Diagnostic Status
                if cur_spec < 1e-6:
                    status = "DEAD (Zero Update)"
                    tag = "dead"
                elif cur_spec >= warn_thresh:
                    status = f"HIGH ENERGY (σ={cur_spec:.2f})"
                    tag = "high_energy"
                    high_energy_count += 1
                elif s_rank < 1.3 and r >= 4:
                    status = f"COLLAPSED (s_rank={s_rank:.2f})"
                    tag = "collapsed"
                    collapsed_count += 1
                else:
                    status = f"Optimal (s_rank={s_rank:.2f})"
                    tag = "safe"

                disp_fro = d['fro_raw'] if is_raw else d['fro_eff']
                disp_spec_max = d['spec_max_raw'] if is_raw else d['spec_max_eff']
                disp_spec_mean = d['spec_mean_raw'] if is_raw else d['spec_mean_eff']

                config_str = "DoRA" if d['has_dora'] else "Standard LoRA"

                row_dict = {
                    "layer": k,
                    "rank": r,
                    "fro_norm": disp_fro,
                    "spec_max": disp_spec_max,
                    "spec_mean": disp_spec_mean,
                    "stable_rank": s_rank,
                    "dora_mean": d['dora_mean'],
                    "dora_min": d['dora_min'],
                    "dora_max": d['dora_max'],
                    "has_dora": d['has_dora'],
                    "status": status,
                    "is_raw_view": is_raw
                }
                self.results_data.append(row_dict)

                self.tree.insert("", "end", values=(
                    k, r, f"{d['alpha']:.1f}", f"{d['scale']:.3f}",
                    f"{disp_fro:.4f}", f"{disp_spec_max:.4f}", f"{disp_spec_mean:.4f}",
                    f"{s_rank:.2f}/{r}", status,
                    f"{d['dora_mean']:.4f}" if d['has_dora'] else "None",
                    f"{d['dora_min']:.4f}" if d['has_dora'] else "None",
                    f"{d['dora_max']:.4f}" if d['has_dora'] else "None",
                    config_str
                ), tags=(tag,))
            
            self.global_diff_var.set(f"Loaded {len(self.cached_r1)} layers | View: {self.norm_view_var.get()} | High-Energy: {high_energy_count} | Collapsed: {collapsed_count}")

        else: # Compare Mode
            r1, r2 = self.cached_r1, self.cached_r2
            cols = (
                "layer", "fro_1", "fro_2", "diff_fro", "spec_1", "spec_2", "diff_spec",
                "srank_1", "srank_2", "diff_srank", "status", "dora_1", "dora_2", "diff_dora"
            )
            headers = [
                "Layer Name",
                f"LoRA1 ‖ΔW‖({norm_label})", f"LoRA2 ‖ΔW‖({norm_label})", "Δ ‖ΔW‖%",
                f"LoRA1 σ_Max({norm_label})", f"LoRA2 σ_Max({norm_label})", "Δ σ_Max%",
                "LoRA1 s_Rank", "LoRA2 s_Rank", "Δ s_Rank%",
                "Status (1|2)", "LoRA1 DoRA (Mean [Min,Max])", "LoRA2 DoRA (Mean [Min,Max])", "Δ DoRA%"
            ]
            self.format_tree_columns(cols, headers)

            all_keys = sorted(set(list(r1.keys()) + list(r2.keys())))
            g = {key: 0.0 for key in ["fro_1", "fro_2", "spec_1", "spec_2", "srank_1", "srank_2", "dora_1", "dora_2"]}
            any_dora = False

            for k in all_keys:
                d1 = r1.get(k, {'rank':0, 'fro_raw':0, 'fro_eff':0, 'spec_max_raw':0, 'spec_max_eff':0, 'spec_mean_eff':0, 'stable_rank':1.0, 'has_dora':False, 'dora_mean':1.0, 'dora_min':1.0, 'dora_max':1.0})
                d2 = r2.get(k, {'rank':0, 'fro_raw':0, 'fro_eff':0, 'spec_max_raw':0, 'spec_max_eff':0, 'spec_mean_eff':0, 'stable_rank':1.0, 'has_dora':False, 'dora_mean':1.0, 'dora_min':1.0, 'dora_max':1.0})
                
                if d1['has_dora'] or d2['has_dora']:
                    any_dora = True

                f1 = d1['fro_raw'] if is_raw else d1['fro_eff']
                f2 = d2['fro_raw'] if is_raw else d2['fro_eff']
                s1 = d1['spec_max_raw'] if is_raw else d1['spec_max_eff']
                s2 = d2['spec_max_raw'] if is_raw else d2['spec_max_eff']
                sr1 = d1['stable_rank']
                sr2 = d2['stable_rank']

                diff_fro = self.calc_pct_diff(f1, f2)
                diff_spec = self.calc_pct_diff(s1, s2)
                diff_srank = self.calc_pct_diff(sr1, sr2)
                diff_dora = self.calc_pct_diff(d1['dora_mean'], d2['dora_mean'])

                status_str = f"{'High' if s1 >= warn_thresh else 'OK'} | {'High' if s2 >= warn_thresh else 'OK'}"
                
                dora1_disp = f"{d1['dora_mean']:.4f} [{d1['dora_min']:.3f}, {d1['dora_max']:.3f}]" if d1['has_dora'] else "None"
                dora2_disp = f"{d2['dora_mean']:.4f} [{d2['dora_min']:.3f}, {d2['dora_max']:.3f}]" if d2['has_dora'] else "None"

                tag = "safe"
                if s1 >= warn_thresh or s2 >= warn_thresh:
                    tag = "high_energy"
                elif (sr1 < 1.3 and d1['rank'] >= 4) or (sr2 < 1.3 and d2['rank'] >= 4):
                    tag = "collapsed"

                row_dict = {
                    "layer": k,
                    "fro_1": f1, "fro_2": f2, "diff_fro": diff_fro,
                    "spec_1": s1, "spec_2": s2, "diff_spec": diff_spec,
                    "srank_1": sr1, "srank_2": sr2, "diff_srank": diff_srank,
                    "status": status_str,
                    "dora_1": d1['dora_mean'], "dora_2": d2['dora_mean'],
                    "dora_min_1": d1['dora_min'], "dora_max_1": d1['dora_max'],
                    "dora_min_2": d2['dora_min'], "dora_max_2": d2['dora_max'],
                    "has_dora": d1['has_dora'] or d2['has_dora'],
                    "diff_dora": diff_dora,
                    "is_raw_view": is_raw
                }
                self.results_data.append(row_dict)

                self.tree.insert("", "end", values=(
                    k, 
                    f"{f1:.4f}", f"{f2:.4f}", f"{diff_fro:+.2f}%",
                    f"{s1:.4f}", f"{s2:.4f}", f"{diff_spec:+.2f}%",
                    f"{sr1:.2f}", f"{sr2:.2f}", f"{diff_srank:+.2f}%",
                    status_str, dora1_disp, dora2_disp, f"{diff_dora:+.2f}%"
                ), tags=(tag,))

                g["fro_1"] += f1; g["fro_2"] += f2
                g["spec_1"] += s1; g["spec_2"] += s2
                g["srank_1"] += sr1; g["srank_2"] += sr2
                g["dora_1"] += d1['dora_mean']; g["dora_2"] += d2['dora_mean']

            gd_fro = self.calc_pct_diff(g["fro_1"], g["fro_2"])
            gd_spec = self.calc_pct_diff(g["spec_1"], g["spec_2"])
            gd_srank = self.calc_pct_diff(g["srank_1"], g["srank_2"])
            gd_dora = self.calc_pct_diff(g["dora_1"], g["dora_2"])
            
            dora_part = f"  |  DoRA: {gd_dora:+.2f}%" if any_dora else ""
            self.global_diff_var.set(f"Global Drift ({norm_label}) → ‖ΔW‖: {gd_fro:+.2f}%  |  SpecMax: {gd_spec:+.2f}%  |  s_Rank: {gd_srank:+.2f}%{dora_part}")

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
                    for item in self.tree.get_children():
                        writer.writerow(self.tree.item(item)['values'])
                messagebox.showinfo("Success", "Data exported successfully!")
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
        chart_win.title(f"LoRA Spectral Norm & Rank Analytics [{norm_desc}]")
        chart_win.geometry("1200x750")
        chart_win.configure(bg=self.bg_color)
        
        notebook = ttk.Notebook(chart_win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        plt.style.use('dark_background')
        layers = [r["layer"].split('.')[-1][:15] for r in self.results_data]
        x = np.arange(len(layers))
        warn_thresh = self.spectral_warn_var.get()

        if self.mode_var.get() == "single":
            # Tab 1: Spectral Norm Max & Mean
            f_spec = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax_spec = f_spec.add_subplot(111)
            ax_spec.plot(x, [r["spec_max"] for r in self.results_data], label=f'Spectral Norm Max ({norm_desc})', marker='o', color='#ff7043', linewidth=1.5)
            ax_spec.plot(x, [r["spec_mean"] for r in self.results_data], label=f'Spectral Norm Mean ({norm_desc})', linestyle='--', color='#42a5f5')
            ax_spec.axhline(warn_thresh, color='#ffd54f', linestyle='--', label=f'Warning Threshold ({warn_thresh})', linewidth=1.5)
            ax_spec.set_title(f"LoRA Spectral Norm (Operator Norm) per Layer [{norm_desc}]")
            ax_spec.set_xticks(x); ax_spec.set_xticklabels(layers, rotation=90, fontsize=8)
            ax_spec.grid(True, alpha=0.2)
            ax_spec.legend()
            self.add_tab(notebook, f_spec, f"Spectral Norm ({norm_desc})")

            # Tab 2: Stable Rank (Rank Utilization)
            f_rank = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax_rank = f_rank.add_subplot(111)
            ax_rank.plot(x, [r["stable_rank"] for r in self.results_data], label='Stable Rank (‖ΔW‖_F² / σ_max²)', marker='s', color='#66bb6a')
            ax_rank.plot(x, [r["rank"] for r in self.results_data], label='Configured Rank (r)', linestyle=':', color='#9e9e9e')
            ax_rank.axhline(1.0, color='#f44336', linestyle='--', label='Complete Rank Collapse (1.0)', linewidth=1.2)
            ax_rank.set_title("LoRA Effective Rank Dimension Utilization (Stable Rank)")
            ax_rank.set_xticks(x); ax_rank.set_xticklabels(layers, rotation=90, fontsize=8)
            ax_rank.grid(True, alpha=0.2)
            ax_rank.legend()
            self.add_tab(notebook, f_rank, "Stable Rank")

            # Tab 3: Frobenius Norm
            f1 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax1 = f1.add_subplot(111)
            ax1.bar(x, [r["fro_norm"] for r in self.results_data], 0.5, label=f'Frobenius Norm [{norm_desc}]', color='#64b5f6')
            ax1.set_title(f"LoRA Frobenius Weight Magnitude per Layer [{norm_desc}]")
            ax1.set_xticks(x); ax1.set_xticklabels(layers, rotation=90, fontsize=8)
            ax1.legend()
            self.add_tab(notebook, f1, f"Frobenius Norm ({norm_desc})")

            # Tab 4: DoRA Multiplier
            if any(r.get("has_dora") for r in self.results_data):
                f_dora = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
                ax_dora = f_dora.add_subplot(111)
                ax_dora.plot(x, [r["dora_mean"] for r in self.results_data], label='DoRA Mean Magnitude', marker='o', color='#ab47bc', linewidth=1.5)
                ax_dora.plot(x, [r["dora_max"] for r in self.results_data], label='DoRA Max', linestyle='--', color='#ff7043', linewidth=1.2)
                ax_dora.plot(x, [r["dora_min"] for r in self.results_data], label='DoRA Min', linestyle=':', color='#29b6f6', linewidth=1.2)
                ax_dora.fill_between(x, [r["dora_min"] for r in self.results_data], [r["dora_max"] for r in self.results_data], color='#ab47bc', alpha=0.18, label='DoRA Min-Max Range')
                ax_dora.axhline(1.0, color='white', linestyle='--', linewidth=1, label='Baseline (1.0)')
                ax_dora.set_title("DoRA Directional Multiplier per Layer")
                ax_dora.set_xticks(x); ax_dora.set_xticklabels(layers, rotation=90, fontsize=8)
                ax_dora.grid(True, alpha=0.2)
                ax_dora.legend()
                self.add_tab(notebook, f_dora, "DoRA Multiplier")

        else: # Compare Mode
            has_dora = any(r.get("has_dora") for r in self.results_data)

            # Tab 1: Spectral Norm Compare
            f_spec_c = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax_spec_c = f_spec_c.add_subplot(111)
            ax_spec_c.plot(x, [r["spec_1"] for r in self.results_data], label=f'LoRA 1 - σ_Max ({norm_desc})', alpha=0.85, color='#64b5f6', marker='o')
            ax_spec_c.plot(x, [r["spec_2"] for r in self.results_data], label=f'LoRA 2 - σ_Max ({norm_desc})', alpha=0.85, color='#ff7043', marker='s')
            ax_spec_c.axhline(warn_thresh, color='#ffd54f', linestyle='--', label=f'Warning Threshold ({warn_thresh})')
            ax_spec_c.fill_between(x, [r["spec_1"] for r in self.results_data], [r["spec_2"] for r in self.results_data], color='gray', alpha=0.2)
            ax_spec_c.set_title(f"Spectral Norm Comparison [{norm_desc}]")
            ax_spec_c.set_xticks(x); ax_spec_c.set_xticklabels(layers, rotation=90, fontsize=8)
            ax_spec_c.legend()
            self.add_tab(notebook, f_spec_c, f"Spectral Norm ({norm_desc})")

            # Tab 2: Stable Rank Compare
            f2 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax2 = f2.add_subplot(111)
            ax2.plot(x, [r["srank_1"] for r in self.results_data], label='LoRA 1 - Stable Rank', alpha=0.85, color='#f44336', marker='o')
            ax2.plot(x, [r["srank_2"] for r in self.results_data], label='LoRA 2 - Stable Rank', alpha=0.85, color='#4caf50', marker='s')
            ax2.fill_between(x, [r["srank_1"] for r in self.results_data], [r["srank_2"] for r in self.results_data], color='gray', alpha=0.2)
            ax2.set_title("Stable Rank Dimension Utilization Comparison")
            ax2.set_xticks(x); ax2.set_xticklabels(layers, rotation=90, fontsize=8)
            ax2.legend()
            self.add_tab(notebook, f2, "Stable Rank Compare")

            # Tab 3: Frobenius Compare
            f1 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax1 = f1.add_subplot(111)
            ax1.plot(x, [r["fro_1"] for r in self.results_data], label=f'LoRA 1 - ‖ΔW‖_F [{norm_desc}]', alpha=0.85, color='#64b5f6')
            ax1.plot(x, [r["fro_2"] for r in self.results_data], label=f'LoRA 2 - ‖ΔW‖_F [{norm_desc}]', alpha=0.85, color='#ffb74d')
            ax1.fill_between(x, [r["fro_1"] for r in self.results_data], [r["fro_2"] for r in self.results_data], color='gray', alpha=0.2)
            ax1.set_title(f"Frobenius Norm Comparison [{norm_desc}]")
            ax1.set_xticks(x); ax1.set_xticklabels(layers, rotation=90, fontsize=8)
            ax1.legend()
            self.add_tab(notebook, f1, f"Frobenius Norm ({norm_desc})")

            # Tab 4: DoRA Compare
            if has_dora:
                f_dora = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
                ax_dora = f_dora.add_subplot(111)
                ax_dora.plot(x, [r["dora_1"] for r in self.results_data], label='LoRA 1 - DoRA Mean', alpha=0.9, color='#ab47bc', marker='o', linewidth=1.5)
                ax_dora.fill_between(x, [r["dora_min_1"] for r in self.results_data], [r["dora_max_1"] for r in self.results_data], color='#ab47bc', alpha=0.15, label='LoRA 1 Range (Min-Max)')
                ax_dora.plot(x, [r["dora_2"] for r in self.results_data], label='LoRA 2 - DoRA Mean', alpha=0.9, color='#ff7043', marker='s', linewidth=1.5)
                ax_dora.fill_between(x, [r["dora_min_2"] for r in self.results_data], [r["dora_max_2"] for r in self.results_data], color='#ff7043', alpha=0.15, label='LoRA 2 Range (Min-Max)')
                ax_dora.axhline(1.0, color='white', linestyle='--', linewidth=1)
                ax_dora.set_title("DoRA Multiplier Comparison (Mean & Min-Max Range)")
                ax_dora.set_xticks(x); ax_dora.set_xticklabels(layers, rotation=90, fontsize=8)
                ax_dora.grid(True, alpha=0.2)
                ax_dora.legend()
                self.add_tab(notebook, f_dora, "DoRA Compare")

            # Tab 5: Drift Scatter Plot (%)
            f3 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax3 = f3.add_subplot(111)
            ax3.scatter(x, [r["diff_fro"] for r in self.results_data], label=f'Δ ‖ΔW‖_F % [{norm_desc}]', color='#ba68c8')
            ax3.scatter(x, [r["diff_spec"] for r in self.results_data], label=f'Δ σ_Max % [{norm_desc}]', color='#ff7043', marker='x')
            ax3.scatter(x, [r["diff_srank"] for r in self.results_data], label='Δ Stable Rank %', color='#00bcd4')
            if has_dora:
                ax3.scatter(x, [r["diff_dora"] for r in self.results_data], label='Δ DoRA Mult %', color='#ffca28', marker='^')
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