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

class OFTInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OFT Weight Inspector Pro + Spectral Norm & Clipping Analytics")
        self.root.geometry("1550x920")
        
        # --- Dark Mode Colors ---
        self.bg_color = "#2d2d2d"
        self.fg_color = "#e0e0e0"
        self.entry_bg = "#3d3d3d"
        self.root.configure(bg=self.bg_color)

        self.mode_var = tk.StringVar(value="compare")
        self.file1_path = tk.StringVar()
        self.file2_path = tk.StringVar()
        self.global_diff_var = tk.StringVar(value="Ready.")
        self.file1_info_var = tk.StringVar(value="OFT 1: No file loaded.")
        self.file2_info_var = tk.StringVar(value="OFT 2: No file loaded.")
        
        # Configuration & Thresholds
        self.clip_threshold_var = tk.DoubleVar(value=0.95)
        self.auto_detect_var = tk.BooleanVar(value=True)
        self.apply_clip_var = tk.BooleanVar(value=True)
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
        tk.Radiobutton(mode_frame, text="Single OFT", variable=self.mode_var, value="single", command=self.toggle_mode, bg=self.bg_color, fg=self.fg_color, selectcolor=self.entry_bg).pack(side=tk.LEFT, padx=6)
        tk.Radiobutton(mode_frame, text="Compare Two OFTs", variable=self.mode_var, value="compare", command=self.toggle_mode, bg=self.bg_color, fg=self.fg_color, selectcolor=self.entry_bg).pack(side=tk.LEFT, padx=6)

        # Options Toolbar
        opt_frame = tk.Frame(mode_frame, bg=self.bg_color)
        opt_frame.pack(side=tk.RIGHT)

        # Norm Mode Selector (Raw vs Scaled)
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

        # Spectral Clipping Controls
        tk.Label(opt_frame, text="Clip Thresh (σ):", bg=self.bg_color, fg="#ffb74d", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(8, 2))
        tk.Entry(opt_frame, textvariable=self.clip_threshold_var, width=5, bg=self.entry_bg, fg=self.fg_color, insertbackground="white").pack(side=tk.LEFT, padx=3)
        tk.Checkbutton(opt_frame, text="Auto-Detect Buffers", variable=self.auto_detect_var, bg=self.bg_color, fg=self.fg_color, selectcolor=self.entry_bg).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(opt_frame, text="Simulate Runtime Clip", variable=self.apply_clip_var, bg=self.bg_color, fg=self.fg_color, selectcolor=self.entry_bg).pack(side=tk.LEFT, padx=5)

        # File Inputs
        tk.Label(top_frame, text="OFT 1 (Baseline):", bg=self.bg_color, fg=self.fg_color).grid(row=1, column=0, sticky="e", pady=2)
        tk.Entry(top_frame, textvariable=self.file1_path, width=82, bg=self.entry_bg, fg=self.fg_color, insertbackground="white").grid(row=1, column=1, padx=5)
        tk.Button(top_frame, text="Browse", command=lambda: self.browse_file(self.file1_path, self.file1_info_var), bg="#555555", fg="white").grid(row=1, column=2)

        self.lbl_file1_info = tk.Label(top_frame, textvariable=self.file1_info_var, font=("Consolas", 8), fg="#81c784", bg=self.bg_color)
        self.lbl_file1_info.grid(row=2, column=1, sticky="w", padx=5)

        self.lbl_file2 = tk.Label(top_frame, text="OFT 2 (Target):", bg=self.bg_color, fg=self.fg_color)
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
        self.tree.tag_configure("divergent", foreground="#ff5252")  # Red
        self.tree.tag_configure("clipped", foreground="#ffb74d")    # Orange
        self.tree.tag_configure("safe", foreground="#e0e0e0")       # Normal

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
        path = filedialog.askopenfilename(filetypes=[("OFT Weights", "*.safetensors *.bin *.pt")])
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
            has_scaled = any('scaled_oft' in k for k in sd.keys()) or metadata.get("oft_scaled", "").lower() == "true"
            has_cans = any('cans_oft' in k for k in sd.keys()) or metadata.get("oft_cans", "").lower() == "true"
            has_matrix_exp = any('matrix_exp_oft' in k for k in sd.keys()) or metadata.get("oft_matrix_exp", "").lower() == "true"
            has_dora = any('dora_log_multiplier' in k for k in sd.keys())
            
            clipped_val = None
            for k, v in sd.items():
                if 'clipped_oft' in k:
                    clipped_val = float(v.item() if hasattr(v, 'item') else v)
                    break
            if clipped_val is None and "oft_clipped_norm" in metadata:
                try: clipped_val = float(metadata["oft_clipped_norm"])
                except Exception: pass

            oft_layers = sum(1 for k in sd.keys() if ('oft' in k.lower() or 'weight' in k) and 'dora_log_multiplier' not in k and len(sd[k].shape) == 2)

            info_str = f"Layers: {oft_layers} | Scaled: {has_scaled} | CANS: {has_cans} | MatExp: {has_matrix_exp} | Clip: {clipped_val if clipped_val is not None else 'None (default 0.95)'} | DoRA: {has_dora}"
            info_var.set(info_str)
        except Exception as e:
            info_var.set(f"Could not read metadata: {e}")

    @staticmethod
    def cans_newton_schulz(G: torch.Tensor, steps: int = 7, eps: float = 1e-7) -> torch.Tensor:
        X = G.float()
        g_norm = X.abs().sum(dim=-1, keepdim=True).amax(dim=-2, keepdim=True).clamp_min(eps)
        X = X / g_norm
        lower_bound = 1.0 / g_norm
        upper_bound = torch.ones_like(lower_bound)

        for _ in range(steps):
            lb, ub = lower_bound, upper_bound
            lb_ub = lb * ub
            e_sq = (lb**2 + lb_ub + ub**2) / 3.0
            K = 2.0 * (e_sq ** 1.5)
            L = lb_ub * (lb + ub)
            denom = K + L + 1e-12
            alpha = 6.0 / denom
            c1 = alpha * e_sq
            c3 = -alpha / 3.0

            A = torch.bmm(X, X.transpose(-2, -1))
            X = c1 * X + c3 * torch.bmm(A, X)

            eps_val = (K - L) / denom
            lower_bound = torch.clamp(1.0 - eps_val, min=1e-6)
            upper_bound = 1.0 + eps_val

        return X

    def compute_rotation_and_spectral_metrics(self, w_tensor, block_size, oft_scaled=False, use_matrix_exp=False, oft_cans=False, use_cayley_neumann=True, clip_threshold=0.95, apply_clip=True):
        if block_size <= 1:
            return {
                "spec_max_eff": 0.0, "spec_mean_eff": 0.0, "eff_spec_max": 0.0,
                "spec_max_raw": 0.0, "spec_mean_raw": 0.0,
                "clipped_blocks": 0, "rot_shift": 0.0, "ortho_err": 0.0, "scale_factor": 1.0
            }

        r, n_elements = w_tensor.shape
        w_float = w_tensor.float()

        # 1. Effective Weight with Scaled OFT (2 * sqrt(block_size - 1))
        scaling_factor = 2.0 * math.sqrt(block_size - 1) if (oft_scaled and block_size > 1) else 1.0
        effective_weight = w_float / scaling_factor

        # 2. Skew-Symmetric matrix Q construction
        rows, cols = torch.triu_indices(block_size, block_size, 1)
        matrix = torch.zeros(r, block_size, block_size, dtype=torch.float32)
        batch_idx = torch.arange(r)[:, None]
        matrix = matrix.index_put((batch_idx, rows, cols), effective_weight)
        Q_skew_eff = matrix - matrix.transpose(-2, -1)

        # 3. Spectral Norms (Effective & Raw)
        try:
            svs = torch.linalg.svdvals(Q_skew_eff)
            spec_norms_eff = svs[:, 0]
            spec_max_eff = spec_norms_eff.max().item()
            spec_mean_eff = spec_norms_eff.mean().item()

            # Raw spectral norm is simply unscaled by scaling_factor
            spec_max_raw = spec_max_eff * scaling_factor
            spec_mean_raw = spec_mean_eff * scaling_factor
        except Exception:
            spec_norms_eff = torch.zeros(r)
            spec_max_eff, spec_mean_eff = 0.0, 0.0
            spec_max_raw, spec_mean_raw = 0.0, 0.0

        # 4. Clipping evaluation (clip applied on effective spectral norm)
        thresh = clip_threshold if clip_threshold is not None else 0.95
        clipped_blocks = (spec_norms_eff > thresh).sum().item()
        
        # Apply runtime clipping to Q_skew if requested and blocks exceeded threshold
        if apply_clip and thresh > 0:
            clamp_scale = torch.clamp(thresh / (spec_norms_eff + 1e-12), max=1.0).unsqueeze(-1).unsqueeze(-1)
            Q_skew_runtime = Q_skew_eff * clamp_scale
            effective_spec_max = min(spec_max_eff, thresh)
        else:
            Q_skew_runtime = Q_skew_eff
            effective_spec_max = spec_max_eff

        # 5. Rotation Matrix R computation
        I = torch.eye(block_size, dtype=torch.float32).unsqueeze(0).expand(r, block_size, block_size)
        try:
            if oft_cans:
                if use_matrix_exp:
                    k = 3
                    Q_scaled = Q_skew_runtime / (2.0 ** k)
                    Q_sq = torch.bmm(Q_scaled, Q_scaled)
                    c = 4.0 - 2.0 * math.sqrt(2.0)
                    d = 6.0 - 4.0 * math.sqrt(2.0)
                    inner = I * 2.0 + Q_scaled * c + Q_sq * d
                    G = I + Q_scaled * 2.0 + torch.bmm(Q_sq, inner)
                    for _ in range(k):
                        G = torch.bmm(G, G)
                    R = self.cans_newton_schulz(G, steps=5)
                else:
                    Q_sq = torch.bmm(Q_skew_runtime, Q_skew_runtime)
                    G = I + 2.0 * Q_skew_runtime + Q_sq
                    R = self.cans_newton_schulz(G, steps=7)
            elif use_matrix_exp:
                Q_sq = torch.bmm(Q_skew_runtime, Q_skew_runtime)
                c = 4.0 - 2.0 * math.sqrt(2.0)
                d = 6.0 - 4.0 * math.sqrt(2.0)
                inner = I * 2.0 + Q_skew_runtime * c + Q_sq * d
                R = I + Q_skew_runtime * 2.0 + torch.bmm(Q_sq, inner)
            elif use_cayley_neumann:
                Q_sq = torch.bmm(Q_skew_runtime, Q_skew_runtime)
                inner = I * 2.0 + Q_skew_runtime * 2.0 + Q_sq
                R = I + Q_skew_runtime * 2.0 + torch.bmm(Q_sq, inner)
            else:
                R = torch.linalg.solve(I + Q_skew_runtime, I - Q_skew_runtime, left=False)

            diff = R - I
            rot_shift = torch.linalg.norm(diff).item() / math.sqrt(r)

            RtR = torch.bmm(R.transpose(-2, -1), R)
            ortho_err = torch.linalg.norm(RtR - I).item() / math.sqrt(r)
        except Exception:
            rot_shift = 0.0
            ortho_err = 0.0

        return {
            "spec_max_eff": spec_max_eff,
            "spec_mean_eff": spec_mean_eff,
            "eff_spec_max": effective_spec_max,
            "spec_max_raw": spec_max_raw,
            "spec_mean_raw": spec_mean_raw,
            "clipped_blocks": clipped_blocks,
            "rot_shift": rot_shift,
            "ortho_err": ortho_err,
            "scale_factor": scaling_factor
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
            layers = {}
            
            dora_mults = {}
            layer_scaled = {}
            layer_cans = {}
            layer_matrix_exp = {}
            layer_clipped = {}

            for k, v in sd.items():
                if 'dora_log_multiplier' in k:
                    base = k.replace('.dora_log_multiplier', '').replace('.oft_R', '')
                    dora_mults[base] = v.float()
                elif 'scaled_oft' in k:
                    base = k.replace('.oft_R.scaled_oft', '').replace('.scaled_oft', '')
                    layer_scaled[base] = bool(v.item() if hasattr(v, 'item') else v)
                elif 'cans_oft' in k:
                    base = k.replace('.oft_R.cans_oft', '').replace('.cans_oft', '')
                    layer_cans[base] = bool(v.item() if hasattr(v, 'item') else v)
                elif 'matrix_exp_oft' in k:
                    base = k.replace('.oft_R.matrix_exp_oft', '').replace('.matrix_exp_oft', '')
                    layer_matrix_exp[base] = bool(v.item() if hasattr(v, 'item') else v)
                elif 'clipped_oft' in k:
                    base = k.replace('.oft_R.clipped_oft', '').replace('.clipped_oft', '')
                    layer_clipped[base] = float(v.item() if hasattr(v, 'item') else v)

            global_scaled = len(layer_scaled) > 0 or metadata.get("oft_scaled", "").lower() == "true"
            global_cans = len(layer_cans) > 0 or metadata.get("oft_cans", "").lower() == "true"
            global_matrix_exp = len(layer_matrix_exp) > 0 or metadata.get("oft_matrix_exp", "").lower() == "true"
            
            global_clip = None
            if len(layer_clipped) > 0:
                global_clip = next(iter(layer_clipped.values()))
            elif "oft_clipped_norm" in metadata:
                try: global_clip = float(metadata["oft_clipped_norm"])
                except Exception: pass

            default_clip_thresh = self.clip_threshold_var.get()
            auto_detect = self.auto_detect_var.get()
            apply_clip = self.apply_clip_var.get()

            for k, v in sd.items():
                if 'oft' not in k.lower() and 'weight' not in k: continue
                if 'dora_log_multiplier' in k: continue 
                if any(buf in k for buf in ['scaled_oft', 'cans_oft', 'matrix_exp_oft', 'clipped_oft', 'rows', 'cols', 'u_state', 'v_state']):
                    continue
                
                base = k.replace(".oft_R.weight", "").replace(".weight", "")
                
                t = v.float()
                if len(t.shape) != 2: continue 
                
                r_blocks, n_elements = t.shape
                block_size = int((1 + math.sqrt(1 + 8 * n_elements)) / 2)
                
                w_l2_raw = torch.linalg.norm(t).item()

                is_scaled = layer_scaled.get(base, global_scaled) if auto_detect else False
                is_cans = layer_cans.get(base, global_cans) if auto_detect else False
                is_matrix_exp = layer_matrix_exp.get(base, global_matrix_exp) if auto_detect else False
                
                clip_threshold = layer_clipped.get(base, global_clip if global_clip is not None else default_clip_thresh)
                if not auto_detect:
                    clip_threshold = default_clip_thresh

                calc_res = self.compute_rotation_and_spectral_metrics(
                    t, block_size,
                    oft_scaled=is_scaled,
                    use_matrix_exp=is_matrix_exp,
                    oft_cans=is_cans,
                    use_cayley_neumann=(not is_cans and not is_matrix_exp),
                    clip_threshold=clip_threshold,
                    apply_clip=apply_clip
                )

                scale_factor = calc_res['scale_factor']
                w_l2_eff = w_l2_raw / scale_factor

                # DoRA Metrics: Mean, Min, Max
                has_dora = base in dora_mults
                if has_dora and dora_mults[base].numel() > 0:
                    exp_dora = torch.exp(dora_mults[base])
                    dora_mult = torch.mean(exp_dora).item()
                    dora_min = torch.min(exp_dora).item()
                    dora_max = torch.max(exp_dora).item()
                else:
                    dora_mult = 1.0
                    dora_min = 1.0
                    dora_max = 1.0

                layers[base] = {
                    'blocks': r_blocks,
                    'block_size': block_size,
                    'scale_factor': scale_factor,
                    'w_l2_raw': w_l2_raw,
                    'w_l2_eff': w_l2_eff,
                    'spec_max_eff': calc_res['spec_max_eff'],
                    'spec_mean_eff': calc_res['spec_mean_eff'],
                    'spec_max_raw': calc_res['spec_max_raw'],
                    'spec_mean_raw': calc_res['spec_mean_raw'],
                    'eff_spec_max': calc_res['eff_spec_max'],
                    'clipped_blocks': calc_res['clipped_blocks'],
                    'clip_thresh': clip_threshold,
                    'rot_shift': calc_res['rot_shift'],
                    'ortho_err': calc_res['ortho_err'],
                    'is_scaled': is_scaled,
                    'is_cans': is_cans,
                    'is_matrix_exp': is_matrix_exp,
                    'has_dora': has_dora,
                    'dora_mult': dora_mult,
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
                width = 280
                anchor = tk.W
            elif col in ["clip_status", "config", "dora_status"]:
                width = 160
                anchor = tk.CENTER
            elif col in ["dora_1", "dora_2"]:
                width = 175
                anchor = tk.CENTER
            elif col in ["dora_mult", "dora_min", "dora_max"]:
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

        if mode == "single":
            cols = (
                "layer", "blocks", "block_size", "scale_factor", "w_l2",
                "spec_max", "spec_mean", "clip_status", "rot_shift", "ortho_err",
                "dora_mult", "dora_min", "dora_max", "config"
            )
            headers = [
                "Layer Name", "Blocks (r)", "Block Size", "Scale Factor",
                f"Weight L2 ({norm_label})", f"SpecNorm Max ({norm_label})", f"SpecNorm Mean ({norm_label})",
                "Clip Status (σ)", "Rot Shift", "Ortho Err",
                "DoRA Mean", "DoRA Min", "DoRA Max", "Configuration"
            ]
            self.format_tree_columns(cols, headers)

            total_clipped_blocks = 0
            divergent_layers = 0

            for k in sorted(self.cached_r1.keys()):
                d = self.cached_r1[k]
                thresh = d['clip_thresh']
                num_clipped = d['clipped_blocks']
                r_total = d['blocks']
                
                # Divergence is mathematically determined by effective spectral norm
                if d['spec_max_eff'] >= 1.0:
                    status = f"DIVERGENT (σ_eff={d['spec_max_eff']:.3f}≥1.0)"
                    tag = "divergent"
                    divergent_layers += 1
                elif num_clipped > 0:
                    status = f"CLIPPED ({num_clipped}/{r_total} > {thresh:.2f})"
                    tag = "clipped"
                    total_clipped_blocks += num_clipped
                else:
                    status = f"Safe (Max {d['spec_max_eff']:.3f} < {thresh:.2f})"
                    tag = "safe"

                config_str = []
                if d['is_scaled']: config_str.append("Scaled")
                if d['is_cans']: config_str.append("CANS")
                if d['is_matrix_exp']: config_str.append("MatExp")
                if not d['is_cans'] and not d['is_matrix_exp']: config_str.append("Neumann")
                if d['has_dora']: config_str.append("DoRA")
                cfg_display = " | ".join(config_str) if config_str else "Standard"

                disp_w = d['w_l2_raw'] if is_raw else d['w_l2_eff']
                disp_spec_max = d['spec_max_raw'] if is_raw else d['spec_max_eff']
                disp_spec_mean = d['spec_mean_raw'] if is_raw else d['spec_mean_eff']

                row_dict = {
                    "layer": k,
                    "w_l2": disp_w,
                    "spec_max": disp_spec_max,
                    "spec_mean": disp_spec_mean,
                    "scale_factor": d['scale_factor'],
                    "rot_shift": d['rot_shift'],
                    "ortho_err": d['ortho_err'],
                    "dora_mult": d['dora_mult'],
                    "dora_min": d['dora_min'],
                    "dora_max": d['dora_max'],
                    "has_dora": d['has_dora'],
                    "status": status,
                    "is_raw_view": is_raw
                }
                self.results_data.append(row_dict)

                self.tree.insert("", "end", values=(
                    k, d['blocks'], d['block_size'], f"{d['scale_factor']:.2f}x",
                    f"{disp_w:.4f}", f"{disp_spec_max:.4f}", f"{disp_spec_mean:.4f}",
                    status, f"{d['rot_shift']:.4f}", f"{d['ortho_err']:.2e}",
                    f"{d['dora_mult']:.4f}" if d['has_dora'] else "None",
                    f"{d['dora_min']:.4f}" if d['has_dora'] else "None",
                    f"{d['dora_max']:.4f}" if d['has_dora'] else "None",
                    cfg_display
                ), tags=(tag,))
            
            self.global_diff_var.set(f"Loaded {len(self.cached_r1)} layers | View: {self.norm_view_var.get()} | Clipped: {total_clipped_blocks} | Divergent: {divergent_layers}")

        else: # Compare Mode
            r1, r2 = self.cached_r1, self.cached_r2
            cols = (
                "layer", "w_1", "w_2", "diff_w", "spec_1", "spec_2", "diff_spec",
                "rot_1", "rot_2", "diff_rot", "clip_status", "dora_1", "dora_2", "diff_dora"
            )
            headers = [
                "Layer Name",
                f"OFT1 W({norm_label})", f"OFT2 W({norm_label})", "Δ W%",
                f"OFT1 σ_Max({norm_label})", f"OFT2 σ_Max({norm_label})", "Δ σ_Max%",
                "OFT1 RotShift", "OFT2 RotShift", "Δ RotShift%",
                "Clip Status (1|2)", "OFT1 DoRA (Mean [Min,Max])", "OFT2 DoRA (Mean [Min,Max])", "Δ DoRA%"
            ]
            self.format_tree_columns(cols, headers)

            all_keys = sorted(set(list(r1.keys()) + list(r2.keys())))
            g = {key: 0.0 for key in ["w_1", "w_2", "rot_1", "rot_2", "spec_1", "spec_2", "dora_1", "dora_2"]}
            any_dora = False

            for k in all_keys:
                d1 = r1.get(k, {'w_l2_raw':0, 'w_l2_eff':0, 'rot_shift':0, 'spec_max_raw':0, 'spec_max_eff':0, 'spec_mean_eff':0, 'clipped_blocks':0, 'clip_thresh':0.95, 'has_dora':False, 'dora_mult':1.0, 'dora_min':1.0, 'dora_max':1.0})
                d2 = r2.get(k, {'w_l2_raw':0, 'w_l2_eff':0, 'rot_shift':0, 'spec_max_raw':0, 'spec_max_eff':0, 'spec_mean_eff':0, 'clipped_blocks':0, 'clip_thresh':0.95, 'has_dora':False, 'dora_mult':1.0, 'dora_min':1.0, 'dora_max':1.0})
                
                if d1['has_dora'] or d2['has_dora']:
                    any_dora = True

                w1 = d1['w_l2_raw'] if is_raw else d1['w_l2_eff']
                w2 = d2['w_l2_raw'] if is_raw else d2['w_l2_eff']
                s1 = d1['spec_max_raw'] if is_raw else d1['spec_max_eff']
                s2 = d2['spec_max_raw'] if is_raw else d2['spec_max_eff']

                diff_w = self.calc_pct_diff(w1, w2)
                diff_spec = self.calc_pct_diff(s1, s2)
                diff_rot = self.calc_pct_diff(d1['rot_shift'], d2['rot_shift'])
                diff_dora = self.calc_pct_diff(d1['dora_mult'], d2['dora_mult'])

                clip_str = f"{'Clipped' if d1['clipped_blocks']>0 else 'Safe'} | {'Clipped' if d2['clipped_blocks']>0 else 'Safe'}"
                
                dora1_disp = f"{d1['dora_mult']:.4f} [{d1['dora_min']:.3f}, {d1['dora_max']:.3f}]" if d1['has_dora'] else "None"
                dora2_disp = f"{d2['dora_mult']:.4f} [{d2['dora_min']:.3f}, {d2['dora_max']:.3f}]" if d2['has_dora'] else "None"

                tag = "safe"
                if d1['spec_max_eff'] >= 1.0 or d2['spec_max_eff'] >= 1.0:
                    tag = "divergent"
                elif d1['clipped_blocks'] > 0 or d2['clipped_blocks'] > 0:
                    tag = "clipped"

                row_dict = {
                    "layer": k,
                    "w_1": w1, "w_2": w2, "diff_w": diff_w,
                    "spec_1": s1, "spec_2": s2, "diff_spec": diff_spec,
                    "rot_1": d1['rot_shift'], "rot_2": d2['rot_shift'], "diff_rot": diff_rot,
                    "clip_status": clip_str,
                    "dora_1": d1['dora_mult'], "dora_2": d2['dora_mult'],
                    "dora_min_1": d1['dora_min'], "dora_max_1": d1['dora_max'],
                    "dora_min_2": d2['dora_min'], "dora_max_2": d2['dora_max'],
                    "has_dora": d1['has_dora'] or d2['has_dora'],
                    "diff_dora": diff_dora,
                    "is_raw_view": is_raw
                }
                self.results_data.append(row_dict)

                self.tree.insert("", "end", values=(
                    k, 
                    f"{w1:.4f}", f"{w2:.4f}", f"{diff_w:+.2f}%",
                    f"{s1:.4f}", f"{s2:.4f}", f"{diff_spec:+.2f}%",
                    f"{d1['rot_shift']:.4f}", f"{d2['rot_shift']:.4f}", f"{diff_rot:+.2f}%",
                    clip_str, dora1_disp, dora2_disp, f"{diff_dora:+.2f}%"
                ), tags=(tag,))

                g["w_1"] += w1; g["w_2"] += w2
                g["spec_1"] += s1; g["spec_2"] += s2
                g["rot_1"] += d1['rot_shift']; g["rot_2"] += d2['rot_shift']
                g["dora_1"] += d1['dora_mult']; g["dora_2"] += d2['dora_mult']

            gd_w = self.calc_pct_diff(g["w_1"], g["w_2"])
            gd_spec = self.calc_pct_diff(g["spec_1"], g["spec_2"])
            gd_r = self.calc_pct_diff(g["rot_1"], g["rot_2"])
            gd_dora = self.calc_pct_diff(g["dora_1"], g["dora_2"])
            
            dora_part = f"  |  DoRA: {gd_dora:+.2f}%" if any_dora else ""
            self.global_diff_var.set(f"Global Drift ({norm_label}) → W: {gd_w:+.2f}%  |  SpecMax: {gd_spec:+.2f}%  |  Rot: {gd_r:+.2f}%{dora_part}")

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
                messagebox.showerror("Error", "Permission Denied! Make sure the file isn't open in Excel or another program.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export CSV:\n{str(e)}")

    def show_charts(self):
        if not self.results_data: 
            messagebox.showwarning("No Data", "Please run analysis first.")
            return
        
        is_raw = "Raw" in self.norm_view_var.get()
        norm_desc = "Raw (Unscaled)" if is_raw else "Effective (Scaled)"
        
        chart_win = tk.Toplevel(self.root)
        chart_win.title(f"OFT Spectral Norm & Rotation Analytics [{norm_desc}]")
        chart_win.geometry("1180x740")
        chart_win.configure(bg=self.bg_color)
        
        notebook = ttk.Notebook(chart_win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        plt.style.use('dark_background')
        layers = [r["layer"].split('.')[-1][:15] for r in self.results_data]
        x = np.arange(len(layers))
        clip_thresh = self.clip_threshold_var.get()

        if self.mode_var.get() == "single":
            # Tab 1: Spectral Norm (Max & Mean)
            f_spec = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax_spec = f_spec.add_subplot(111)
            ax_spec.plot(x, [r["spec_max"] for r in self.results_data], label=f'Spectral Norm Max ({norm_desc})', marker='o', color='#ff7043', linewidth=1.5)
            ax_spec.plot(x, [r["spec_mean"] for r in self.results_data], label=f'Spectral Norm Mean ({norm_desc})', linestyle='--', color='#42a5f5')
            if not is_raw:
                ax_spec.axhline(clip_thresh, color='#ffd54f', linestyle='--', label=f'Clip Threshold ({clip_thresh})', linewidth=1.5)
                ax_spec.axhline(1.0, color='#f44336', linestyle=':', label='Neumann Limit (1.0)', linewidth=1.5)
            ax_spec.set_title(f"OFT Skew Matrix Spectral Norm per Layer [{norm_desc}]")
            ax_spec.set_xticks(x); ax_spec.set_xticklabels(layers, rotation=90, fontsize=8)
            ax_spec.grid(True, alpha=0.2)
            ax_spec.legend()
            self.add_tab(notebook, f_spec, f"Spectral Norm ({norm_desc})")

            # Tab 2: Rotation Shift & Orthogonality Error
            f_rot = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax_rot = f_rot.add_subplot(111)
            ax_rot.plot(x, [r["rot_shift"] for r in self.results_data], label='Rotation Shift (‖R-I‖)', marker='s', color='#66bb6a')
            ax_rot.set_title("OFT Matrix Rotation Shift per Layer")
            ax_rot.set_xticks(x); ax_rot.set_xticklabels(layers, rotation=90, fontsize=8)
            ax_rot.grid(True, alpha=0.2)
            ax_rot.legend()
            self.add_tab(notebook, f_rot, "Rotation Shift")

            # Tab 3: Weight L2 Norms
            f1 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax1 = f1.add_subplot(111)
            ax1.bar(x, [r["w_l2"] for r in self.results_data], 0.5, label=f'Weight L2 Norm [{norm_desc}]', color='#64b5f6')
            ax1.set_title(f"OFT Weight Magnitude (L2 Norm) per Layer [{norm_desc}]")
            ax1.set_xticks(x); ax1.set_xticklabels(layers, rotation=90, fontsize=8)
            ax1.legend()
            self.add_tab(notebook, f1, f"Weight L2 ({norm_desc})")

            # Tab 4: DoRA Multiplier (Mean, Min, Max)
            if any(r.get("has_dora") for r in self.results_data):
                f_dora = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
                ax_dora = f_dora.add_subplot(111)
                ax_dora.plot(x, [r["dora_mult"] for r in self.results_data], label='DoRA Mean', marker='o', color='#ab47bc', linewidth=1.5)
                ax_dora.plot(x, [r["dora_max"] for r in self.results_data], label='DoRA Max', linestyle='--', color='#ff7043', linewidth=1.2)
                ax_dora.plot(x, [r["dora_min"] for r in self.results_data], label='DoRA Min', linestyle=':', color='#29b6f6', linewidth=1.2)
                ax_dora.fill_between(x, [r["dora_min"] for r in self.results_data], [r["dora_max"] for r in self.results_data], color='#ab47bc', alpha=0.18, label='DoRA Min-Max Range')
                ax_dora.axhline(1.0, color='white', linestyle='--', linewidth=1, label='Baseline (1.0)')
                ax_dora.set_title("DoRA Multiplier Magnitude (Mean, Min, Max) per Layer")
                ax_dora.set_xticks(x); ax_dora.set_xticklabels(layers, rotation=90, fontsize=8)
                ax_dora.grid(True, alpha=0.2)
                ax_dora.legend()
                self.add_tab(notebook, f_dora, "DoRA Multiplier")

        else: # Compare Mode
            has_dora = any(r.get("has_dora") for r in self.results_data)

            # Tab 1: Spectral Norm Compare
            f_spec_c = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax_spec_c = f_spec_c.add_subplot(111)
            ax_spec_c.plot(x, [r["spec_1"] for r in self.results_data], label=f'OFT 1 - σ_Max ({norm_desc})', alpha=0.8, color='#64b5f6', marker='o')
            ax_spec_c.plot(x, [r["spec_2"] for r in self.results_data], label=f'OFT 2 - σ_Max ({norm_desc})', alpha=0.8, color='#ff7043', marker='s')
            if not is_raw:
                ax_spec_c.axhline(clip_thresh, color='#ffd54f', linestyle='--', label=f'Clip Threshold ({clip_thresh})')
                ax_spec_c.axhline(1.0, color='#f44336', linestyle=':', label='Neumann Limit (1.0)')
            ax_spec_c.fill_between(x, [r["spec_1"] for r in self.results_data], [r["spec_2"] for r in self.results_data], color='gray', alpha=0.2)
            ax_spec_c.set_title(f"Spectral Norm Comparison [{norm_desc}]")
            ax_spec_c.set_xticks(x); ax_spec_c.set_xticklabels(layers, rotation=90, fontsize=8)
            ax_spec_c.legend()
            self.add_tab(notebook, f_spec_c, f"Spectral Norm ({norm_desc})")

            # Tab 2: Rotation Shift Compare
            f2 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax2 = f2.add_subplot(111)
            ax2.plot(x, [r["rot_1"] for r in self.results_data], label='OFT 1 - RotShift', alpha=0.8, color='#f44336')
            ax2.plot(x, [r["rot_2"] for r in self.results_data], label='OFT 2 - RotShift', alpha=0.8, color='#4caf50')
            ax2.fill_between(x, [r["rot_1"] for r in self.results_data], [r["rot_2"] for r in self.results_data], color='gray', alpha=0.2)
            ax2.set_title("Rotation Magnitude Shift Comparison (‖R-I‖)")
            ax2.set_xticks(x); ax2.set_xticklabels(layers, rotation=90, fontsize=8)
            ax2.legend()
            self.add_tab(notebook, f2, "Rotation Compare")

            # Tab 3: Weight L2 Compare
            f1 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax1 = f1.add_subplot(111)
            ax1.plot(x, [r["w_1"] for r in self.results_data], label=f'OFT 1 - W(L2) [{norm_desc}]', alpha=0.8, color='#64b5f6')
            ax1.plot(x, [r["w_2"] for r in self.results_data], label=f'OFT 2 - W(L2) [{norm_desc}]', alpha=0.8, color='#ffb74d')
            ax1.fill_between(x, [r["w_1"] for r in self.results_data], [r["w_2"] for r in self.results_data], color='gray', alpha=0.2)
            ax1.set_title(f"Weight L2 Norm Comparison [{norm_desc}]")
            ax1.set_xticks(x); ax1.set_xticklabels(layers, rotation=90, fontsize=8)
            ax1.legend()
            self.add_tab(notebook, f1, f"Weight L2 ({norm_desc})")

            # Tab 4: DoRA Compare (Mean & Min-Max Range)
            if has_dora:
                f_dora = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
                ax_dora = f_dora.add_subplot(111)
                ax_dora.plot(x, [r["dora_1"] for r in self.results_data], label='OFT 1 - DoRA Mean', alpha=0.9, color='#ab47bc', marker='o', linewidth=1.5)
                ax_dora.fill_between(x, [r["dora_min_1"] for r in self.results_data], [r["dora_max_1"] for r in self.results_data], color='#ab47bc', alpha=0.15, label='OFT 1 Range (Min-Max)')
                ax_dora.plot(x, [r["dora_2"] for r in self.results_data], label='OFT 2 - DoRA Mean', alpha=0.9, color='#ff7043', marker='s', linewidth=1.5)
                ax_dora.fill_between(x, [r["dora_min_2"] for r in self.results_data], [r["dora_max_2"] for r in self.results_data], color='#ff7043', alpha=0.15, label='OFT 2 Range (Min-Max)')
                ax_dora.axhline(1.0, color='white', linestyle='--', linewidth=1)
                ax_dora.set_title("DoRA Multiplier Comparison (Mean & Min-Max Range)")
                ax_dora.set_xticks(x); ax_dora.set_xticklabels(layers, rotation=90, fontsize=8)
                ax_dora.grid(True, alpha=0.2)
                ax_dora.legend()
                self.add_tab(notebook, f_dora, "DoRA Compare")

            # Tab 5: Drift Scatter Plot (%)
            f3 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax3 = f3.add_subplot(111)
            ax3.scatter(x, [r["diff_w"] for r in self.results_data], label=f'Δ Weight(L2) % [{norm_desc}]', color='#ba68c8')
            ax3.scatter(x, [r["diff_spec"] for r in self.results_data], label=f'Δ SpecNorm Max % [{norm_desc}]', color='#ff7043', marker='x')
            ax3.scatter(x, [r["diff_rot"] for r in self.results_data], label='Δ RotShift %', color='#00bcd4')
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
    app = OFTInspectorApp(root)
    root.mainloop()