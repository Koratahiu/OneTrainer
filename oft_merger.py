"""
OFT Model Merger Pro (Up to 4 OFTs) + Lie Algebra Analytics
Supports merging up to 4 OFT safetensors models with Lie-space algorithms,
spectral norm clamping, DoRA harmonization, and interactive visual analytics.
"""

import os
import math
import csv
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class OFTMathEngine:
    """Mathematical utilities for OFT Lie algebra, Cayley transforms, and spectral norms."""

    @staticmethod
    def get_block_size(n_elements: int) -> int:
        """
        Validates if n_elements is a valid triangular number (b * (b - 1) / 2)
        and returns block_size b. Returns -1 if invalid.
        """
        if n_elements <= 0:
            return -1
        disc = 1 + 8 * n_elements
        s = int(math.isqrt(disc))
        if s * s != disc or (1 + s) % 2 != 0:
            return -1
        b = (1 + s) // 2
        return b if (b * (b - 1)) // 2 == n_elements else -1

    @staticmethod
    def vec_to_skew(vec: torch.Tensor, block_size: int) -> torch.Tensor:
        """Constructs batch skew-symmetric matrices from upper-triangular vectors."""
        r = vec.shape[0]
        rows, cols = torch.triu_indices(block_size, block_size, 1, device=vec.device)
        matrix = torch.zeros(r, block_size, block_size, dtype=vec.dtype, device=vec.device)
        batch_idx = torch.arange(r, device=vec.device)[:, None]
        matrix = matrix.index_put((batch_idx, rows, cols), vec)
        return matrix - matrix.transpose(-2, -1)

    @staticmethod
    def skew_to_vec(matrix: torch.Tensor, block_size: int) -> torch.Tensor:
        """Extracts upper-triangular vectors from batch skew-symmetric matrices."""
        rows, cols = torch.triu_indices(block_size, block_size, 1, device=matrix.device)
        return matrix[:, rows, cols]

    @staticmethod
    def compute_rotation(Q_skew: torch.Tensor, method: str = "exact") -> torch.Tensor:
        """Computes orthogonal rotation matrix R from skew-symmetric Q."""
        r, b, _ = Q_skew.shape
        I = torch.eye(b, dtype=Q_skew.dtype, device=Q_skew.device).unsqueeze(0).expand(r, b, b)

        if method == "exact":
            # Exact Cayley transform: (I - Q)(I + Q)^-1
            return torch.linalg.solve(I + Q_skew, I - Q_skew, left=False)
        elif method == "matrix_exp":
            Q_sq = torch.bmm(Q_skew, Q_skew)
            c = 4.0 - 2.0 * math.sqrt(2.0)
            d = 6.0 - 4.0 * math.sqrt(2.0)
            inner = I * 2.0 + Q_skew * c + Q_sq * d
            return I + Q_skew * 2.0 + torch.bmm(Q_sq, inner)
        else:  # neumann
            Q_sq = torch.bmm(Q_skew, Q_skew)
            inner = I * 2.0 + Q_skew * 2.0 + Q_sq
            return I + Q_skew * 2.0 + torch.bmm(Q_sq, inner)

    @staticmethod
    def invert_cayley_to_skew(R: torch.Tensor) -> torch.Tensor:
        """
        Inverts orthogonal rotation matrix R back to skew-symmetric Q via Cayley inversion:
        Q = (I - R)(I + R)^-1
        """
        r, b, _ = R.shape
        I = torch.eye(b, dtype=R.dtype, device=R.device).unsqueeze(0).expand(r, b, b)
        denom = I + R + (1e-6 * I)
        Q = torch.linalg.solve(denom, I - R)
        return 0.5 * (Q - Q.transpose(-2, -1))

    @staticmethod
    def compute_spectral_norm(Q_skew: torch.Tensor) -> torch.Tensor:
        """Computes max singular value per block."""
        try:
            return torch.linalg.svdvals(Q_skew.float())[:, 0]
        except Exception:
            return torch.zeros(Q_skew.shape[0], device=Q_skew.device)

    @staticmethod
    def clamp_spectral_norm(vec: torch.Tensor, block_size: int, threshold: float = 0.95) -> torch.Tensor:
        """Clamps the spectral norm of skew-symmetric Q to threshold, modifying vector directly."""
        if threshold <= 0:
            return vec
        Q = OFTMathEngine.vec_to_skew(vec.float(), block_size)
        svs = OFTMathEngine.compute_spectral_norm(Q)
        clamp_scale = torch.clamp(threshold / (svs + 1e-12), max=1.0).unsqueeze(-1)
        return (vec.float() * clamp_scale).to(vec.dtype)

    @staticmethod
    def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
        """Spherical linear interpolation between two tensors with magnitude blending."""
        v0_f = v0.flatten()
        v1_f = v1.flatten()
        norm_v0 = torch.linalg.norm(v0_f)
        norm_v1 = torch.linalg.norm(v1_f)
        if norm_v0 < 1e-9 or norm_v1 < 1e-9:
            return (1.0 - t) * v0 + t * v1

        u0 = v0_f / norm_v0
        u1 = v1_f / norm_v1
        dot = torch.dot(u0, u1)

        # Ensure shortest spherical path
        if dot < 0.0:
            u1 = -u1
            dot = -dot

        dot = torch.clamp(dot, -1.0, 1.0)
        if dot > 0.9995:
            res_dir = (1.0 - t) * u0 + t * u1
        else:
            omega = torch.acos(dot)
            sin_omega = torch.sin(omega)
            s0 = torch.sin((1.0 - t) * omega) / sin_omega
            s1 = torch.sin(t * omega) / sin_omega
            res_dir = s0 * u0 + s1 * u1

        interp_norm = (1.0 - t) * norm_v0 + t * norm_v1
        res = interp_norm * (res_dir / torch.linalg.norm(res_dir).clamp_min(1e-9))
        return res.view_as(v0)


class OFTMergerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OFT Model Merger Pro (Up to 4 OFTs) + Lie Algebra Analytics")
        self.root.geometry("1620x960")

        # Color Theme
        self.bg_color = "#242424"
        self.panel_bg = "#2d2d2d"
        self.entry_bg = "#383838"
        self.fg_color = "#e0e0e0"
        self.root.configure(bg=self.bg_color)

        # Merge Options
        self.merge_method_var = tk.StringVar(value="Linear Lie Blend")
        self.sparsity_density_var = tk.DoubleVar(value=0.70)
        self.spectral_clamp_var = tk.BooleanVar(value=True)
        self.clip_threshold_var = tk.DoubleVar(value=0.95)
        self.target_scaled_var = tk.StringVar(value="Auto (Detect from Model 1)")
        self.output_dtype_var = tk.StringVar(value="Float16")
        self.global_status_var = tk.StringVar(value="Ready. Load OFT models to merge.")

        # Model Slots Config (Up to 4)
        self.model_slots = []
        default_enables = [True, True, False, False]
        default_weights = [0.5, 0.5, 0.5, 0.5]
        for i in range(4):
            self.model_slots.append({
                "enabled": tk.BooleanVar(value=default_enables[i]),
                "weight": tk.DoubleVar(value=default_weights[i]),
                "path": tk.StringVar(),
                "info": tk.StringVar(value=f"Model {i+1}: No file loaded."),
                "cache": None,
                "metadata": {}
            })

        self.merged_state_dict = {}
        self.merged_metadata = {}
        self.analysis_rows = []
        self.current_headers = []

        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background=self.entry_bg, foreground=self.fg_color, fieldbackground=self.entry_bg, borderwidth=0)
        style.configure("Treeview.Heading", background="#4a4a4a", foreground=self.fg_color, borderwidth=1)
        style.map("Treeview", background=[('selected', '#1a73e8')])
        style.configure("TNotebook", background=self.bg_color, borderwidth=0)
        style.configure("TNotebook.Tab", background="#4a4a4a", foreground=self.fg_color, padding=[12, 5])
        style.map("TNotebook.Tab", background=[('selected', '#1a73e8')])

        # --- Top Section: 4 Model Input Cards ---
        slots_container = tk.Frame(self.root, bg=self.bg_color, padx=12, pady=6)
        slots_container.pack(fill=tk.X)

        slot_colors = ["#42a5f5", "#ff7043", "#66bb6a", "#ab47bc"]
        for i in range(4):
            slot = self.model_slots[i]
            card = tk.LabelFrame(
                slots_container,
                text=f" OFT Slot {i+1} ",
                bg=self.panel_bg,
                fg=slot_colors[i],
                font=("Arial", 9, "bold"),
                padx=8,
                pady=4
            )
            card.pack(fill=tk.X, pady=3)

            cb = tk.Checkbutton(card, text="Active", variable=slot["enabled"], bg=self.panel_bg, fg=self.fg_color, selectcolor=self.entry_bg)
            cb.pack(side=tk.LEFT, padx=5)

            tk.Label(card, text="Weight (α):", bg=self.panel_bg, fg=self.fg_color).pack(side=tk.LEFT, padx=(5, 2))
            w_spin = tk.Spinbox(card, from_=-5.0, to=5.0, increment=0.05, textvariable=slot["weight"], width=5, bg=self.entry_bg, fg=self.fg_color, insertbackground="white")
            w_spin.pack(side=tk.LEFT, padx=3)

            w_scale = tk.Scale(card, variable=slot["weight"], from_=-2.0, to=2.0, resolution=0.05, orient=tk.HORIZONTAL, length=120, bg=self.panel_bg, fg=self.fg_color, highlightthickness=0, showvalue=False)
            w_scale.pack(side=tk.LEFT, padx=5)

            tk.Entry(card, textvariable=slot["path"], width=68, bg=self.entry_bg, fg=self.fg_color, insertbackground="white").pack(side=tk.LEFT, padx=6)
            tk.Button(card, text="Browse", command=lambda idx=i: self.browse_model(idx), bg="#555555", fg="white", font=("Arial", 8)).pack(side=tk.LEFT, padx=2)

            tk.Label(card, textvariable=slot["info"], font=("Consolas", 8), fg="#90caf9", bg=self.panel_bg).pack(side=tk.LEFT, padx=10)

        # --- Middle Section: Merge Strategy & Settings Toolbar ---
        settings_bar = tk.Frame(self.root, bg=self.panel_bg, padx=12, pady=6)
        settings_bar.pack(fill=tk.X, padx=12, pady=4)

        tk.Label(settings_bar, text="Merge Method:", bg=self.panel_bg, fg="#ffcc80", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        methods = [
            "Linear Lie Blend",
            "Normalized Lie Blend",
            "Rotation Composition (Matrix Prod)",
            "TIES (Trim & Elect Sign)",
            "DARE (Drop & Rescale)",
            "SLERP (Spherical Linear)"
        ]
        method_combo = ttk.Combobox(settings_bar, textvariable=self.merge_method_var, values=methods, state="readonly", width=26)
        method_combo.pack(side=tk.LEFT, padx=4)

        tk.Label(settings_bar, text="Density (TIES/DARE):", bg=self.panel_bg, fg=self.fg_color).pack(side=tk.LEFT, padx=(12, 2))
        tk.Entry(settings_bar, textvariable=self.sparsity_density_var, width=5, bg=self.entry_bg, fg=self.fg_color, insertbackground="white").pack(side=tk.LEFT, padx=3)

        tk.Checkbutton(settings_bar, text="Enforce Spectral Clamp", variable=self.spectral_clamp_var, bg=self.panel_bg, fg="#80cbc4", selectcolor=self.entry_bg).pack(side=tk.LEFT, padx=(12, 2))
        tk.Label(settings_bar, text="Thresh (σ):", bg=self.panel_bg, fg=self.fg_color).pack(side=tk.LEFT, padx=(2, 2))
        tk.Entry(settings_bar, textvariable=self.clip_threshold_var, width=5, bg=self.entry_bg, fg=self.fg_color, insertbackground="white").pack(side=tk.LEFT, padx=3)

        tk.Label(settings_bar, text="Target Scaling:", bg=self.panel_bg, fg=self.fg_color).pack(side=tk.LEFT, padx=(12, 2))
        scale_combo = ttk.Combobox(settings_bar, textvariable=self.target_scaled_var, values=["Auto (Detect from Model 1)", "Force Scaled OFT", "Force Unscaled OFT"], state="readonly", width=22)
        scale_combo.pack(side=tk.LEFT, padx=4)

        tk.Label(settings_bar, text="Dtype:", bg=self.panel_bg, fg=self.fg_color).pack(side=tk.LEFT, padx=(12, 2))
        dtype_combo = ttk.Combobox(settings_bar, textvariable=self.output_dtype_var, values=["Float16", "BFloat16", "Float32"], state="readonly", width=10)
        dtype_combo.pack(side=tk.LEFT, padx=4)

        # --- Action Buttons & Status ---
        btn_bar = tk.Frame(self.root, bg=self.bg_color, padx=12, pady=5)
        btn_bar.pack(fill=tk.X)

        tk.Button(btn_bar, text="1. Run Merge & Analyze", command=self.perform_merge, bg="#2e7d32", fg="white", font=("Arial", 9, "bold"), width=22).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_bar, text="2. Show Analytics Charts", command=self.show_charts, bg="#1565c0", fg="white", font=("Arial", 9, "bold"), width=22).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_bar, text="3. Save Merged (.safetensors)", command=self.save_merged_file, bg="#ef6c00", fg="white", font=("Arial", 9, "bold"), width=25).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_bar, text="4. Export CSV Report", command=self.export_csv, bg="#6a1b9a", fg="white", font=("Arial", 9, "bold"), width=20).pack(side=tk.LEFT, padx=4)

        tk.Label(btn_bar, textvariable=self.global_status_var, bg=self.bg_color, fg="#ffb74d", font=("Arial", 9, "bold")).pack(side=tk.RIGHT, padx=12)

        # --- Table View Frame ---
        table_frame = tk.Frame(self.root, padx=12, pady=4, bg=self.bg_color)
        table_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(table_frame, show="headings")
        y_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscroll=y_scroll.set, xscroll=x_scroll.set)

        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.tree.tag_configure("safe", foreground="#e0e0e0")
        self.tree.tag_configure("clipped", foreground="#ffb74d")
        self.tree.tag_configure("divergent", foreground="#ff5252")

    def browse_model(self, idx: int):
        path = filedialog.askopenfilename(filetypes=[("OFT Weights", "*.safetensors *.bin *.pt")])
        if not path:
            return
        slot = self.model_slots[idx]
        slot["path"].set(path)
        
        try:
            metadata = {}
            if path.endswith(".safetensors"):
                try:
                    with safe_open(path, framework="pt") as f:
                        metadata = f.metadata() or {}
                except Exception:
                    pass

            sd = load_file(path) if path.endswith(".safetensors") else torch.load(path, map_location="cpu")
            slot["cache"] = sd
            slot["metadata"] = metadata

            has_scaled = any('scaled_oft' in k for k in sd.keys()) or metadata.get("oft_scaled", "").lower() == "true"
            has_cans = any('cans_oft' in k for k in sd.keys()) or metadata.get("oft_cans", "").lower() == "true"
            has_matexp = any('matrix_exp_oft' in k for k in sd.keys()) or metadata.get("oft_matrix_exp", "").lower() == "true"
            has_dora = any('dora_log_multiplier' in k for k in sd.keys())

            clip_val = None
            for k, v in sd.items():
                if 'clipped_oft' in k:
                    clip_val = float(v.item() if hasattr(v, 'item') else v)
                    break
            if clip_val is None and "oft_clipped_norm" in metadata:
                try: clip_val = float(metadata["oft_clipped_norm"])
                except Exception: pass

            oft_layers = sum(
                1 for k, v in sd.items()
                if ('oft' in k.lower() or 'weight' in k)
                and 'dora_log_multiplier' not in k
                and len(v.shape) == 2
                and OFTMathEngine.get_block_size(v.shape[1]) > 0
            )
            clip_str = f"Clip: {clip_val:.2f}" if clip_val is not None else "Clip: None"
            slot["info"].set(f"Layers: {oft_layers} | Scaled: {has_scaled} | CANS: {has_cans} | MatExp: {has_matexp} | {clip_str} | DoRA: {has_dora}")
        except Exception as e:
            slot["info"].set(f"Error reading file: {e}")

    def load_slot_data(self, idx: int):
        slot = self.model_slots[idx]
        path = slot["path"].get()
        if not path or not os.path.exists(path):
            return None, {}
        if slot["cache"] is not None:
            return slot["cache"], slot["metadata"]
        try:
            metadata = {}
            if path.endswith(".safetensors"):
                with safe_open(path, framework="pt") as f:
                    metadata = f.metadata() or {}
            sd = load_file(path) if path.endswith(".safetensors") else torch.load(path, map_location="cpu")
            slot["cache"] = sd
            slot["metadata"] = metadata
            return sd, metadata
        except Exception as e:
            messagebox.showerror("Loading Error", f"Failed to load Model {idx+1}:\n{e}")
            return None, {}

    def extract_oft_blocks(self, sd: dict, metadata: dict):
        """Extracts parsed OFT modules, validating block geometry and detecting configurations."""
        layers = {}
        dora_mults = {}
        dora_keys = {}
        layer_scaled = {}
        layer_cans = {}
        layer_matrix_exp = {}

        for k, v in sd.items():
            if 'dora_log_multiplier' in k:
                base = k.replace('.dora_log_multiplier', '').replace('.oft_R', '')
                dora_mults[base] = v.float()
                dora_keys[base] = k
            elif 'scaled_oft' in k:
                base = k.replace('.oft_R.scaled_oft', '').replace('.scaled_oft', '')
                layer_scaled[base] = bool(v.item() if hasattr(v, 'item') else v)
            elif 'cans_oft' in k:
                base = k.replace('.oft_R.cans_oft', '').replace('.cans_oft', '')
                layer_cans[base] = bool(v.item() if hasattr(v, 'item') else v)
            elif 'matrix_exp_oft' in k:
                base = k.replace('.oft_R.matrix_exp_oft', '').replace('.matrix_exp_oft', '')
                layer_matrix_exp[base] = bool(v.item() if hasattr(v, 'item') else v)

        global_scaled = len(layer_scaled) > 0 or metadata.get("oft_scaled", "").lower() == "true"
        global_cans = len(layer_cans) > 0 or metadata.get("oft_cans", "").lower() == "true"
        global_matexp = len(layer_matrix_exp) > 0 or metadata.get("oft_matrix_exp", "").lower() == "true"

        for k, v in sd.items():
            if 'oft' not in k.lower() and 'weight' not in k: continue
            if 'dora_log_multiplier' in k: continue
            if any(buf in k for buf in ['scaled_oft', 'cans_oft', 'matrix_exp_oft', 'clipped_oft', 'rows', 'cols', 'u_state', 'v_state']):
                continue

            t = v.float()
            if len(t.shape) != 2:
                continue

            r_blocks, n_elements = t.shape
            block_size = OFTMathEngine.get_block_size(n_elements)
            if block_size <= 1:
                # Discard non-OFT weights that do not match upper-triangular parameter dimensions
                continue

            base = k.replace(".oft_R.weight", "").replace(".weight", "")
            is_scaled = layer_scaled.get(base, global_scaled)
            scale_factor = 2.0 * math.sqrt(block_size - 1) if (is_scaled and block_size > 1) else 1.0

            layers[base] = {
                "tensor_key": k,
                "weight_raw": t,
                "weight_eff": t / scale_factor,
                "r_blocks": r_blocks,
                "block_size": block_size,
                "scale_factor": scale_factor,
                "is_scaled": is_scaled,
                "is_cans": layer_cans.get(base, global_cans),
                "is_matrix_exp": layer_matrix_exp.get(base, global_matexp),
                "has_dora": base in dora_mults,
                "dora_mult": dora_mults.get(base, None),
                "dora_key": dora_keys.get(base, None)
            }
        return layers

    def merge_tensors_lie(self, tensors: list, weights: list, method: str, density: float) -> torch.Tensor:
        """Merges a list of effective skew vectors using the selected Lie algebra strategy."""
        k = len(tensors)
        if k == 1:
            return tensors[0] * weights[0]

        if method == "Linear Lie Blend":
            res = torch.zeros_like(tensors[0])
            for t, w in zip(tensors, weights):
                res += w * t
            return res

        elif method == "Normalized Lie Blend":
            w_sum = sum(weights)
            if abs(w_sum) < 1e-8:
                w_sum = 1.0
            res = torch.zeros_like(tensors[0])
            for t, w in zip(tensors, weights):
                res += (w / w_sum) * t
            return res

        elif method == "DARE (Drop & Rescale)":
            res = torch.zeros_like(tensors[0])
            p = max(0.01, min(1.0, density))
            for t, w in zip(tensors, weights):
                mask = torch.empty_like(t).bernoulli_(p=p)
                dare_t = (t * mask) / p
                res += w * dare_t
            return res

        elif method == "TIES (Trim & Elect Sign)":
            p = max(0.01, min(1.0, density))
            trimmed = []
            for t, w in zip(tensors, weights):
                flat = t.abs().flatten()
                k_val = max(1, int(flat.numel() * (1 - p)))
                thresh = torch.kthvalue(flat, k_val).values
                trimmed.append(torch.where(t.abs() >= thresh, t * w, torch.zeros_like(t)))

            stacked = torch.stack(trimmed, dim=0)
            dominant_sign = torch.sign(torch.sum(stacked, dim=0))
            disjoint = torch.where(torch.sign(stacked) == dominant_sign, stacked, torch.zeros_like(stacked))

            # Normalize by the cumulative active absolute weights to avoid double-division
            active_weights = torch.stack([
                torch.where(
                    (trimmed[i] != 0) & (torch.sign(trimmed[i]) == dominant_sign),
                    torch.full_like(tensors[0], abs(weights[i])),
                    torch.zeros_like(tensors[0])
                )
                for i in range(len(weights))
            ], dim=0).sum(dim=0).clamp_min(1e-6)

            return disjoint.sum(dim=0) / active_weights

        elif method == "Rotation Composition (Matrix Prod)":
            # Multiplies rotations sequentially in forward order: x * R1 * R2
            b_size = OFTMathEngine.get_block_size(tensors[0].shape[1])
            R_total = None
            for t, w in zip(tensors, weights):
                Q = OFTMathEngine.vec_to_skew(t * w, b_size)
                R = OFTMathEngine.compute_rotation(Q, method="exact")
                R_total = R if R_total is None else torch.bmm(R_total, R)
            Q_total = OFTMathEngine.invert_cayley_to_skew(R_total)
            return OFTMathEngine.skew_to_vec(Q_total, b_size)

        elif method == "SLERP (Spherical Linear)":
            w_sum = sum(weights)
            norm_weights = [w / (w_sum + 1e-9) for w in weights]
            res = tensors[0]
            cum_w = norm_weights[0]
            for i in range(1, k):
                w_i = norm_weights[i]
                cum_w += w_i
                t_ratio = w_i / max(cum_w, 1e-9)
                res = OFTMathEngine.slerp(res, tensors[i], t_ratio)
            return res

        return tensors[0]

    def perform_merge(self):
        active_slots = []
        for i, s in enumerate(self.model_slots):
            if s["enabled"].get() and s["path"].get():
                active_slots.append((i, s))

        if not active_slots:
            messagebox.showwarning("No Models", "Please enable and select at least one OFT model to merge.")
            return

        loaded_models = []
        for idx, slot in active_slots:
            sd, meta = self.load_slot_data(idx)
            if sd is None:
                return
            parsed = self.extract_oft_blocks(sd, meta)
            loaded_models.append({
                "slot_idx": idx,
                "weight": slot["weight"].get(),
                "layers": parsed,
                "metadata": meta,
                "name": os.path.basename(slot["path"].get())
            })

        all_layer_names = sorted(list(set().union(*[m["layers"].keys() for m in loaded_models])))
        if not all_layer_names:
            messagebox.showerror("Error", "No valid OFT layers could be extracted from the specified models.")
            return

        # Target Scaling Decision
        target_mode = self.target_scaled_var.get()
        if "Auto" in target_mode:
            target_scaled = loaded_models[0]["layers"][all_layer_names[0]]["is_scaled"] if all_layer_names else True
        else:
            target_scaled = "Force Scaled" in target_mode

        # Detect matrix_exp & cans usage across models
        has_matexp = any(c["layers"][k]["is_matrix_exp"] for c in loaded_models for k in c["layers"])
        has_cans = any(c["layers"][k]["is_cans"] for c in loaded_models for k in c["layers"])

        method = self.merge_method_var.get()
        density = self.sparsity_density_var.get()
        apply_clip = self.spectral_clamp_var.get()
        clip_thresh = self.clip_threshold_var.get()

        merged_sd = {}
        analysis_data = []
        total_divergent = 0
        total_clipped = 0

        for layer_name in all_layer_names:
            contributing = [m for m in loaded_models if layer_name in m["layers"]]
            b_size = contributing[0]["layers"][layer_name]["block_size"]
            r_blocks = contributing[0]["layers"][layer_name]["r_blocks"]

            # Validate identical geometry
            for c in contributing:
                if c["layers"][layer_name]["block_size"] != b_size or c["layers"][layer_name]["r_blocks"] != r_blocks:
                    messagebox.showerror("Dimension Mismatch", f"Layer {layer_name} has mismatched rank or block sizes across models.")
                    return

            eff_tensors = [c["layers"][layer_name]["weight_eff"] for c in contributing]
            weights = [c["weight"] for c in contributing]

            # 1. Lie Algebra Blend in normalized effective space
            v_merged_eff = self.merge_tensors_lie(eff_tensors, weights, method, density)

            # 2. Spectral Norm computation & Clamping
            Q_skew = OFTMathEngine.vec_to_skew(v_merged_eff, b_size)
            svs = OFTMathEngine.compute_spectral_norm(Q_skew)
            spec_max_before = svs.max().item()
            clipped_blocks = (svs > clip_thresh).sum().item()

            if apply_clip and clip_thresh > 0:
                v_merged_eff = OFTMathEngine.clamp_spectral_norm(v_merged_eff, b_size, clip_thresh)
                Q_skew = OFTMathEngine.vec_to_skew(v_merged_eff, b_size)
                svs_after = OFTMathEngine.compute_spectral_norm(Q_skew)
                spec_max = svs_after.max().item()

                if spec_max_before >= 1.0:
                    status = f"Clipped (was div σ={spec_max_before:.3f}→{spec_max:.3f})"
                    tag = "clipped"
                    total_clipped += 1
                elif clipped_blocks > 0:
                    status = f"Clipped ({clipped_blocks}/{r_blocks} > {clip_thresh:.2f})"
                    tag = "clipped"
                    total_clipped += clipped_blocks
                else:
                    status = f"Safe (σ={spec_max:.3f})"
                    tag = "safe"
            else:
                spec_max = spec_max_before
                if spec_max >= 1.0:
                    status = f"DIVERGENT (σ={spec_max:.3f}≥1.0)"
                    tag = "divergent"
                    total_divergent += 1
                elif clipped_blocks > 0:
                    status = f"Unclipped ({clipped_blocks}/{r_blocks} > {clip_thresh:.2f})"
                    tag = "clipped"
                else:
                    status = f"Safe (σ={spec_max:.3f})"
                    tag = "safe"

            # 3. Convert back to target scaled representation
            target_factor = (2.0 * math.sqrt(b_size - 1)) if (target_scaled and b_size > 1) else 1.0
            v_final = v_merged_eff * target_factor

            # 4. Rotation Shift (‖R - I‖) evaluated exactly to prevent divergence explosion
            R_mat = OFTMathEngine.compute_rotation(Q_skew, method="exact")
            I_mat = torch.eye(b_size, dtype=R_mat.dtype, device=R_mat.device).unsqueeze(0).expand_as(R_mat)
            rot_shift = (torch.linalg.norm(R_mat - I_mat).item()) / math.sqrt(r_blocks)

            # 5. DoRA Multiplier Merging: non-DoRA models contribute 0.0 in log space
            has_dora = any(c["layers"][layer_name]["has_dora"] for c in contributing)
            merged_dora_val = 1.0
            dora_tensor = None
            dora_key_save = None

            if has_dora:
                ref_item = next(c["layers"][layer_name] for c in contributing if c["layers"][layer_name]["has_dora"])
                ref_dora = ref_item["dora_mult"]
                dora_key_save = ref_item["dora_key"]

                total_w = sum(c["weight"] for c in contributing)
                denom = total_w if abs(total_w) > 1e-8 else 1.0
                dora_tensor = torch.zeros_like(ref_dora)

                for c in contributing:
                    w = c["weight"]
                    dt = c["layers"][layer_name]["dora_mult"]
                    # If this specific model has DoRA, accumulate; if not, log(1.0) = 0 is naturally added
                    if dt is not None:
                        dora_tensor += (w / denom) * dt

                merged_dora_val = torch.exp(dora_tensor).mean().item()

            # Assign into final state dict
            orig_key = contributing[0]["layers"][layer_name]["tensor_key"]
            merged_sd[orig_key] = v_final
            if target_scaled:
                merged_sd[orig_key.replace(".weight", ".scaled_oft")] = torch.tensor(True)
            if has_matexp:
                merged_sd[orig_key.replace(".weight", ".matrix_exp_oft")] = torch.tensor(True)
            if has_cans:
                merged_sd[orig_key.replace(".weight", ".cans_oft")] = torch.tensor(True)
            if apply_clip and clip_thresh is not None:
                merged_sd[orig_key.replace(".weight", ".clipped_oft")] = torch.tensor(clip_thresh)
            if dora_tensor is not None and dora_key_save is not None:
                merged_sd[dora_key_save] = dora_tensor

            # Record metrics per slot for UI and analytics
            row_dict = {
                "layer": layer_name,
                "r_blocks": r_blocks,
                "block_size": b_size,
                "target_factor": target_factor,
                "spec_max": spec_max,
                "rot_shift": rot_shift,
                "merged_l2": torch.linalg.norm(v_final).item(),
                "status": status,
                "tag": tag,
                "has_dora": has_dora,
                "dora_mult": merged_dora_val
            }
            for i in range(4):
                found = next((c for c in contributing if c["slot_idx"] == i), None)
                if found:
                    row_dict[f"m{i+1}_l2"] = torch.linalg.norm(found["layers"][layer_name]["weight_raw"]).item()
                    Q_i = OFTMathEngine.vec_to_skew(found["layers"][layer_name]["weight_eff"], b_size)
                    row_dict[f"m{i+1}_spec"] = OFTMathEngine.compute_spectral_norm(Q_i).max().item()
                else:
                    row_dict[f"m{i+1}_l2"] = 0.0
                    row_dict[f"m{i+1}_spec"] = 0.0

            analysis_data.append(row_dict)

        self.merged_state_dict = merged_sd
        self.merged_metadata = {
            "oft_scaled": "true" if target_scaled else "false",
            "oft_matrix_exp": "true" if has_matexp else "false",
            "oft_cans": "true" if has_cans else "false",
            "oft_clipped_norm": str(clip_thresh) if apply_clip else "none",
            "merge_method": method,
            "models_merged": ", ".join([f"Slot{c['slot_idx']+1}(α={c['weight']}):{c['name']}" for c in loaded_models])
        }
        self.analysis_rows = analysis_data
        self.refresh_table()

        self.global_status_var.set(
            f"Merged {len(all_layer_names)} Layers | Method: {method} | Max σ: {max([r['spec_max'] for r in analysis_data]):.3f} | Clipped: {total_clipped} | Divergent: {total_divergent}"
        )

    def refresh_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        cols = ("layer", "r", "block_size", "scale", "m1_w", "m2_w", "m3_w", "m4_w", "merged_l2", "spec_max", "rot_shift", "dora", "status")
        headers = [
            "Layer Name", "Blocks (r)", "Block Size", "Scale",
            "M1 W(L2)", "M2 W(L2)", "M3 W(L2)", "M4 W(L2)",
            "Merged W(L2)", "Merged σ_Max", "Rot Shift", "DoRA Mean", "Status"
        ]
        self.tree["columns"] = cols
        self.current_headers = headers

        for col, head in zip(cols, headers):
            self.tree.heading(col, text=head)
            if col == "layer":
                self.tree.column(col, width=280, anchor=tk.W)
            elif col == "status":
                self.tree.column(col, width=170, anchor=tk.CENTER)
            elif col in ["m1_w", "m2_w", "m3_w", "m4_w", "merged_l2", "spec_max", "rot_shift", "dora"]:
                self.tree.column(col, width=95, anchor=tk.CENTER)
            else:
                self.tree.column(col, width=80, anchor=tk.CENTER)

        for r in self.analysis_rows:
            self.tree.insert("", "end", values=(
                r["layer"], r["r_blocks"], r["block_size"], f"{r['target_factor']:.2f}x",
                f"{r['m1_l2']:.3f}" if r['m1_l2'] > 0 else "-",
                f"{r['m2_l2']:.3f}" if r['m2_l2'] > 0 else "-",
                f"{r['m3_l2']:.3f}" if r['m3_l2'] > 0 else "-",
                f"{r['m4_l2']:.3f}" if r['m4_l2'] > 0 else "-",
                f"{r['merged_l2']:.4f}", f"{r['spec_max']:.4f}", f"{r['rot_shift']:.4f}",
                f"{r['dora_mult']:.4f}" if r["has_dora"] else "None",
                r["status"]
            ), tags=(r["tag"],))

    def save_merged_file(self):
        if not self.merged_state_dict:
            messagebox.showwarning("No Data", "Please perform a merge operation first before saving.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".safetensors",
            filetypes=[("SafeTensors Model", "*.safetensors")]
        )
        if not save_path:
            return

        target_dtype_str = self.output_dtype_var.get()
        if target_dtype_str == "Float16":
            cast_dtype = torch.float16
        elif target_dtype_str == "BFloat16":
            cast_dtype = torch.bfloat16
        else:
            cast_dtype = torch.float32

        out_dict = {}
        for k, v in self.merged_state_dict.items():
            if v.dtype.is_floating_point:
                out_dict[k] = v.to(cast_dtype)
            else:
                out_dict[k] = v

        try:
            save_file(out_dict, save_path, metadata=self.merged_metadata)
            messagebox.showinfo("Success", f"Successfully exported merged OFT model to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save safetensors file:\n{e}")

    def export_csv(self):
        if not self.analysis_rows:
            messagebox.showwarning("No Data", "Please merge models first before exporting CSV.")
            return

        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not path:
            return

        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.current_headers)
                for item in self.tree.get_children():
                    writer.writerow(self.tree.item(item)['values'])
            messagebox.showinfo("Success", "Merge analysis exported successfully!")
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not write CSV:\n{e}")

    def show_charts(self):
        if not self.analysis_rows:
            messagebox.showwarning("No Data", "Please run merge first to inspect analytics.")
            return

        chart_win = tk.Toplevel(self.root)
        chart_win.title(f"OFT Merged Model Analytics [{self.merge_method_var.get()}]")
        chart_win.geometry("1200x760")
        chart_win.configure(bg=self.bg_color)

        notebook = ttk.Notebook(chart_win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        plt.style.use('dark_background')
        layers = [r["layer"].split('.')[-1][:14] for r in self.analysis_rows]
        x = np.arange(len(layers))

        # Tab 1: Spectral Norm Comparison across Models & Merged
        f_spec = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
        ax_spec = f_spec.add_subplot(111)
        colors = ["#42a5f5", "#ff7043", "#66bb6a", "#ab47bc"]
        for i in range(4):
            if any(r[f"m{i+1}_spec"] > 0 for r in self.analysis_rows):
                ax_spec.plot(x, [r[f"m{i+1}_spec"] for r in self.analysis_rows], label=f'Model {i+1} σ', color=colors[i], alpha=0.5, linestyle=":")
        ax_spec.plot(x, [r["spec_max"] for r in self.analysis_rows], label='Merged σ_Max', color='#ffd54f', linewidth=2.0, marker='o')
        ax_spec.axhline(self.clip_threshold_var.get(), color='#ff5252', linestyle='--', label=f'Clip Limit ({self.clip_threshold_var.get()})')
        ax_spec.axhline(1.0, color='#d50000', linestyle=':', label='Neumann Limit (1.0)')
        ax_spec.set_title("Lie Algebra Spectral Norm (σ_Max) per Layer")
        ax_spec.set_xticks(x); ax_spec.set_xticklabels(layers, rotation=90, fontsize=8)
        ax_spec.grid(True, alpha=0.2)
        ax_spec.legend()
        self.add_tab(notebook, f_spec, "Spectral Norms")

        # Tab 2: Rotation Shift (‖R - I‖)
        f_rot = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
        ax_rot = f_rot.add_subplot(111)
        ax_rot.plot(x, [r["rot_shift"] for r in self.analysis_rows], label='Merged Rotation Shift (‖R - I‖)', color='#26a69a', marker='s', linewidth=1.8)
        ax_rot.set_title("Merged Orthogonal Rotation Shift Magnitude")
        ax_rot.set_xticks(x); ax_rot.set_xticklabels(layers, rotation=90, fontsize=8)
        ax_rot.grid(True, alpha=0.2)
        ax_rot.legend()
        self.add_tab(notebook, f_rot, "Rotation Shifts")

        # Tab 3: Weight L2 Comparison
        f_w = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
        ax_w = f_w.add_subplot(111)
        for i in range(4):
            if any(r[f"m{i+1}_l2"] > 0 for r in self.analysis_rows):
                ax_w.plot(x, [r[f"m{i+1}_l2"] for r in self.analysis_rows], label=f'Model {i+1} W(L2)', color=colors[i], alpha=0.5, linestyle='--')
        ax_w.plot(x, [r["merged_l2"] for r in self.analysis_rows], label='Merged W(L2)', color='#29b6f6', linewidth=2.0)
        ax_w.set_title("Weight L2 Norm Magnitudes")
        ax_w.set_xticks(x); ax_w.set_xticklabels(layers, rotation=90, fontsize=8)
        ax_w.grid(True, alpha=0.2)
        ax_w.legend()
        self.add_tab(notebook, f_w, "Weight L2")

        # Tab 4: DoRA Multipliers
        if any(r["has_dora"] for r in self.analysis_rows):
            f_dora = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax_dora = f_dora.add_subplot(111)
            ax_dora.plot(x, [r["dora_mult"] for r in self.analysis_rows], label='Merged DoRA Multiplier', color='#ab47bc', marker='^', linewidth=1.8)
            ax_dora.axhline(1.0, color='white', linestyle='--', label='Baseline (1.0)')
            ax_dora.set_title("Harmonized DoRA Multipliers per Layer")
            ax_dora.set_xticks(x); ax_dora.set_xticklabels(layers, rotation=90, fontsize=8)
            ax_dora.grid(True, alpha=0.2)
            ax_dora.legend()
            self.add_tab(notebook, f_dora, "DoRA Multipliers")

    def add_tab(self, notebook, figure, title):
        frame = tk.Frame(notebook, bg=self.bg_color)
        figure.tight_layout()
        canvas = FigureCanvasTkAgg(figure, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        notebook.add(frame, text=title)


if __name__ == "__main__":
    root = tk.Tk()
    app = OFTMergerApp(root)
    root.mainloop()