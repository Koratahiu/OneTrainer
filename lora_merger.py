"""
LoRA Model Merger Pro (Up to 4 LoRAs) + SVD Subspace Analytics
Supports merging up to 4 LoRA/DoRA safetensors models with SVD rank reduction,
TIES, DARE, SLERP, DoRA magnitude harmonization, CUDA/GPU acceleration, and interactive visual analytics.
"""

import os
import gc
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


class LoRAMathEngine:
    """Mathematical utilities for LoRA weight reconstruction, SVD factorization, and merging."""

    @staticmethod
    def reconstruct_delta(
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        alpha: float,
        is_conv: bool = False
    ) -> torch.Tensor:
        """
        Reconstructs delta W = (alpha / rank) * (Up @ Down).
        Handles both 2D Linear layers and 4D Convolutional layers.
        """
        rank = down_weight.shape[0]
        scaling = (alpha / rank) if rank > 0 else 1.0

        if is_conv and down_weight.dim() == 4:
            # LyCORIS / Kohya Conv2d:
            # up: (out_channels, rank, 1, 1) or (out_channels, rank)
            # down: (rank, in_channels, k_h, k_w)
            up_flat = up_weight.view(up_weight.size(0), -1)
            down_flat = down_weight.view(down_weight.size(0), -1)
            delta = up_flat @ down_flat
            delta = delta.view(up_weight.size(0), down_weight.size(1), down_weight.size(2), down_weight.size(3))
        else:
            up_flat = up_weight.view(up_weight.size(0), -1)
            down_flat = down_weight.view(down_weight.size(0), -1)
            delta = up_flat @ down_flat

        return delta * scaling

    @staticmethod
    def decompose_svd(
        delta_w: torch.Tensor,
        target_rank: int,
        target_alpha: float,
        is_conv: bool = False,
        original_shape: tuple = None
    ):
        """
        Factorizes delta_w back into Up (B) and Down (A) matrices of target_rank via SVD on CUDA/CPU:
        (alpha / rank) * (B @ A) = delta_w  ==>  B @ A = delta_w * (rank / alpha)
        Distributes singular values symmetrically: B = U * sqrt(S), A = sqrt(S) * V^T.
        Returns: down_w, up_w, energy_ratio, singular_values
        """
        device = delta_w.device
        dtype = delta_w.dtype

        # Flatten delta_w to 2D matrix
        out_dim = delta_w.shape[0]
        in_dim_flat = delta_w.numel() // out_dim
        matrix_2d = delta_w.reshape(out_dim, in_dim_flat).float()

        # Scale target matrix
        scaling = target_alpha / target_rank
        target_prod = matrix_2d / scaling

        # Perform SVD (Runs on GPU if tensor is on CUDA)
        U, S, Vh = torch.linalg.svd(target_prod, full_matrices=False)

        total_energy = torch.sum(S ** 2).item()
        k = min(target_rank, len(S))
        retained_energy = torch.sum(S[:k] ** 2).item()
        energy_ratio = (retained_energy / total_energy * 100.0) if total_energy > 1e-12 else 100.0

        # Truncate to target rank
        U_r = U[:, :k]
        S_r = S[:k]
        Vh_r = Vh[:k, :]

        sqrt_S = torch.sqrt(S_r)
        B_mat = U_r * sqrt_S.unsqueeze(0)        # (out_dim, k)
        A_mat = sqrt_S.unsqueeze(1) * Vh_r        # (k, in_dim_flat)

        if is_conv and original_shape is not None:
            # Reshape back to Conv2d weights
            _, c_in, kh, kw = original_shape
            up_weight = B_mat.view(out_dim, k, 1, 1).to(dtype).to(device)
            down_weight = A_mat.view(k, c_in, kh, kw).to(dtype).to(device)
        else:
            up_weight = B_mat.to(dtype).to(device)
            down_weight = A_mat.to(dtype).to(device)

        return down_weight, up_weight, energy_ratio, S_r.detach().cpu().numpy()

    @staticmethod
    def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
        """Spherical linear interpolation between two parameter tensors."""
        v0_f = v0.flatten()
        v1_f = v1.flatten()
        norm_v0 = torch.linalg.norm(v0_f)
        norm_v1 = torch.linalg.norm(v1_f)

        if norm_v0 < 1e-9 or norm_v1 < 1e-9:
            return (1.0 - t) * v0 + t * v1

        u0 = v0_f / norm_v0
        u1 = v1_f / norm_v1
        dot = torch.dot(u0, u1)

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

    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity between two tensors."""
        af = a.flatten().float()
        bf = b.flatten().float()
        denom = torch.linalg.norm(af) * torch.linalg.norm(bf)
        if denom < 1e-12:
            return 1.0 if torch.linalg.norm(af - bf) < 1e-6 else 0.0
        return torch.clamp(torch.dot(af, bf) / denom, -1.0, 1.0).item()


class LoRAMergerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LoRA Model Merger Pro (Up to 4 LoRAs) + CUDA SVD Subspace Analytics")
        self.root.geometry("1660x980")

        # Color Theme
        self.bg_color = "#242424"
        self.panel_bg = "#2d2d2d"
        self.entry_bg = "#383838"
        self.fg_color = "#e0e0e0"
        self.root.configure(bg=self.bg_color)

        # Detect Hardware Devices (CUDA / CPU)
        self.available_devices = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                self.available_devices.append(f"cuda:{i} ({name})")
        self.available_devices.append("cpu")

        # Merge Options
        self.device_var = tk.StringVar(value=self.available_devices[0])
        self.merge_method_var = tk.StringVar(value="Linear Blend (Task Arithmetic)")
        self.sparsity_density_var = tk.DoubleVar(value=0.70)
        self.target_rank_mode_var = tk.StringVar(value="Auto (Max Detected Rank)")
        self.custom_rank_var = tk.IntVar(value=16)
        self.target_alpha_mode_var = tk.StringVar(value="Auto (Match Target Rank)")
        self.custom_alpha_var = tk.DoubleVar(value=16.0)
        self.dora_harmonize_var = tk.BooleanVar(value=True)
        self.output_dtype_var = tk.StringVar(value="Float16")
        self.global_status_var = tk.StringVar(
            value=f"Ready. Acceleration: {self.device_var.get().split(' ')[0].upper()}."
        )

        # 4 Model Slots
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
        self.singular_value_data = {}
        self.current_headers = []

        self.setup_ui()

    def get_selected_torch_device(self) -> torch.device:
        dev_str = self.device_var.get()
        if "cuda" in dev_str:
            clean_name = dev_str.split(" ")[0]
            return torch.device(clean_name)
        return torch.device("cpu")

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
                text=f" LoRA Slot {i+1} ",
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

        # --- Middle Section: Merge Settings Bar ---
        settings_bar = tk.Frame(self.root, bg=self.panel_bg, padx=12, pady=6)
        settings_bar.pack(fill=tk.X, padx=12, pady=4)

        # Device Selection (CUDA vs CPU)
        tk.Label(settings_bar, text="Device:", bg=self.panel_bg, fg="#00e676", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 2))
        device_combo = ttk.Combobox(settings_bar, textvariable=self.device_var, values=self.available_devices, state="readonly", width=24)
        device_combo.pack(side=tk.LEFT, padx=(2, 10))

        tk.Label(settings_bar, text="Merge Method:", bg=self.panel_bg, fg="#ffcc80", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        methods = [
            "Linear Blend (Task Arithmetic)",
            "Normalized Blend",
            "TIES (Trim & Elect Sign)",
            "DARE (Drop & Rescale)",
            "DARE-TIES",
            "SLERP (Spherical Linear)"
        ]
        method_combo = ttk.Combobox(settings_bar, textvariable=self.merge_method_var, values=methods, state="readonly", width=23)
        method_combo.pack(side=tk.LEFT, padx=4)

        tk.Label(settings_bar, text="Density:", bg=self.panel_bg, fg=self.fg_color).pack(side=tk.LEFT, padx=(6, 2))
        tk.Entry(settings_bar, textvariable=self.sparsity_density_var, width=5, bg=self.entry_bg, fg=self.fg_color, insertbackground="white").pack(side=tk.LEFT, padx=3)

        tk.Label(settings_bar, text="Target Rank:", bg=self.panel_bg, fg="#80cbc4", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(8, 2))
        rank_modes = [
            "Auto (Max Detected Rank)",
            "Auto (Min Detected Rank)",
            "Fixed: 4", "Fixed: 8", "Fixed: 16", "Fixed: 32", "Fixed: 64", "Fixed: 128",
            "Custom Rank"
        ]
        rank_combo = ttk.Combobox(settings_bar, textvariable=self.target_rank_mode_var, values=rank_modes, state="readonly", width=20)
        rank_combo.pack(side=tk.LEFT, padx=4)

        tk.Label(settings_bar, text="Custom r:", bg=self.panel_bg, fg=self.fg_color).pack(side=tk.LEFT, padx=(4, 2))
        tk.Entry(settings_bar, textvariable=self.custom_rank_var, width=5, bg=self.entry_bg, fg=self.fg_color, insertbackground="white").pack(side=tk.LEFT, padx=3)

        tk.Label(settings_bar, text="Target α:", bg=self.panel_bg, fg=self.fg_color).pack(side=tk.LEFT, padx=(8, 2))
        alpha_modes = [
            "Auto (Match Target Rank)",
            "Auto (Model 1 Alpha)",
            "Custom Alpha"
        ]
        alpha_combo = ttk.Combobox(settings_bar, textvariable=self.target_alpha_mode_var, values=alpha_modes, state="readonly", width=19)
        alpha_combo.pack(side=tk.LEFT, padx=4)

        tk.Checkbutton(settings_bar, text="Harmonize DoRA", variable=self.dora_harmonize_var, bg=self.panel_bg, fg="#ce93d8", selectcolor=self.entry_bg).pack(side=tk.LEFT, padx=(8, 2))

        tk.Label(settings_bar, text="Dtype:", bg=self.panel_bg, fg=self.fg_color).pack(side=tk.LEFT, padx=(8, 2))
        dtype_combo = ttk.Combobox(settings_bar, textvariable=self.output_dtype_var, values=["Float16", "BFloat16", "Float32"], state="readonly", width=9)
        dtype_combo.pack(side=tk.LEFT, padx=4)

        # --- Action Buttons & Status ---
        btn_bar = tk.Frame(self.root, bg=self.bg_color, padx=12, pady=5)
        btn_bar.pack(fill=tk.X)

        tk.Button(btn_bar, text="1. Run Merge & Analyze", command=self.perform_merge, bg="#2e7d32", fg="white", font=("Arial", 9, "bold"), width=21).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_bar, text="2. Show Analytics Charts", command=self.show_charts, bg="#1565c0", fg="white", font=("Arial", 9, "bold"), width=21).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_bar, text="3. Save Merged (.safetensors)", command=self.save_merged_file, bg="#ef6c00", fg="white", font=("Arial", 9, "bold"), width=24).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_bar, text="4. Export CSV Report", command=self.export_csv, bg="#6a1b9a", fg="white", font=("Arial", 9, "bold"), width=18).pack(side=tk.LEFT, padx=3)

        self.progress_bar = ttk.Progressbar(btn_bar, orient=tk.HORIZONTAL, length=140, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=8)

        tk.Label(btn_bar, textvariable=self.global_status_var, bg=self.bg_color, fg="#ffb74d", font=("Arial", 9, "bold")).pack(side=tk.RIGHT, padx=8)

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

        self.tree.tag_configure("high_energy", foreground="#e0e0e0")
        self.tree.tag_configure("warn_energy", foreground="#ffb74d")
        self.tree.tag_configure("low_energy", foreground="#ff5252")

    def browse_model(self, idx: int):
        path = filedialog.askopenfilename(filetypes=[("LoRA Weights", "*.safetensors *.bin *.pt")])
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

            parsed = self.extract_lora_layers(sd)
            n_layers = len(parsed)
            ranks = [item["rank"] for item in parsed.values()]
            max_r = max(ranks) if ranks else 0
            min_r = min(ranks) if ranks else 0
            has_conv = any(item["is_conv"] for item in parsed.values())
            has_dora = any(item["has_dora"] for item in parsed.values())

            slot["info"].set(f"Layers: {n_layers} | Rank: {min_r}-{max_r} | Conv2d: {has_conv} | DoRA: {has_dora}")
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
            # Models are retained in CPU RAM to prevent exhausting VRAM when loading multiple multi-GB LoRAs
            sd = load_file(path) if path.endswith(".safetensors") else torch.load(path, map_location="cpu")
            slot["cache"] = sd
            slot["metadata"] = metadata
            return sd, metadata
        except Exception as e:
            messagebox.showerror("Loading Error", f"Failed to load Model {idx+1}:\n{e}")
            return None, {}

    def extract_lora_layers(self, sd: dict) -> dict:
        """
        Extracts and normalizes LoRA modules from arbitrary formats:
        - Kohya/LyCORIS: .lora_down.weight, .lora_up.weight, .alpha, .dora_scale
        - PEFT: .lora_A.weight, .lora_B.weight
        - Diffusers: .lora.down.weight, .lora.up.weight
        """
        layers = {}
        down_keys = {}
        up_keys = {}
        alphas = {}
        dora_scales = {}

        for k, v in sd.items():
            if k.endswith(".alpha"):
                base = k[:-6]
                alphas[base] = float(v.item() if hasattr(v, 'item') else v)
            elif 'dora_scale' in k:
                base = k.replace('.dora_scale', '')
                dora_scales[base] = v.float()
            elif any(k.endswith(s) for s in [".lora_down.weight", ".lora_A.weight", ".lora.down.weight"]):
                for s in [".lora_down.weight", ".lora_A.weight", ".lora.down.weight"]:
                    if k.endswith(s):
                        base = k[:-len(s)]
                        down_keys[base] = (k, s)
                        break
            elif any(k.endswith(s) for s in [".lora_up.weight", ".lora_B.weight", ".lora.up.weight"]):
                for s in [".lora_up.weight", ".lora_B.weight", ".lora.up.weight"]:
                    if k.endswith(s):
                        base = k[:-len(s)]
                        up_keys[base] = (k, s)
                        break

        common_bases = set(down_keys.keys()).intersection(set(up_keys.keys()))

        for base in common_bases:
            d_key, d_suffix = down_keys[base]
            u_key, u_suffix = up_keys[base]
            down_t = sd[d_key].float()
            up_t = sd[u_key].float()

            rank = down_t.shape[0]
            alpha_val = alphas.get(base, float(rank))
            is_conv = (down_t.dim() == 4)

            layers[base] = {
                "base_name": base,
                "down_key": d_key,
                "up_key": u_key,
                "down_suffix": d_suffix,
                "up_suffix": u_suffix,
                "down_weight": down_t,
                "up_weight": up_t,
                "rank": rank,
                "alpha": alpha_val,
                "is_conv": is_conv,
                "has_dora": (base in dora_scales),
                "dora_scale": dora_scales.get(base, None),
                "dora_key": f"{base}.dora_scale" if base in dora_scales else None
            }

        return layers

    def merge_deltas(self, deltas: list, weights: list, method: str, density: float) -> torch.Tensor:
        """Merges reconstructed delta tensors on GPU/CUDA using the selected strategy."""
        k = len(deltas)
        if k == 1:
            return deltas[0] * weights[0]

        if method == "Linear Blend (Task Arithmetic)":
            res = torch.zeros_like(deltas[0])
            for d, w in zip(deltas, weights):
                res += w * d
            return res

        elif method == "Normalized Blend":
            w_sum = sum(abs(w) for w in weights)
            if w_sum < 1e-8:
                w_sum = 1.0
            res = torch.zeros_like(deltas[0])
            for d, w in zip(deltas, weights):
                res += (w / w_sum) * d
            return res

        elif method == "DARE (Drop & Rescale)":
            res = torch.zeros_like(deltas[0])
            p = max(0.01, min(1.0, density))
            for d, w in zip(deltas, weights):
                mask = torch.empty_like(d).bernoulli_(p=p)
                res += w * ((d * mask) / p)
            return res

        elif method == "TIES (Trim & Elect Sign)":
            p = max(0.01, min(1.0, density))
            trimmed = []
            for d, w in zip(deltas, weights):
                flat = d.abs().flatten()
                k_val = max(1, int(flat.numel() * (1 - p)))
                thresh = torch.kthvalue(flat, k_val).values
                trimmed.append(torch.where(d.abs() >= thresh, d * w, torch.zeros_like(d)))

            stacked = torch.stack(trimmed, dim=0)
            dominant_sign = torch.sign(torch.sum(stacked, dim=0))
            disjoint = torch.where(torch.sign(stacked) == dominant_sign, stacked, torch.zeros_like(stacked))

            active_weights = torch.stack([
                torch.where(
                    (trimmed[i] != 0) & (torch.sign(trimmed[i]) == dominant_sign),
                    torch.full_like(deltas[0], abs(weights[i])),
                    torch.zeros_like(deltas[0])
                )
                for i in range(len(weights))
            ], dim=0).sum(dim=0).clamp_min(1e-6)

            return disjoint.sum(dim=0) / active_weights

        elif method == "DARE-TIES":
            p = max(0.01, min(1.0, density))
            dare_trimmed = []
            for d, w in zip(deltas, weights):
                mask = torch.empty_like(d).bernoulli_(p=p)
                dare_d = (d * mask) / p
                dare_trimmed.append(dare_d * w)

            stacked = torch.stack(dare_trimmed, dim=0)
            dominant_sign = torch.sign(torch.sum(stacked, dim=0))
            disjoint = torch.where(torch.sign(stacked) == dominant_sign, stacked, torch.zeros_like(stacked))

            active_count = (disjoint != 0).sum(dim=0).clamp_min(1).float()
            return disjoint.sum(dim=0) / active_count

        elif method == "SLERP (Spherical Linear)":
            w_sum = sum(weights)
            norm_weights = [w / (w_sum + 1e-9) for w in weights]
            res = deltas[0]
            cum_w = norm_weights[0]
            for i in range(1, k):
                w_i = norm_weights[i]
                cum_w += w_i
                t_ratio = w_i / max(cum_w, 1e-9)
                res = LoRAMathEngine.slerp(res, deltas[i], t_ratio)
            return res

        return deltas[0]

    def perform_merge(self):
        active_slots = []
        for i, s in enumerate(self.model_slots):
            if s["enabled"].get() and s["path"].get():
                active_slots.append((i, s))

        if not active_slots:
            messagebox.showwarning("No Models", "Please enable and select at least one LoRA model to merge.")
            return

        device = self.get_selected_torch_device()
        is_cuda = (device.type == "cuda")

        loaded_models = []
        for idx, slot in active_slots:
            sd, meta = self.load_slot_data(idx)
            if sd is None:
                return
            parsed = self.extract_lora_layers(sd)
            loaded_models.append({
                "slot_idx": idx,
                "weight": slot["weight"].get(),
                "layers": parsed,
                "metadata": meta,
                "name": os.path.basename(slot["path"].get())
            })

        all_layer_names = sorted(list(set().union(*[m["layers"].keys() for m in loaded_models])))
        if not all_layer_names:
            messagebox.showerror("Error", "No valid LoRA layers found in the selected models.")
            return

        total_layers = len(all_layer_names)
        rank_mode = self.target_rank_mode_var.get()
        alpha_mode = self.target_alpha_mode_var.get()
        method = self.merge_method_var.get()
        density = self.sparsity_density_var.get()
        harmonize_dora = self.dora_harmonize_var.get()

        merged_sd = {}
        analysis_data = []
        singular_value_records = {}
        total_lossy_ranks = 0

        self.progress_bar["value"] = 0
        self.global_status_var.set(f"Starting merge on {device}...")
        self.root.update_idletasks()

        try:
            for idx_layer, layer_name in enumerate(all_layer_names):
                contributing = [m for m in loaded_models if layer_name in m["layers"]]
                ref_layer = contributing[0]["layers"][layer_name]
                is_conv = ref_layer["is_conv"]
                orig_shape = ref_layer["down_weight"].shape

                contributing_ranks = [
                    c["layers"][layer_name]["rank"] for c in contributing
                ]
                if "Max" in rank_mode:
                    target_r = max(contributing_ranks)
                elif "Min" in rank_mode:
                    target_r = min(contributing_ranks)
                elif "Fixed" in rank_mode:
                    target_r = int(rank_mode.split(":")[-1].strip())
                else:
                    target_r = max(1, self.custom_rank_var.get())

                if "Match Target Rank" in alpha_mode:
                    target_alpha = float(target_r)
                elif "Model 1 Alpha" in alpha_mode:
                    target_alpha = float(contributing[0]["layers"][layer_name]["alpha"])
                else:
                    target_alpha = max(0.1, self.custom_alpha_var.get())

                # 1. Reconstruct Deltas once and store model norms
                deltas = []
                weights = []
                contrib_l2_norms = {}

                for c in contributing:
                    l = c["layers"][layer_name]
                    down_t = l["down_weight"].to(device, non_blocking=True)
                    up_t = l["up_weight"].to(device, non_blocking=True)
                    d = LoRAMathEngine.reconstruct_delta(
                        down_t, up_t, l["alpha"], is_conv
                    )
                    deltas.append(d)
                    weights.append(c["weight"])
                    # Record norm here without recomputing later
                    contrib_l2_norms[c["slot_idx"]] = (
                        torch.linalg.norm(d.float()).item()
                    )

                # 2. Merge Deltas
                merged_delta = self.merge_deltas(deltas, weights, method, density)

                # 3. Factorize using torch.svd_lowrank
                down_w, up_w, energy, svs = LoRAMathEngine.decompose_svd(
                    merged_delta, target_r, target_alpha, is_conv, orig_shape
                )
                singular_value_records[layer_name] = svs

                # 4. Status
                if energy >= 95.0:
                    status = f"Exact ({energy:.1f}%)"
                    tag = "high_energy"
                elif energy >= 80.0:
                    status = f"Good ({energy:.1f}%)"
                    tag = "warn_energy"
                else:
                    status = f"Lossy ({energy:.1f}%)"
                    tag = "low_energy"
                    total_lossy_ranks += 1

                # 5. DoRA Harmonization
                merged_dora = None
                has_dora = any(
                    c["layers"][layer_name]["has_dora"] for c in contributing
                )
                dora_mean_val = 1.0

                if has_dora and harmonize_dora:
                    dora_tensors = []
                    d_weights = []
                    for c in contributing:
                        dt = c["layers"][layer_name]["dora_scale"]
                        if dt is not None:
                            dora_tensors.append(dt.to(device, non_blocking=True))
                            d_weights.append(c["weight"])
                    if dora_tensors:
                        w_sum = sum(d_weights)
                        denom = w_sum if abs(w_sum) > 1e-8 else 1.0
                        merged_dora = sum(
                            (w / denom) * dt for w, dt in zip(d_weights, dora_tensors)
                        )
                        dora_mean_val = merged_dora.mean().item()

                # Metrics
                m1_delta = deltas[0]
                cos_to_m1 = LoRAMathEngine.cosine_similarity(merged_delta, m1_delta)
                merged_l2 = torch.linalg.norm(merged_delta.float()).item()

                row_dict = {
                    "layer": layer_name,
                    "is_conv": "Conv2d" if is_conv else "Linear",
                    "target_r": target_r,
                    "target_alpha": target_alpha,
                    "energy": energy,
                    "status": status,
                    "tag": tag,
                    "merged_l2": merged_l2,
                    "cos_m1": cos_to_m1,
                    "has_dora": has_dora,
                    "dora_mean": dora_mean_val,
                }
                for i in range(4):
                    found = next((c for c in contributing if c["slot_idx"] == i), None)
                    if found:
                        l_info = found["layers"][layer_name]
                        row_dict[f"m{i+1}_r"] = l_info["rank"]
                        row_dict[f"m{i+1}_alpha"] = l_info["alpha"]
                        row_dict[f"m{i+1}_l2"] = contrib_l2_norms.get(i, 0.0)
                    else:
                        row_dict[f"m{i+1}_r"] = "-"
                        row_dict[f"m{i+1}_alpha"] = "-"
                        row_dict[f"m{i+1}_l2"] = 0.0

                analysis_data.append(row_dict)

                # Offload to CPU
                base_model1 = contributing[0]["layers"][layer_name]
                down_key = base_model1["down_key"]
                up_key = base_model1["up_key"]
                alpha_key = f"{layer_name}.alpha"

                merged_sd[down_key] = down_w.cpu()
                merged_sd[up_key] = up_w.cpu()
                merged_sd[alpha_key] = torch.tensor(target_alpha, dtype=torch.float32)
                if merged_dora is not None and base_model1["dora_key"]:
                    merged_sd[base_model1["dora_key"]] = merged_dora.cpu()

                del deltas, merged_delta, down_w, up_w
                if merged_dora is not None:
                    del merged_dora

                # UI Progress Update (DO NOT call empty_cache() here)
                progress_pct = ((idx_layer + 1) / total_layers) * 100
                self.progress_bar["value"] = progress_pct

                if idx_layer % 15 == 0 or idx_layer == total_layers - 1:
                    self.global_status_var.set(
                        f"Merging layer {idx_layer+1}/{total_layers} ({progress_pct:.1f}%) on {device.type.upper()}..."
                    )
                    self.root.update_idletasks()

        except torch.cuda.OutOfMemoryError as oom_err:
            if is_cuda:
                torch.cuda.empty_cache()
            messagebox.showerror(
                "CUDA Out of Memory",
                f"GPU VRAM exhausted at layer: {layer_name}\n\n"
                f"Suggestions:\n"
                f"1. Choose a lower fixed Target Rank (e.g., Fixed: 16 or 32).\n"
                f"2. Switch 'Device' dropdown to 'cpu' to bypass VRAM limitations."
            )
            self.global_status_var.set("Merge aborted: CUDA Out Of Memory.")
            return
        except Exception as e:
            messagebox.showerror("Merge Error", f"An error occurred during merging:\n{e}")
            self.global_status_var.set("Merge failed.")
            return
        finally:
            if is_cuda:
                torch.cuda.empty_cache()
            gc.collect()

        self.merged_state_dict = merged_sd
        self.merged_metadata = {
            "lora_target_rank_mode": rank_mode,
            "lora_target_alpha_mode": alpha_mode,
            "merge_method": method,
            "device_used": str(device),
            "models_merged": ", ".join([f"Slot{c['slot_idx']+1}(α={c['weight']}):{c['name']}" for c in loaded_models])
        }
        self.analysis_rows = analysis_data
        self.singular_value_data = singular_value_records
        self.refresh_table()

        avg_energy = np.mean([r["energy"] for r in analysis_data]) if analysis_data else 100.0
        self.global_status_var.set(
            f"Done on {device.type.upper()}! Merged {len(all_layer_names)} Layers | Method: {method} | Mean SVD Energy: {avg_energy:.2f}% | Lossy (<80%): {total_lossy_ranks}"
        )

    def refresh_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        cols = ("layer", "type", "r_m1", "r_m2", "r_m3", "r_m4", "target_r", "alpha", "energy", "merged_l2", "cos_m1", "dora", "status")
        headers = [
            "Layer Name", "Type", "M1 (r)", "M2 (r)", "M3 (r)", "M4 (r)",
            "Target r", "Target α", "SVD Energy", "Merged ΔW (L2)", "CosSim (M1)", "DoRA Mean", "Status"
        ]
        self.tree["columns"] = cols
        self.current_headers = headers

        for col, head in zip(cols, headers):
            self.tree.heading(col, text=head)
            if col == "layer":
                self.tree.column(col, width=320, anchor=tk.W)
            elif col == "status":
                self.tree.column(col, width=140, anchor=tk.CENTER)
            elif col in ["energy", "merged_l2", "cos_m1", "dora"]:
                self.tree.column(col, width=105, anchor=tk.CENTER)
            else:
                self.tree.column(col, width=70, anchor=tk.CENTER)

        for r in self.analysis_rows:
            self.tree.insert("", "end", values=(
                r["layer"], r["is_conv"],
                r["m1_r"], r["m2_r"], r["m3_r"], r["m4_r"],
                r["target_r"], f"{r['target_alpha']:.1f}",
                f"{r['energy']:.1f}%", f"{r['merged_l2']:.4f}",
                f"{r['cos_m1']:.4f}",
                f"{r['dora_mean']:.4f}" if r["has_dora"] else "None",
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
            if isinstance(v, torch.Tensor):
                # Ensure every tensor is contiguous in memory on CPU
                if v.dtype.is_floating_point:
                    out_dict[k] = v.to(cast_dtype).contiguous().cpu()
                else:
                    out_dict[k] = v.contiguous().cpu()
            else:
                out_dict[k] = v

        try:
            save_file(out_dict, save_path, metadata=self.merged_metadata)
            messagebox.showinfo("Success", f"Successfully exported merged LoRA model to:\n{save_path}")
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
            messagebox.showinfo("Success", "LoRA merge analysis exported successfully!")
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not write CSV:\n{e}")

    def show_charts(self):
        if not self.analysis_rows:
            messagebox.showwarning("No Data", "Please run merge first to inspect analytics.")
            return

        chart_win = tk.Toplevel(self.root)
        chart_win.title(f"LoRA Merged Analytics & SVD Subspace Spectrum [{self.merge_method_var.get()}]")
        chart_win.geometry("1200x760")
        chart_win.configure(bg=self.bg_color)

        notebook = ttk.Notebook(chart_win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        plt.style.use('dark_background')
        layers = [r["layer"].split('.')[-1][:16] for r in self.analysis_rows]
        x = np.arange(len(layers))
        colors = ["#42a5f5", "#ff7043", "#66bb6a", "#ab47bc"]

        # Tab 1: SVD Energy Retention (%)
        f_eng = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
        ax_eng = f_eng.add_subplot(111)
        energies = [r["energy"] for r in self.analysis_rows]
        ax_eng.plot(x, energies, label='SVD Energy Retained (%)', color='#00e676', marker='o', linewidth=2.0)
        ax_eng.axhline(95.0, color='#81c784', linestyle='--', label='95% Threshold (Near Lossless)')
        ax_eng.axhline(80.0, color='#ff5252', linestyle=':', label='80% Threshold (Lossy Cutoff)')
        ax_eng.set_title("SVD Energy Retention (%) across Layers")
        ax_eng.set_xticks(x); ax_eng.set_xticklabels(layers, rotation=90, fontsize=8)
        ax_eng.set_ylabel("Retained Energy %")
        ax_eng.grid(True, alpha=0.2)
        ax_eng.legend()
        self.add_tab(notebook, f_eng, "SVD Energy Retention")

        # Tab 2: Delta W L2 Norm Comparison
        f_l2 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
        ax_l2 = f_l2.add_subplot(111)
        for i in range(4):
            if any(r[f"m{i+1}_l2"] > 0 for r in self.analysis_rows):
                ax_l2.plot(x, [r[f"m{i+1}_l2"] for r in self.analysis_rows], label=f'Model {i+1} ΔW', color=colors[i], alpha=0.5, linestyle=':')
        ax_l2.plot(x, [r["merged_l2"] for r in self.analysis_rows], label='Merged ΔW (L2)', color='#ffd54f', linewidth=2.0)
        ax_l2.set_title("LoRA Weight Delta (‖ΔW‖) Magnitudes")
        ax_l2.set_xticks(x); ax_l2.set_xticklabels(layers, rotation=90, fontsize=8)
        ax_l2.set_ylabel("Frobenius / L2 Norm")
        ax_l2.grid(True, alpha=0.2)
        ax_l2.legend()
        self.add_tab(notebook, f_l2, "ΔW Magnitudes")

        # Tab 3: Cosine Similarity with Model 1 (Directional Drift)
        f_cos = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
        ax_cos = f_cos.add_subplot(111)
        ax_cos.plot(x, [r["cos_m1"] for r in self.analysis_rows], label='CosSim(Merged, M1)', color='#29b6f6', marker='^', linewidth=1.8)
        ax_cos.axhline(1.0, color='gray', linestyle='--')
        ax_cos.set_title("Subspace Alignment with Model 1 (Cosine Similarity)")
        ax_cos.set_xticks(x); ax_cos.set_xticklabels(layers, rotation=90, fontsize=8)
        ax_cos.set_ylabel("Cosine Similarity")
        ax_cos.grid(True, alpha=0.2)
        ax_cos.legend()
        self.add_tab(notebook, f_cos, "Cosine Alignment")

        # Tab 4: Top Singular Values for Mid-layer
        if self.singular_value_data:
            f_sv = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax_sv = f_sv.add_subplot(111)
            mid_layer_name = self.analysis_rows[len(self.analysis_rows) // 2]["layer"]
            svs = self.singular_value_data[mid_layer_name]
            ax_sv.bar(np.arange(len(svs)), svs, color="#ba68c8", alpha=0.8)
            ax_sv.set_title(f"Singular Value Spectrum (Mid Layer: {mid_layer_name.split('.')[-1]})")
            ax_sv.set_xlabel("Rank Index")
            ax_sv.set_ylabel("Singular Value (σ)")
            ax_sv.grid(True, alpha=0.2)
            self.add_tab(notebook, f_sv, "Singular Spectrum")

    def add_tab(self, notebook, figure, title):
        frame = tk.Frame(notebook, bg=self.bg_color)
        figure.tight_layout()
        canvas = FigureCanvasTkAgg(figure, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        notebook.add(frame, text=title)


if __name__ == "__main__":
    root = tk.Tk()
    app = LoRAMergerApp(root)
    root.mainloop()