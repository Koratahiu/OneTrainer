import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import re
import csv
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ScheduleInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Entropic Schedule Inspector Pro + Timestep Density Analytics")
        self.root.geometry("1550x920")

        # --- Dark Mode Colors ---
        self.bg_color = "#2d2d2d"
        self.fg_color = "#e0e0e0"
        self.entry_bg = "#3d3d3d"
        self.root.configure(bg=self.bg_color)

        # State Variables
        self.mode_var = tk.StringVar(value="compare")
        self.file1_path = tk.StringVar()
        self.file2_path = tk.StringVar()
        self.global_diff_var = tk.StringVar(value="Ready.")
        self.file1_info_var = tk.StringVar(value="Schedule 1: No file loaded.")
        self.file2_info_var = tk.StringVar(value="Schedule 2: No file loaded.")

        # Analytics Settings
        self.sim_samples_var = tk.IntVar(value=50000)
        self.density_smooth_var = tk.BooleanVar(value=True)

        # In-memory cached analysis results
        self.cached_s1 = {}
        self.cached_s2 = {}
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
        tk.Radiobutton(mode_frame, text="Single Schedule", variable=self.mode_var, value="single", command=self.toggle_mode, bg=self.bg_color, fg=self.fg_color, selectcolor=self.entry_bg).pack(side=tk.LEFT, padx=6)
        tk.Radiobutton(mode_frame, text="Compare Two Schedules", variable=self.mode_var, value="compare", command=self.toggle_mode, bg=self.bg_color, fg=self.fg_color, selectcolor=self.entry_bg).pack(side=tk.LEFT, padx=6)

        # Options Toolbar
        opt_frame = tk.Frame(mode_frame, bg=self.bg_color)
        opt_frame.pack(side=tk.RIGHT)

        tk.Label(opt_frame, text="Sim Samples:", bg=self.bg_color, fg="#80cbc4", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(8, 2))
        tk.Entry(opt_frame, textvariable=self.sim_samples_var, width=8, bg=self.entry_bg, fg=self.fg_color, insertbackground="white").pack(side=tk.LEFT, padx=4)

        # File Inputs
        tk.Label(top_frame, text="Schedule 1 (Baseline):", bg=self.bg_color, fg=self.fg_color).grid(row=1, column=0, sticky="e", pady=2)
        tk.Entry(top_frame, textvariable=self.file1_path, width=82, bg=self.entry_bg, fg=self.fg_color, insertbackground="white").grid(row=1, column=1, padx=5)
        tk.Button(top_frame, text="Browse", command=lambda: self.browse_file(self.file1_path, self.file1_info_var), bg="#555555", fg="white").grid(row=1, column=2)

        self.lbl_file1_info = tk.Label(top_frame, textvariable=self.file1_info_var, font=("Consolas", 8), fg="#81c784", bg=self.bg_color)
        self.lbl_file1_info.grid(row=2, column=1, sticky="w", padx=5)

        self.lbl_file2 = tk.Label(top_frame, text="Schedule 2 (Target):", bg=self.bg_color, fg=self.fg_color)
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

        # Tree tags for status highlights
        self.tree.tag_configure("high_density", foreground="#ffb74d")  # Orange (Peak learning)
        self.tree.tag_configure("detail_phase", foreground="#81c784")  # Green (Refinement)
        self.tree.tag_configure("semantic_phase", foreground="#64b5f6") # Blue (Macro structure)
        self.tree.tag_configure("divergent", foreground="#ff5252")     # Red (Large shift between schedules)

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
        path = filedialog.askopenfilename(filetypes=[("Schedule Files", "*.pt *.pth *.bin"), ("All files", "*.*")])
        if path:
            path_var.set(path)
            self.inspect_file_header(path, info_var)

    def inspect_file_header(self, path, info_var):
        if not path or not os.path.exists(path):
            info_var.set("File not found.")
            return
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(data, dict) or 'time' not in data or 'time_func' not in data:
                info_var.set("Invalid schedule format (keys 'time' and 'time_func' required).")
                return

            time = data['time']
            func = data['time_func']
            steps = len(time)
            help_str = data.get('help', '')

            # Extract samples / resolution info if present
            samples_match = re.search(r'(\d+)\s+samples', help_str)
            res_match = re.search(r'~?(\d+)x(\d+)', help_str)
            samples = samples_match.group(1) if samples_match else "N/A"
            res = f"{res_match.group(1)}x{res_match.group(2)}" if res_match else "Native"

            auc = float(np.trapezoid(func.numpy(), time.numpy()))
            info_var.set(f"Steps: {steps} | Samples: {samples} | Res: {res} | AUC: {auc:.4f} | Range: [{float(time[0]):.2f}, {float(time[-1]):.2f}]")
        except Exception as e:
            info_var.set(f"Could not read schedule: {e}")

    def load_schedule(self, path):
        if not path or not os.path.exists(path):
            return {}
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(data, dict) or 'time' not in data or 'time_func' not in data:
                messagebox.showerror("Error", f"File {os.path.basename(path)} does not contain 'time' and 'time_func' keys.")
                return {}

            time = data['time'].float().numpy()
            time_func = data['time_func'].float().numpy()
            help_str = data.get('help', '')

            # Sort ascending just in case
            sort_idx = np.argsort(time)
            time = time[sort_idx]
            time_func = time_func[sort_idx]

            # Numerical derivatives
            dt = np.diff(time)
            df = np.diff(time_func)
            dt_safe = np.where(dt <= 1e-12, 1e-12, dt)
            point_density = df / dt_safe
            density = np.concatenate([[point_density[0]], point_density])

            # AUC and midpoint
            auc = float(np.trapezoid(time_func, time))
            # Find sigma at 50% entropy
            sigma_at_half = float(np.interp(0.5, time_func, time))

            return {
                'path': path,
                'time': time,
                'time_func': time_func,
                'density': density,
                'auc': auc,
                'sigma_half': sigma_at_half,
                'help': help_str,
                'n_steps': len(time)
            }
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {path}:\n{e}")
            return {}

    def format_tree_columns(self, cols, headers):
        self.tree["columns"] = cols
        self.current_columns = headers
        for col, head in zip(cols, headers):
            self.tree.heading(col, text=head)
            if col in ["layer", "regime", "status"]:
                width = 220
                anchor = tk.W
            else:
                width = 110
                anchor = tk.CENTER
            self.tree.column(col, width=width, anchor=anchor)

    def run_analysis(self):
        s1 = self.load_schedule(self.file1_path.get())
        if not s1: return
        self.cached_s1 = s1

        if self.mode_var.get() == "compare":
            s2 = self.load_schedule(self.file2_path.get())
            if not s2: return
            self.cached_s2 = s2
        else:
            self.cached_s2 = {}

        self.refresh_display()

    def refresh_display(self):
        if not self.cached_s1:
            return

        for item in self.tree.get_children():
            self.tree.delete(item)
        self.results_data = []

        mode = self.mode_var.get()

        if mode == "single":
            cols = ("step", "sigma", "entropy", "d_sigma", "d_entropy", "density", "bias_uniform", "regime")
            headers = [
                "Step Index", "Sigma (σ)", "Entropy S(σ)", "Δσ", "ΔS",
                "Density (dS/dσ)", "Bias vs Uniform (S-σ)", "Dynamic Regime"
            ]
            self.format_tree_columns(cols, headers)

            s = self.cached_s1
            t = s['time']
            f = s['time_func']
            dens = s['density']
            max_dens = np.max(dens)

            for i in range(len(t)):
                d_sig = (t[i] - t[i-1]) if i > 0 else 0.0
                d_ent = (f[i] - f[i-1]) if i > 0 else 0.0
                bias = f[i] - t[i]
                cur_dens = dens[i]

                # Regime heuristics
                if cur_dens >= 0.75 * max_dens:
                    regime = "🔥 Peak Information Zone"
                    tag = "high_density"
                elif t[i] > 0.65:
                    regime = "Macro / Semantic Phase"
                    tag = "semantic_phase"
                elif t[i] < 0.25:
                    regime = "Fine Detail Phase"
                    tag = "detail_phase"
                else:
                    regime = "Transition Phase"
                    tag = ""

                row_dict = {
                    "step": i,
                    "sigma": t[i],
                    "entropy": f[i],
                    "d_sigma": d_sig,
                    "d_entropy": d_ent,
                    "density": cur_dens,
                    "bias_uniform": bias,
                    "regime": regime
                }
                self.results_data.append(row_dict)

                self.tree.insert("", "end", values=(
                    f"#{i:02d}", f"{t[i]:.4f}", f"{f[i]:.4f}",
                    f"{d_sig:.4f}", f"{d_ent:.4f}", f"{cur_dens:.3f}",
                    f"{bias:+.4f}", regime
                ), tags=(tag,))

            self.global_diff_var.set(
                f"Steps: {len(t)} | AUC: {s['auc']:.4f} (Uniform: 0.5000) | Midpoint (S=0.5): σ={s['sigma_half']:.3f} | Max Density: {max_dens:.2f}"
            )

        else: # Compare Mode
            s1, s2 = self.cached_s1, self.cached_s2
            cols = ("sigma", "ent_1", "ent_2", "diff_ent", "dens_1", "dens_2", "diff_dens", "status")
            headers = [
                "Sigma (σ)", "Schedule 1 S(σ)", "Schedule 2 S(σ)", "Δ Entropy",
                "Density 1", "Density 2", "Δ Density %", "Status"
            ]
            self.format_tree_columns(cols, headers)

            # Create a unified sigma grid for comparison (50 uniform evaluation points)
            eval_sigma = np.linspace(0.0, 1.0, 51)
            f1_interp = np.interp(eval_sigma, s1['time'], s1['time_func'])
            f2_interp = np.interp(eval_sigma, s2['time'], s2['time_func'])
            d1_interp = np.interp(eval_sigma, s1['time'], s1['density'])
            d2_interp = np.interp(eval_sigma, s2['time'], s2['density'])

            max_diff = 0.0
            for i, sig in enumerate(eval_sigma):
                diff_e = f2_interp[i] - f1_interp[i]
                d1_val = d1_interp[i]
                d2_val = d2_interp[i]
                diff_d_pct = ((d2_val - d1_val) / (d1_val + 1e-6)) * 100.0

                tag = ""
                if abs(diff_e) > 0.10:
                    status = "DIVERGENT (>0.10 shift)"
                    tag = "divergent"
                elif abs(diff_e) > 0.04:
                    status = "Shifted"
                    tag = "high_density"
                else:
                    status = "Aligned"

                max_diff = max(max_diff, abs(diff_e))

                row_dict = {
                    "sigma": sig,
                    "ent_1": f1_interp[i],
                    "ent_2": f2_interp[i],
                    "diff_ent": diff_e,
                    "dens_1": d1_val,
                    "dens_2": d2_val,
                    "diff_dens": diff_d_pct,
                    "status": status
                }
                self.results_data.append(row_dict)

                self.tree.insert("", "end", values=(
                    f"{sig:.3f}", f"{f1_interp[i]:.4f}", f"{f2_interp[i]:.4f}",
                    f"{diff_e:+.4f}", f"{d1_val:.3f}", f"{d2_val:.3f}",
                    f"{diff_d_pct:+.1f}%", status
                ), tags=(tag,))

            auc_diff = s2['auc'] - s1['auc']
            mid_diff = s2['sigma_half'] - s1['sigma_half']
            self.global_diff_var.set(
                f"ΔAUC: {auc_diff:+.4f} | ΔMidpoint (S=0.5): {mid_diff:+.3f}σ | Max Pointwise Δ: {max_diff:.4f}"
            )

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
                    for row in self.results_data:
                        writer.writerow([row.get(col, "") for col in self.tree["columns"]])
                messagebox.showinfo("Success", "Schedule analytics exported successfully!")
            except PermissionError:
                messagebox.showerror("Error", "Permission Denied! File may be open in another application.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export CSV:\n{e}")

    def show_charts(self):
        if not self.cached_s1:
            messagebox.showwarning("No Data", "Please load a schedule and run analysis first.")
            return

        chart_win = tk.Toplevel(self.root)
        chart_win.title("Entropic Schedule Visual Analytics")
        chart_win.geometry("1200x780")
        chart_win.configure(bg=self.bg_color)

        notebook = ttk.Notebook(chart_win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        plt.style.use('dark_background')
        n_sim = self.sim_samples_var.get()

        if self.mode_var.get() == "single":
            s = self.cached_s1
            t = s['time']
            f = s['time_func']
            dens = s['density']

            # Tab 1: Cumulative Entropy vs Uniform
            fig1 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax1 = fig1.add_subplot(111)
            ax1.plot(t, f, label='Entropic Time S(σ)', color='#64b5f6', marker='o', linewidth=2)
            ax1.plot([0, 1], [0, 1], label='Uniform Schedule (Baseline)', color='#e0e0e0', linestyle='--', alpha=0.6)
            ax1.axhline(0.5, color='#ffd54f', linestyle=':', label=f'50% Entropy (σ={s["sigma_half"]:.3f})')
            ax1.axvline(s["sigma_half"], color='#ffd54f', linestyle=':')
            ax1.fill_between(t, f, t, color='#64b5f6', alpha=0.15, label='Information Reallocation')
            ax1.set_title("Cumulative Entropic Time vs. Uniform Noise Schedule")
            ax1.set_xlabel("Noise Level (Sigma σ)")
            ax1.set_ylabel("Normalized Entropy S(σ)")
            ax1.grid(True, alpha=0.2)
            ax1.legend()
            self.add_tab(notebook, fig1, "Cumulative Entropy S(σ)")

            # Tab 2: Information Density (Marginal Weight)
            fig2 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax2 = fig2.add_subplot(111)
            ax2.plot(t, dens, label='Information Density (dS/dσ)', color='#ff7043', marker='s', linewidth=2)
            ax2.axhline(1.0, color='white', linestyle='--', label='Uniform Density (1.0)', alpha=0.6)
            ax2.fill_between(t, dens, color='#ff7043', alpha=0.2)
            ax2.set_title("Information Density / Timestep Importance per Noise Level")
            ax2.set_xlabel("Noise Level (Sigma σ)")
            ax2.set_ylabel("Marginal Sampling Density")
            ax2.grid(True, alpha=0.2)
            ax2.legend()
            self.add_tab(notebook, fig2, "Density (dS/dσ)")

            # Tab 3: Simulated Training Timestep Distribution
            fig3 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax3 = fig3.add_subplot(111)
            # Inverse CDF sampling simulation: u ~ U(0, 1) -> sigma = S^-1(u)
            u = np.random.uniform(0.0, 1.0, size=n_sim)
            sampled_sigmas = np.interp(u, f, t)
            ax3.hist(sampled_sigmas, bins=40, density=True, color='#81c784', alpha=0.7, edgecolor='#2d2d2d', label=f'Simulated Batches (N={n_sim})')
            ax3.plot(t, dens, color='#ffb74d', linewidth=2, label='Theoretical Sampling Density')
            ax3.set_title(f"Simulated Timestep Allocation During Training (Where Compute is Spent)")
            ax3.set_xlabel("Sampled Noise Level (Sigma σ)")
            ax3.set_ylabel("Effective Probability Density")
            ax3.grid(True, alpha=0.2)
            ax3.legend()
            self.add_tab(notebook, fig3, "Simulated Sampling")

            # Tab 4: Step Spacing (dt)
            fig4 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax4 = fig4.add_subplot(111)
            steps = np.arange(len(t))
            ax4.bar(steps, np.concatenate([[0], np.diff(t)]), color='#ab47bc', alpha=0.7, label='Δσ per Evaluation Step')
            ax4.set_title("Grid Spacing (Resolution of Sigma Estimation)")
            ax4.set_xlabel("Step Index")
            ax4.set_ylabel("ΔSigma")
            ax4.grid(True, alpha=0.2)
            ax4.legend()
            self.add_tab(notebook, fig4, "Grid Spacing")

        else: # Compare Mode
            s1, s2 = self.cached_s1, self.cached_s2
            eval_sig = np.linspace(0.0, 1.0, 100)
            f1 = np.interp(eval_sig, s1['time'], s1['time_func'])
            f2 = np.interp(eval_sig, s2['time'], s2['time_func'])
            d1 = np.interp(eval_sig, s1['time'], s1['density'])
            d2 = np.interp(eval_sig, s2['time'], s2['density'])

            # Tab 1: Curves Overlay
            fig1 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax1 = fig1.add_subplot(111)
            ax1.plot(eval_sig, f1, label='Schedule 1 (Baseline)', color='#64b5f6', linewidth=2)
            ax1.plot(eval_sig, f2, label='Schedule 2 (Target)', color='#ff7043', linewidth=2)
            ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5, label='Uniform')
            ax1.fill_between(eval_sig, f1, f2, color='gray', alpha=0.25, label='Entropy Divergence')
            ax1.set_title("Entropy Curve Comparison S(σ)")
            ax1.set_xlabel("Sigma σ")
            ax1.set_ylabel("Normalized Entropy")
            ax1.grid(True, alpha=0.2)
            ax1.legend()
            self.add_tab(notebook, fig1, "Entropy Curves")

            # Tab 2: Density Overlay
            fig2 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax2 = fig2.add_subplot(111)
            ax2.plot(eval_sig, d1, label='Schedule 1 Density', color='#64b5f6', linewidth=2)
            ax2.plot(eval_sig, d2, label='Schedule 2 Density', color='#ff7043', linewidth=2)
            ax2.axhline(1.0, linestyle='--', color='white', alpha=0.4)
            ax2.set_title("Information Density Comparison (dS/dσ)")
            ax2.set_xlabel("Sigma σ")
            ax2.set_ylabel("Density / Weight")
            ax2.grid(True, alpha=0.2)
            ax2.legend()
            self.add_tab(notebook, fig2, "Density Comparison")

            # Tab 3: Pointwise Drift
            fig3 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax3 = fig3.add_subplot(111)
            diff_f = f2 - f1
            ax3.plot(eval_sig, diff_f, label='ΔS(σ) (Schedule 2 - Schedule 1)', color='#00e676', linewidth=2)
            ax3.axhline(0.0, linestyle='--', color='white', alpha=0.5)
            ax3.fill_between(eval_sig, diff_f, 0, where=(diff_f >= 0), color='#00e676', alpha=0.2)
            ax3.fill_between(eval_sig, diff_f, 0, where=(diff_f < 0), color='#ff5252', alpha=0.2)
            ax3.set_title("Pointwise Schedule Drift ΔS(σ)")
            ax3.set_xlabel("Sigma σ")
            ax3.set_ylabel("Δ Entropy Shift")
            ax3.grid(True, alpha=0.2)
            ax3.legend()
            self.add_tab(notebook, fig3, "Pointwise Drift")

            # Tab 4: Simulated Sampling Comparison
            fig4 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax4 = fig4.add_subplot(111)
            u = np.random.uniform(0.0, 1.0, size=n_sim)
            sig1 = np.interp(u, s1['time_func'], s1['time'])
            sig2 = np.interp(u, s2['time_func'], s2['time'])
            ax4.hist(sig1, bins=40, density=True, color='#64b5f6', alpha=0.5, label='Schedule 1 Allocation')
            ax4.hist(sig2, bins=40, density=True, color='#ff7043', alpha=0.5, label='Schedule 2 Allocation')
            ax4.set_title("Effective Timestep Allocation Comparison (Inverse Sampling)")
            ax4.set_xlabel("Sampled Noise Level (Sigma σ)")
            ax4.set_ylabel("Density")
            ax4.grid(True, alpha=0.2)
            ax4.legend()
            self.add_tab(notebook, fig4, "Training Allocation Overlay")

    def add_tab(self, notebook, figure, title):
        frame = tk.Frame(notebook, bg=self.bg_color)
        figure.tight_layout()
        canvas = FigureCanvasTkAgg(figure, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        notebook.add(frame, text=title)


if __name__ == "__main__":
    root = tk.Tk()
    app = ScheduleInspectorApp(root)
    root.mainloop()