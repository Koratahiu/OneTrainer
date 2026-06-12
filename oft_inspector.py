import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import torch
from safetensors.torch import load_file
import os
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class OFTInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OFT Weight Inspector Pro + Analytics & Diff (Dual DOFT Support)")
        self.root.geometry("1500x850")
        
        # --- Dark Mode Colors ---
        self.bg_color = "#2d2d2d"
        self.fg_color = "#e0e0e0"
        self.entry_bg = "#3d3d3d"
        self.root.configure(bg=self.bg_color)

        self.mode_var = tk.StringVar(value="compare")
        self.file1_path = tk.StringVar()
        self.file2_path = tk.StringVar()
        self.global_diff_var = tk.StringVar(value="Ready.")
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
        top_frame = tk.Frame(self.root, padx=10, pady=10, bg=self.bg_color)
        top_frame.pack(fill=tk.X)

        # Mode Selection
        mode_frame = tk.Frame(top_frame, bg=self.bg_color)
        mode_frame.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        tk.Label(mode_frame, text="Operation Mode:", bg=self.bg_color, fg=self.fg_color, font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Inspect Single OFT", variable=self.mode_var, value="single", command=self.toggle_mode, bg=self.bg_color, fg=self.fg_color, selectcolor=self.entry_bg).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(mode_frame, text="Compare Two OFTs", variable=self.mode_var, value="compare", command=self.toggle_mode, bg=self.bg_color, fg=self.fg_color, selectcolor=self.entry_bg).pack(side=tk.LEFT)

        # File Inputs
        tk.Label(top_frame, text="OFT 1 (Baseline):", bg=self.bg_color, fg=self.fg_color).grid(row=1, column=0, sticky="e", pady=2)
        tk.Entry(top_frame, textvariable=self.file1_path, width=80, bg=self.entry_bg, fg=self.fg_color, insertbackground="white").grid(row=1, column=1, padx=5)
        tk.Button(top_frame, text="Browse", command=lambda: self.browse_file(self.file1_path), bg="#555555", fg="white").grid(row=1, column=2)

        self.lbl_file2 = tk.Label(top_frame, text="OFT 2 (Target):", bg=self.bg_color, fg=self.fg_color)
        self.lbl_file2.grid(row=2, column=0, sticky="e", pady=2)
        self.entry_file2 = tk.Entry(top_frame, textvariable=self.file2_path, width=80, bg=self.entry_bg, fg=self.fg_color, insertbackground="white")
        self.entry_file2.grid(row=2, column=1, padx=5)
        self.btn_file2 = tk.Button(top_frame, text="Browse", command=lambda: self.browse_file(self.file2_path), bg="#555555", fg="white")
        self.btn_file2.grid(row=2, column=2)

        # --- Button Bar & Global Stats ---
        btn_frame = tk.Frame(self.root, pady=10, bg=self.bg_color)
        btn_frame.pack(fill=tk.X)
        
        tk.Button(btn_frame, text="1. Run Analysis", command=self.run_analysis, bg="#2e7d32", fg="white", width=18).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="2. Show Analytics", command=self.show_charts, bg="#1565c0", fg="white", width=18).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="3. Export CSV", command=self.export_csv, bg="#ef6c00", fg="white", width=18).pack(side=tk.LEFT, padx=5)

        tk.Label(btn_frame, textvariable=self.global_diff_var, font=("Arial", 11, "bold"), fg="#ffb74d", bg=self.bg_color).pack(side=tk.RIGHT, padx=20)

        # --- Table Frame ---
        bottom_frame = tk.Frame(self.root, padx=10, pady=10, bg=self.bg_color)
        bottom_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(bottom_frame, show="headings")
        
        y_scroll = ttk.Scrollbar(bottom_frame, orient=tk.VERTICAL, command=self.tree.yview)
        x_scroll = ttk.Scrollbar(bottom_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscroll=y_scroll.set, xscroll=x_scroll.set)
        
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.toggle_mode()

    def toggle_mode(self):
        if self.mode_var.get() == "single":
            self.lbl_file2.grid_remove()
            self.entry_file2.grid_remove()
            self.btn_file2.grid_remove()
        else:
            self.lbl_file2.grid()
            self.entry_file2.grid()
            self.btn_file2.grid()

    def browse_file(self, path_var):
        path = filedialog.askopenfilename(filetypes=[("OFT Weights", "*.safetensors *.bin *.pt")])
        if path: path_var.set(path)

    def get_rotation_magnitude(self, w_tensor, block_size):
        if block_size <= 1: return 0.0
        r, n = w_tensor.shape
        rows, cols = torch.triu_indices(block_size, block_size, 1)
        matrix = torch.zeros(r, block_size, block_size, device=w_tensor.device, dtype=w_tensor.dtype)
        batch_idx = torch.arange(r, device=w_tensor.device)[:, None]
        matrix = matrix.index_put((batch_idx, rows, cols), w_tensor)
        matrix = matrix - matrix.transpose(-2, -1)
        I = torch.eye(block_size, device=w_tensor.device, dtype=w_tensor.dtype).unsqueeze(0).expand(r, block_size, block_size)
        try:
            R = torch.linalg.solve(I + matrix, I - matrix, left=False)
            diff = R - I
            return torch.linalg.norm(diff).item() / math.sqrt(r)
        except Exception:
            return 0.0

    def get_metrics(self, path):
        if not path or not os.path.exists(path): return {}
        try:
            sd = load_file(path) if path.endswith(".safetensors") else torch.load(path, map_location="cpu")
            layers = {}
            
            # --- Pre-scan for Dual DoRA Multipliers ---
            dora_mults_row = {}
            dora_mults_col = {}
            
            for k, v in sd.items():
                if 'dora_log_multiplier_row' in k:
                    base = k.replace('.dora_log_multiplier_row', '')
                    dora_mults_row[base] = v.float()
                elif 'dora_log_multiplier_col' in k:
                    base = k.replace('.dora_log_multiplier_col', '')
                    dora_mults_col[base] = v.float()
                elif 'dora_log_multiplier' in k: # Fallback for old single DoRAOFT exports
                    base = k.replace('.dora_log_multiplier', '')
                    dora_mults_row[base] = v.float()
            # -------------------------------

            for k, v in sd.items():
                if 'oft' not in k.lower() and 'weight' not in k: continue
                if 'dora_log_multiplier' in k: continue 
                
                # Clean key names
                base = k.replace(".oft_R.weight", "").replace(".weight", "")
                
                t = v.float()
                if len(t.shape) != 2: continue # OFT weights are typically (r, n_elements)
                
                r_blocks, n_elements = t.shape
                block_size = int((1 + math.sqrt(1 + 8 * n_elements)) / 2)
                
                l2_norm = torch.linalg.norm(t).item()
                rot_shift = self.get_rotation_magnitude(t, block_size)

                # --- Dual DoRA Extraction ---
                has_dora_row = base in dora_mults_row
                has_dora_col = base in dora_mults_col
                
                # Default is 1.0 (no scale modification)
                # Instead of the average, find the maximum absolute deviation from 1.0
                if has_dora_row:
                    row_vals = torch.exp(dora_mults_row[base])
                    # Find the value furthest from 1.0
                    dora_mult_row = row_vals[torch.argmax(torch.abs(row_vals - 1.0))].item()
                else:
                    dora_mult_row = 1.0

                if has_dora_col:
                    col_vals = torch.exp(dora_mults_col[base] - dora_mults_col[base].mean())
                    dora_mult_col = col_vals[torch.argmax(torch.abs(col_vals - 1.0))].item()
                else:
                    dora_mult_col = 1.0

                layers[base] = {
                    'blocks': r_blocks,
                    'block_size': block_size,
                    'w_l2': l2_norm,
                    'rot_shift': rot_shift,
                    'has_dora': has_dora_row or has_dora_col,
                    'dora_mult_row': dora_mult_row,
                    'dora_mult_col': dora_mult_col
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
            width = 300 if col == "layer" else 115
            self.tree.column(col, width=width, anchor=tk.W if col == "layer" else tk.CENTER)

    def run_analysis(self):
        self.results_data = []
        for item in self.tree.get_children(): self.tree.delete(item)
        mode = self.mode_var.get()

        r1 = self.get_metrics(self.file1_path.get())
        if not r1: return

        if mode == "single":
            cols = ("layer", "blocks", "block_size", "w_l2", "rot_shift", "has_dora", "dora_mult_row", "dora_mult_col")
            headers = ["Layer Name", "Blocks (R)", "Block Size", "Weight (L2 Norm)", "Rotation Shift (‖R-I‖)", "DoRA", "DoRA Row(Out)", "DoRA Col(In)"]
            self.format_tree_columns(cols, headers)

            for k in sorted(r1.keys()):
                d = r1[k]
                self.results_data.append({"layer": k, **d})
                self.tree.insert("", "end", values=(k, d['blocks'], d['block_size'], f"{d['w_l2']:.4f}", f"{d['rot_shift']:.4f}", "Yes" if d['has_dora'] else "No", f"{d['dora_mult_row']:.4f}", f"{d['dora_mult_col']:.4f}"))
            
            self.global_diff_var.set(f"Loaded {len(r1)} OFT layers successfully.")

        else: # Compare Mode
            r2 = self.get_metrics(self.file2_path.get())
            if not r2: return

            cols = ("layer", "w_l2_1", "w_l2_2", "diff_w", "rot_1", "rot_2", "diff_rot", "dora_status", "diff_dora_row", "diff_dora_col")
            headers = ["Layer Name", "OFT1 W(L2)", "OFT2 W(L2)", "Δ W(L2)%", "OFT1 RotShift", "OFT2 RotShift", "Δ RotShift%", "DoRA(1|2)", "Δ DoRA Row%", "Δ DoRA Col%"]
            self.format_tree_columns(cols, headers)

            all_keys = sorted(set(list(r1.keys()) + list(r2.keys())))
            g = {key: 0 for key in ["w_l2_1", "w_l2_2", "rot_1", "rot_2", "dora_r_1", "dora_r_2", "dora_c_1", "dora_c_2"]}
            any_dora = False

            for k in all_keys:
                d1 = r1.get(k, {'w_l2':0, 'rot_shift':0, 'has_dora':False, 'dora_mult_row':1.0, 'dora_mult_col':1.0})
                d2 = r2.get(k, {'w_l2':0, 'rot_shift':0, 'has_dora':False, 'dora_mult_row':1.0, 'dora_mult_col':1.0})
                if d1['has_dora'] or d2['has_dora']:
                    any_dora = True

                diff_w = self.calc_pct_diff(d1['w_l2'], d2['w_l2'])
                diff_rot = self.calc_pct_diff(d1['rot_shift'], d2['rot_shift'])
                diff_dora_row = self.calc_pct_diff(d1['dora_mult_row'], d2['dora_mult_row'])
                diff_dora_col = self.calc_pct_diff(d1['dora_mult_col'], d2['dora_mult_col'])

                dora_str = f"{'Y' if d1['has_dora'] else 'N'} | {'Y' if d2['has_dora'] else 'N'}"

                row_data = {
                    "layer": k,
                    "w_l2_1": d1['w_l2'], "w_l2_2": d2['w_l2'], "diff_w": diff_w,
                    "rot_1": d1['rot_shift'], "rot_2": d2['rot_shift'], "diff_rot": diff_rot,
                    "dora_r_1": d1['dora_mult_row'], "dora_r_2": d2['dora_mult_row'], "diff_dora_row": diff_dora_row,
                    "dora_c_1": d1['dora_mult_col'], "dora_c_2": d2['dora_mult_col'], "diff_dora_col": diff_dora_col,
                    "dora_status": dora_str
                }
                self.results_data.append(row_data)

                self.tree.insert("", "end", values=(
                    k, 
                    f"{d1['w_l2']:.4f}", f"{d2['w_l2']:.4f}", f"{diff_w:+.2f}%",
                    f"{d1['rot_shift']:.4f}", f"{d2['rot_shift']:.4f}", f"{diff_rot:+.2f}%",
                    dora_str, f"{diff_dora_row:+.2f}%", f"{diff_dora_col:+.2f}%"
                ))

                g["w_l2_1"] += d1['w_l2']; g["w_l2_2"] += d2['w_l2']
                g["rot_1"] += d1['rot_shift']; g["rot_2"] += d2['rot_shift']
                g["dora_r_1"] += d1['dora_mult_row']; g["dora_r_2"] += d2['dora_mult_row']
                g["dora_c_1"] += d1['dora_mult_col']; g["dora_c_2"] += d2['dora_mult_col']

            gd_w = self.calc_pct_diff(g["w_l2_1"], g["w_l2_2"])
            gd_r = self.calc_pct_diff(g["rot_1"], g["rot_2"])
            
            if any_dora:
                gd_d_r = self.calc_pct_diff(g["dora_r_1"], g["dora_r_2"])
                gd_d_c = self.calc_pct_diff(g["dora_c_1"], g["dora_c_2"])
                self.global_diff_var.set(f"Drift → W(L2): {gd_w:+.2f}% | Rot: {gd_r:+.2f}% | DoRA Row: {gd_d_r:+.2f}% | DoRA Col: {gd_d_c:+.2f}%")
            else:
                self.global_diff_var.set(f"Global Drift → Weight L2: {gd_w:+.2f}%  |  Rotation Shift: {gd_r:+.2f}%")

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
                        writer.writerow(list(row.values()))
                messagebox.showinfo("Success", "Data exported successfully!")
            except PermissionError:
                messagebox.showerror("Error", "Permission Denied! Make sure the file isn't open in Excel or another program.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export CSV:\n{str(e)}")

    def show_charts(self):
        if not self.results_data: 
            messagebox.showwarning("No Data", "Please run analysis first.")
            return
        
        chart_win = tk.Toplevel(self.root)
        chart_win.title("OFT Advanced Analytics")
        chart_win.geometry("1100x700")
        chart_win.configure(bg=self.bg_color)
        
        notebook = ttk.Notebook(chart_win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        plt.style.use('dark_background')
        layers = [r["layer"].split('.')[-1][:15] for r in self.results_data]
        x = np.arange(len(layers))

        if self.mode_var.get() == "single":
            # Tab 1: Weight L2 Norms
            f1 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax1 = f1.add_subplot(111)
            ax1.bar(x, [r["w_l2"] for r in self.results_data], 0.5, label='Weight L2 Norm', color='#64b5f6')
            ax1.set_title("OFT Weight Magnitude (L2 Norm) per Layer")
            ax1.set_xticks(x); ax1.set_xticklabels(layers, rotation=90, fontsize=8)
            ax1.legend()
            self.add_tab(notebook, f1, "Weight L2 Norm")

            # Tab 2: Rotation Shifts
            f2 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax2 = f2.add_subplot(111)
            ax2.plot(x, [r["rot_shift"] for r in self.results_data], label='Rotation Shift (‖R-I‖)', marker='o', color='#f44336')
            ax2.set_title("OFT Matrix Rotation Shift per Layer")
            ax2.set_xticks(x); ax2.set_xticklabels(layers, rotation=90, fontsize=8)
            ax2.grid(True, alpha=0.2)
            ax2.legend()
            self.add_tab(notebook, f2, "Rotation Shift")

            # Tab 3: Dual DoRA Multipliers (Conditional)
            if any(r.get("has_dora") for r in self.results_data):
                f3 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
                ax3 = f3.add_subplot(111)
                ax3.bar(x - 0.2, [r.get("dora_mult_row", 1.0) for r in self.results_data], 0.4, label='DoRA Row (Output)', color='#ab47bc')
                ax3.bar(x + 0.2, [r.get("dora_mult_col", 1.0) for r in self.results_data], 0.4, label='DoRA Col (Input)', color='#ff9800')
                ax3.axhline(1.0, color='white', linestyle='--', linewidth=1) # Baseline multiplier line
                ax3.set_title("Dual DoRA Multipliers Magnitude per Layer (Baseline = 1.0)")
                ax3.set_xticks(x); ax3.set_xticklabels(layers, rotation=90, fontsize=8)
                ax3.legend()
                self.add_tab(notebook, f3, "DoRA Multipliers")

        else:
            has_dora = any(r.get("dora_r_1", 1.0) != 1.0 or r.get("dora_r_2", 1.0) != 1.0 or r.get("dora_c_1", 1.0) != 1.0 for r in self.results_data)

            # Tab 1: Weight L2 Compare
            f1 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax1 = f1.add_subplot(111)
            ax1.plot(x, [r["w_l2_1"] for r in self.results_data], label='OFT 1 - W(L2)', alpha=0.8, color='#64b5f6')
            ax1.plot(x, [r["w_l2_2"] for r in self.results_data], label='OFT 2 - W(L2)', alpha=0.8, color='#ffb74d')
            ax1.fill_between(x, [r["w_l2_1"] for r in self.results_data], [r["w_l2_2"] for r in self.results_data], color='gray', alpha=0.2)
            ax1.set_title("Weight L2 Norm Comparison")
            ax1.set_xticks(x); ax1.set_xticklabels(layers, rotation=90, fontsize=8)
            ax1.legend()
            self.add_tab(notebook, f1, "Weight L2 Compare")

            # Tab 2: Rotation Compare
            f2 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax2 = f2.add_subplot(111)
            ax2.plot(x, [r["rot_1"] for r in self.results_data], label='OFT 1 - RotShift', alpha=0.8, color='#f44336')
            ax2.plot(x, [r["rot_2"] for r in self.results_data], label='OFT 2 - RotShift', alpha=0.8, color='#4caf50')
            ax2.fill_between(x, [r["rot_1"] for r in self.results_data], [r["rot_2"] for r in self.results_data], color='gray', alpha=0.2)
            ax2.set_title("Rotation Magnitude Shift Comparison")
            ax2.set_xticks(x); ax2.set_xticklabels(layers, rotation=90, fontsize=8)
            ax2.legend()
            self.add_tab(notebook, f2, "Rotation Compare")

            # Tab 3: DoRA Compare (Conditional)
            if has_dora:
                f_dora = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
                ax_dora = f_dora.add_subplot(111)
                ax_dora.plot(x, [r.get("dora_r_1", 1.0) for r in self.results_data], label='OFT 1 - Row', alpha=0.8, color='#ab47bc')
                ax_dora.plot(x, [r.get("dora_r_2", 1.0) for r in self.results_data], label='OFT 2 - Row', alpha=0.8, color='#ff7043')
                ax_dora.plot(x, [r.get("dora_c_1", 1.0) for r in self.results_data], label='OFT 1 - Col', alpha=0.8, color='#29b6f6', linestyle='--')
                ax_dora.plot(x, [r.get("dora_c_2", 1.0) for r in self.results_data], label='OFT 2 - Col', alpha=0.8, color='#66bb6a', linestyle='--')
                ax_dora.axhline(1.0, color='white', linestyle='--', linewidth=1)
                ax_dora.set_title("Dual DoRA Relative Scaler Comparison")
                ax_dora.set_xticks(x); ax_dora.set_xticklabels(layers, rotation=90, fontsize=8)
                ax_dora.legend()
                self.add_tab(notebook, f_dora, "DoRA Compare")

            # Tab 4: % Difference Scatter
            f3 = plt.Figure(figsize=(10, 5), facecolor=self.bg_color)
            ax3 = f3.add_subplot(111)
            ax3.scatter(x, [r["diff_w"] for r in self.results_data], label='Δ Weight(L2) %', color='#ba68c8')
            ax3.scatter(x, [r["diff_rot"] for r in self.results_data], label='Δ RotShift %', color='#00bcd4')
            if has_dora:
                ax3.scatter(x, [r.get("diff_dora_row", 0) for r in self.results_data], label='Δ DoRA Row %', color='#ffca28', marker='^')
                ax3.scatter(x, [r.get("diff_dora_col", 0) for r in self.results_data], label='Δ DoRA Col %', color='#4caf50', marker='v')
            ax3.axhline(0, color='white', linestyle='--', linewidth=1)
            ax3.set_title("Percentage Drift Across Layers (Baseline at 0%)")
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