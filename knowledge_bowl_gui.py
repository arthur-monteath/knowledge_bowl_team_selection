from __future__ import annotations
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Iterable
import json
import os
import sys
import tkinter as tk
import math

@dataclass
class Player:
    name: str
    categories: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add_point(self, category: str, points: int = 1) -> None:
        if points < 0:
            raise ValueError("points must be non-negative")
        self.categories[category] += points

    def total_points(self) -> int:
        return sum(self.categories.values())

class KnowledgeBowl:
    def __init__(self) -> None:
        self.players: Dict[str, Player] = {}
        self.category_weights: Dict[str, float] = {}

    def add_player(self, name: str) -> None:
        if name in self.players:
            raise ValueError(f"Player '{name}' already exists.")
        self.players[name] = Player(name=name)

    def add_point(self, name: str, category: str, points: int = 1) -> None:
        if name not in self.players:
            self.add_player(name)
        self.players[name].add_point(category, points)

    def set_weight(self, category: str, weight: float) -> None:
        if weight <= 0:
            raise ValueError("weight must be > 0")
        self.category_weights[category] = float(weight)

    def list_players(self) -> List[str]:
        return sorted(self.players.keys())

    def list_categories(self) -> List[str]:
        cats = set()
        for p in self.players.values():
            cats.update(p.categories.keys())
        cats.update(self.category_weights.keys())
        return sorted(cats)

    def _team_category_totals(self, team_names: Iterable[str]) -> Dict[str, int]:
        totals: Dict[str, int] = defaultdict(int)
        for n in team_names:
            p = self.players[n]
            for c, v in p.categories.items():
                totals[c] += v
        return totals

    def _apply_weights(self, category_totals: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for c, v in category_totals.items():
            w = self.category_weights.get(c, 1.0)
            out[c] = v * w
        for c, w in self.category_weights.items():
            out.setdefault(c, 0.0)
        return out

    def team_score(
        self,
        team_names: Iterable[str],
        aggregator: str = "sum",
        epsilon: float = 1e-9,
    ) -> Tuple[float, Dict[str, float]]:
        totals = self._team_category_totals(team_names)
        for c in self.category_weights:
            totals.setdefault(c, 0.0)
        weighted = self._apply_weights(totals)

        if not weighted:
            return 0.0, {}

        if aggregator == "sum":
            score = float(sum(weighted.values()))
        elif aggregator == "min":
            score = float(min(weighted.values()))
        elif aggregator == "geomean":
            vals = [max(epsilon, float(v)) for v in weighted.values()]
            log_sum = sum(math.log(v) for v in vals)
            score = math.exp(log_sum / len(vals))
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")

        return score, dict(weighted)

    def best_combinations(
        self,
        team_size: int = 4,
        top_n: int = 10,
        aggregator: str = "sum",
    ) -> List[Dict]:
        names = self.list_players()
        if len(names) < team_size:
            raise ValueError(f"Need at least {team_size} players, have {len(names)}.")

        results = []
        for combo in combinations(names, team_size):
            score, breakdown = self.team_score(combo, aggregator=aggregator)
            results.append({"team": list(combo), "score": score, "breakdown": breakdown})

        results.sort(key=lambda r: (-r["score"], tuple(r["team"])))
        return results[:top_n]

    def to_json(self) -> str:
        data = {
            "players": {name: dict(p.categories) for name, p in self.players.items()},
            "category_weights": dict(self.category_weights),
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "KnowledgeBowl":
        data = json.loads(s)
        kb = cls()
        for name, cats in data.get("players", {}).items():
            kb.add_player(name)
            for c, v in cats.items():
                kb.add_point(name, c, int(v))
        for c, w in data.get("category_weights", {}).items():
            kb.set_weight(c, float(w))
        return kb

# Persistence
def data_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "kb_data.json")

def load_kb():
    p = data_path()
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return KnowledgeBowl.from_json(f.read())
        except Exception:
            messagebox.showwarning("Load Error", "Could not read kb_data.json, starting fresh.")
    return KnowledgeBowl()

def save_kb(kb: KnowledgeBowl):
    try:
        with open(data_path(), "w", encoding="utf-8") as f:
            f.write(kb.to_json())
    except Exception as e:
        messagebox.showerror("Save Error", f"Failed to save data: {e}")

# GUI
class KBApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Knowledge Bowl Team Builder")
        self.geometry("1140x670")
        self.minsize(1000, 560)

        self.kb = load_kb()

        # History stacks: States always stored before changes.
        self._undo_stack = []
        self._redo_stack = []
        self._suspend_history = False  # avoid recording during apply of undo/redo
        self._HISTORY_LIMIT = 100

        self._build_style()
        self._build_layout()
        self._refresh_players()
        self._refresh_selected_player_view()
        self._refresh_weights()
        self._update_edit_menu_labels()
        self._update_fixed_toggle_label()

        # Hotkeys
        self.bind_all("<Control-z>", self.on_undo)
        self.bind_all("<Control-y>", self.on_redo)
        # macOS Command-Z/Y
        self.bind_all("<Command-z>", self.on_undo)
        self.bind_all("<Command-y>", self.on_redo)

    # History helpers
    def _snapshot(self):
        return {"kb_json": self.kb.to_json(), "fixed": self._fixed_names()}

    def _apply_snapshot(self, snap):
        self._suspend_history = True
        try:
            self.kb = KnowledgeBowl.from_json(snap["kb_json"])
            self._set_fixed_names(snap.get("fixed", []))
            save_kb(self.kb)
            self._refresh_players()
            self._refresh_selected_player_view()
            self._refresh_weights()
            self._update_fixed_toggle_label()
            self.status_var.set("History applied.")
        finally:
            self._suspend_history = False

    def _record_state(self, label: str):
        if self._suspend_history:
            return
        self._undo_stack.append({"snap": self._snapshot(), "label": label})
        if len(self._undo_stack) > self._HISTORY_LIMIT:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._update_edit_menu_labels()

    def on_undo(self, event=None):
        if not self._undo_stack:
            self.status_var.set("Nothing to undo.")
            return "break"
        cur = {"snap": self._snapshot(), "label": self._undo_stack[-1]["label"]}
        entry = self._undo_stack.pop()
        self._redo_stack.append(cur)
        self._apply_snapshot(entry["snap"])
        self.status_var.set(f"Undid: {entry['label']}")
        self._update_edit_menu_labels()
        return "break"

    def on_redo(self, event=None):
        if not self._redo_stack:
            self.status_var.set("Nothing to redo.")
            return "break"
        cur = {"snap": self._snapshot(), "label": self._redo_stack[-1]["label"]}
        entry = self._redo_stack.pop()
        self._undo_stack.append(cur)
        self._apply_snapshot(entry["snap"])
        self.status_var.set(f"Redid: {entry['label']}")
        self._update_edit_menu_labels()
        return "break"

    # UI
    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TButton", padding=8)
        style.configure("TLabel", padding=2)
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Small.TLabel", foreground="#555")
        style.configure("TEntry", padding=4)
        style.configure("TCombobox", padding=4)
        style.configure("Treeview", rowheight=24)

    def _build_layout(self):
        # Menu
        self.menubar = tk.Menu(self)
        file_menu = tk.Menu(self.menubar, tearoff=0)
        file_menu.add_command(label="Import...", command=self.on_import)
        file_menu.add_command(label="Export...", command=self.on_export)
        file_menu.add_separator()
        file_menu.add_command(label="Open Data File", command=self.open_data_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        self.menubar.add_cascade(label="File", menu=file_menu)

        self.edit_menu = tk.Menu(self.menubar, tearoff=0)
        self.edit_menu.add_command(label="Undo", command=self.on_undo, accelerator="Ctrl+Z")
        self.edit_menu.add_command(label="Redo", command=self.on_redo, accelerator="Ctrl+Y")
        self.menubar.add_cascade(label="Edit", menu=self.edit_menu)

        self.config(menu=self.menubar)

        self.columnconfigure(0, weight=1, uniform="col")
        self.columnconfigure(1, weight=2, uniform="col")
        self.columnconfigure(2, weight=2, uniform="col")
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        # Left: Players
        left = ttk.Frame(self, padding=12)
        left.grid(row=0, column=0, sticky="nsew")
        ttk.Label(left, text="Players", style="Header.TLabel").pack(anchor="w")

        self.player_list = tk.Listbox(left, exportselection=False, selectmode="extended")
        self.player_list.pack(fill="both", expand=True, pady=(8, 8))
        self.player_list.bind("<<ListboxSelect>>", lambda e: (self._refresh_selected_player_view(), self._update_fixed_toggle_label()))

        add_row = ttk.Frame(left)
        add_row.pack(fill="x")
        self.add_name_var = tk.StringVar()
        ttk.Entry(add_row, textvariable=self.add_name_var).pack(side="left", fill="x", expand=True)
        ttk.Button(add_row, text="Add Player", command=self.on_add_player).pack(side="left", padx=(8, 0))

        actions = ttk.Frame(left)
        actions.pack(fill="x", pady=(6,0))
        ttk.Button(actions, text="Remove Selected", command=self.on_remove_selected).pack(side="left")
        self.fixed_toggle_btn = ttk.Button(actions, text="Fix Selected", command=self.on_toggle_fixed)
        self.fixed_toggle_btn.pack(side="left", padx=6)

        # Middle: Add Points + Player detail
        mid = ttk.Frame(self, padding=12)
        mid.grid(row=0, column=1, sticky="nsew")
        mid.rowconfigure(6, weight=1)

        ttk.Label(mid, text="Add Points", style="Header.TLabel").grid(row=0, column=0, columnspan=4, sticky="w")

        ttk.Label(mid, text="Player").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(mid, text="Category").grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Label(mid, text="Points").grid(row=1, column=2, sticky="w", pady=(6, 0))

        self.player_combo = ttk.Combobox(mid, values=[], state="readonly")
        self.player_combo.grid(row=2, column=0, sticky="ew", padx=(0, 6))
        self.category_var = tk.StringVar()
        ttk.Entry(mid, textvariable=self.category_var).grid(row=2, column=1, sticky="ew", padx=(0, 6))
        self.points_var = tk.IntVar(value=1)
        ttk.Spinbox(mid, from_=1, to=1000, textvariable=self.points_var, width=6).grid(row=2, column=2, sticky="w")
        ttk.Button(mid, text="Add Point", command=self.on_add_point).grid(row=2, column=3, padx=(8, 0))

        ttk.Separator(mid, orient="horizontal").grid(row=3, column=0, columnspan=4, sticky="ew", pady=8)

        ttk.Label(mid, text="Selected Player Breakdown", style="Header.TLabel").grid(row=4, column=0, columnspan=4, sticky="w")
        self.breakdown = ttk.Treeview(mid, columns=("cat", "pts"), show="headings", height=10)
        self.breakdown.heading("cat", text="Category")
        self.breakdown.heading("pts", text="Points")
        self.breakdown.column("cat", width=160, anchor="w")
        self.breakdown.column("pts", width=80, anchor="center")
        self.breakdown.grid(row=5, column=0, columnspan=4, sticky="nsew")

        # Right: Compute Results + Fixed players + Weights
        right = ttk.Frame(self, padding=12)
        right.grid(row=0, column=2, sticky="nsew")
        right.rowconfigure(10, weight=1)

        ttk.Label(right, text="Build Teams", style="Header.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")

        ttk.Label(right, text="Team size").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(right, text="Top N").grid(row=1, column=1, sticky="w", pady=(6, 0))

        self.team_size_var = tk.IntVar(value=4)
        ttk.Spinbox(right, from_=2, to=10, textvariable=self.team_size_var, width=6).grid(row=2, column=0, sticky="w")
        self.topn_var = tk.IntVar(value=10)
        ttk.Spinbox(right, from_=1, to=50, textvariable=self.topn_var, width=6).grid(row=2, column=1, sticky="w")

        ttk.Label(right, text="Aggregator").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.agg_var = tk.StringVar(value="sum")
        ttk.Combobox(right, values=["sum", "min", "geomean"], textvariable=self.agg_var, state="readonly").grid(row=4, column=0, sticky="w")

        ttk.Label(right, text="Fixed Players (always included)", style="Header.TLabel").grid(row=5, column=0, columnspan=3, sticky="w", pady=(12,0))
        self.fixed_list = tk.Listbox(right, exportselection=False, height=5, selectmode="extended")
        self.fixed_list.grid(row=6, column=0, columnspan=3, sticky="nsew")

        ttk.Button(right, text="Compute Best Teams", command=self.on_compute).grid(row=7, column=0, columnspan=3, sticky="ew", pady=(10, 6))

        self.results = ttk.Treeview(right, columns=("rank", "team", "score"), show="headings", height=8)
        self.results.heading("rank", text="#")
        self.results.heading("team", text="Team")
        self.results.heading("score", text="Score")
        self.results.column("rank", width=36, anchor="center")
        self.results.column("team", width=320, anchor="w")
        self.results.column("score", width=80, anchor="e")
        self.results.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=(8, 0))

        ttk.Label(right, text="Breakdown (selected result)", style="Header.TLabel").grid(row=9, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self.result_breakdown = tk.Text(right, height=6, wrap="word")
        self.result_breakdown.grid(row=10, column=0, columnspan=3, sticky="nsew")

        self.results.bind("<<TreeviewSelect>>", self.on_select_result)

        # Weights editor (bottom section)
        weights_frame = ttk.LabelFrame(self, text="Category Weights", padding=8)
        weights_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=12, pady=(6,12))
        weights_frame.columnconfigure(1, weight=1)
        weights_frame.columnconfigure(3, weight=1)

        self.weights_tree = ttk.Treeview(weights_frame, columns=("cat","w"), show="headings", height=4)
        self.weights_tree.heading("cat", text="Category")
        self.weights_tree.heading("w", text="Weight")
        self.weights_tree.column("cat", width=200, anchor="w")
        self.weights_tree.column("w", width=80, anchor="center")
        self.weights_tree.grid(row=0, column=0, columnspan=4, sticky="ew")

        ttk.Label(weights_frame, text="Category").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.weight_cat_var = tk.StringVar()
        ttk.Entry(weights_frame, textvariable=self.weight_cat_var).grid(row=1, column=1, sticky="ew", padx=(6,12))

        ttk.Label(weights_frame, text="Weight (>0)").grid(row=1, column=2, sticky="w")
        self.weight_val_var = tk.DoubleVar(value=1.0)
        ttk.Entry(weights_frame, textvariable=self.weight_val_var, width=12).grid(row=1, column=3, sticky="w", padx=(6,0))

        btns = ttk.Frame(weights_frame)
        btns.grid(row=2, column=0, columnspan=4, sticky="w", pady=(6,0))
        ttk.Button(btns, text="Set / Update Weight", command=self.on_set_weight).pack(side="left")
        ttk.Button(btns, text="Remove Weight", command=self.on_remove_weight).pack(side="left", padx=6)
        ttk.Button(btns, text="Refresh", command=self._refresh_weights).pack(side="left")

        # Status bar
        status = ttk.Frame(self, padding=(8,4))
        status.grid(row=2, column=0, columnspan=3, sticky="ew")
        status.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status, textvariable=self.status_var, style="Small.TLabel").grid(row=0, column=0, sticky="w")

    def _update_edit_menu_labels(self):
        if self._undo_stack:
            label = self._undo_stack[-1]["label"]
            self.edit_menu.entryconfig(0, label=f"Undo: {label}", state="normal")
        else:
            self.edit_menu.entryconfig(0, label="Undo", state="disabled")

        if self._redo_stack:
            label = self._redo_stack[-1]["label"]
            self.edit_menu.entryconfig(1, label=f"Redo: {label}", state="normal")
        else:
            self.edit_menu.entryconfig(1, label="Redo", state="disabled")

    # File menu
    def on_import(self):
        path = filedialog.askopenfilename(title="Import KnowledgeBowl JSON",
                                          filetypes=[("JSON files","*.json"), ("All files","*.*")])
        if not path:
            return
        self._record_state(f"Import from {os.path.basename(path)}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.kb = KnowledgeBowl.from_json(f.read())
            save_kb(self.kb)
            self._refresh_players()
            self._refresh_selected_player_view()
            self._refresh_weights()
            self._update_fixed_toggle_label()
            self.status_var.set(f"Imported from {os.path.basename(path)} and saved.")
        except Exception as e:
            messagebox.showerror("Import", str(e))

    def on_export(self):
        path = filedialog.asksaveasfilename(title="Export KnowledgeBowl JSON",
                                            defaultextension=".json",
                                            filetypes=[("JSON files","*.json"), ("All files","*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.kb.to_json())
            self.status_var.set(f"Exported to {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Export", str(e))

    def open_data_file(self):
        p = data_path()
        if not os.path.exists(p):
            save_kb(self.kb)
        try:
            if sys.platform.startswith("win"):
                os.startfile(p)
            elif sys.platform == "darwin":
                os.system(f'open "{p}"')
            else:
                os.system(f'xdg-open "{p}"')
        except Exception:
            messagebox.showinfo("Data File", f"Data saved at:\n{p}")

    # Selection/fixed helpers
    def _selected_players(self):
        return [self.player_list.get(i) for i in self.player_list.curselection()]

    def _fixed_names(self):
        return [self.fixed_list.get(i) for i in range(self.fixed_list.size())]

    def _set_fixed_names(self, names):
        self.fixed_list.delete(0, "end")
        for n in names:
            self.fixed_list.insert("end", n)
        self._update_fixed_toggle_label()

    def _selected_all_fixed(self):
        sel = self._selected_players()
        fixed = set(self._fixed_names())
        return len(sel) > 0 and all(n in fixed for n in sel)

    def _update_fixed_toggle_label(self):
        sel = self._selected_players()
        if not sel:
            self.fixed_toggle_btn.config(text="Toggle Fixed", state="disabled")
            return
        self.fixed_toggle_btn.config(state="normal")
        if self._selected_all_fixed():
            self.fixed_toggle_btn.config(text="Remove from Fixed")
        else:
            self.fixed_toggle_btn.config(text="Add to Fixed")

    def on_toggle_fixed(self):
        sel = self._selected_players()
        if not sel:
            return
        fixed = set(self._fixed_names())
        if any(n not in fixed for n in sel):
            # Add all selected to fixed
            self._record_state(f"Add to Fixed: {', '.join(sel)}")
            for n in sel:
                fixed.add(n)
        else:
            # Remove selected from fixed
            self._record_state(f"Remove from Fixed: {', '.join(sel)}")
            for n in sel:
                if n in fixed:
                    fixed.remove(n)
        self._set_fixed_names(sorted(fixed))

    # Player operations
    def on_add_player(self):
        name = self.add_name_var.get().strip()
        if not name:
            messagebox.showinfo("Input", "Enter a player name.")
            return
        self._record_state(f"Add Player '{name}'")
        try:
            self.kb.add_player(name)
            self._save("Player added.")
            self.add_name_var.set("")
            self._refresh_players()
            self._select_player_in_list(name)
        except Exception as e:
            messagebox.showerror("Add Player", str(e))

    def on_remove_selected(self):
        sel_indices = list(self.player_list.curselection())
        if not sel_indices:
            messagebox.showinfo("Remove Player", "Select at least one player to remove.")
            return
        names = [self.player_list.get(i) for i in sel_indices]
        if not messagebox.askyesno("Confirm", f"Remove {len(names)} player(s)?"):
            return
        self._record_state(f"Remove {len(names)} Player(s)")
        for n in names:
            if n in self.kb.players:
                del self.kb.players[n]
        fixed_names = list(self._fixed_names())
        for n in names:
            if n in fixed_names:
                fixed_names.remove(n)
        self._set_fixed_names(fixed_names)
        self._save("Player(s) removed.")
        self._refresh_players()
        self._refresh_selected_player_view()

    # Points operations
    def on_add_point(self):
        player = self.player_combo.get().strip()
        category = self.category_var.get().strip()
        try:
            points = int(self.points_var.get())
        except Exception:
            messagebox.showinfo("Input", "Points must be an integer.")
            return
        if not player:
            messagebox.showinfo("Input", "Choose a player.")
            return
        if not category:
            messagebox.showinfo("Input", "Enter a category.")
            return
        if points <= 0:
            messagebox.showinfo("Input", "Points must be positive.")
            return
        self._record_state(f"Add {points} pt(s) to {player} in '{category}'")
        try:
            self.kb.add_point(player, category, points)
            self._save(f"Added {points} point(s) to {player} in '{category}'.")
            self.category_var.set("")
            self._refresh_selected_player_view()
            self._refresh_weights()
        except Exception as e:
            messagebox.showerror("Add Point", str(e))

    # Compute
    def on_compute(self):
        try:
            team_size = int(self.team_size_var.get())
            top_n = int(self.topn_var.get())
            agg = self.agg_var.get()
        except Exception as e:
            messagebox.showerror("Compute", str(e))
            return

        fixed = self._fixed_names()
        all_players = self.kb.list_players()
        for f in fixed:
            if f not in all_players:
                messagebox.showerror("Fixed Players", f"Fixed player '{f}' no longer exists.")
                return

        if len(fixed) > team_size:
            messagebox.showerror("Team Size", "Team size must be >= number of fixed players.")
            return

        remaining = [p for p in all_players if p not in fixed]

        results = []
        from itertools import combinations
        if len(fixed) == team_size:
            score, breakdown = self.kb.team_score(fixed, aggregator=agg)
            results = [{"team": list(fixed), "score": score, "breakdown": breakdown}]
        else:
            k = team_size - len(fixed)
            for combo in combinations(remaining, k):
                team = list(sorted(set(fixed) | set(combo)))
                score, breakdown = self.kb.team_score(team, aggregator=agg)
                results.append({"team": team, "score": score, "breakdown": breakdown})
            results.sort(key=lambda r: (-r["score"], tuple(r["team"])))
            results = results[:top_n]

        for row in self.results.get_children():
            self.results.delete(row)
        for i, r in enumerate(results, start=1):
            team_str = ", ".join(r["team"])
            self.results.insert("", "end", values=(i, team_str, f"{r['score']:.3f}"), tags=(json.dumps(r),))

        self.status_var.set(f"Computed {len(results)} result(s) (agg={agg}).")
        self.result_breakdown.delete("1.0", "end")
        if results:
            first = self.results.get_children()
            if first:
                self.results.selection_set(first[0])
                self.results.focus(first[0])
                self.on_select_result()

    def on_select_result(self, event=None):
        sel = self.results.selection()
        self.result_breakdown.delete("1.0", "end")
        if not sel:
            return
        item = sel[0]
        tags = self.results.item(item, "tags")
        if not tags:
            return
        try:
            r = json.loads(tags[0])
        except Exception:
            return
        bd = r.get("breakdown", {})
        lines = [f"Team: {', '.join(r.get('team', []))}",
                 f"Score: {r.get('score', 0):.3f}",
                 "-" * 24]
        for c in sorted(bd.keys()):
            lines.append(f"{c}: {bd[c]:.3f}")
        self.result_breakdown.insert("1.0", "\n".join(lines))

    # Weights
    def on_set_weight(self):
        cat = self.weight_cat_var.get().strip()
        try:
            w = float(self.weight_val_var.get())
        except Exception:
            messagebox.showinfo("Weights", "Weight must be a number > 0.")
            return
        if not cat:
            messagebox.showinfo("Weights", "Enter a category name.")
            return
        if w <= 0:
            messagebox.showinfo("Weights", "Weight must be > 0.")
            return
        self._record_state(f"Set weight {cat} -> {w:g}")
        try:
            self.kb.set_weight(cat, w)
            self._save(f"Weight set: {cat} -> {w}")
            self._refresh_weights()
        except Exception as e:
            messagebox.showerror("Weights", str(e))

    def on_remove_weight(self):
        sel = self.weights_tree.selection()
        if not sel:
            messagebox.showinfo("Weights", "Select a weight to remove.")
            return
        item = sel[0]
        values = self.weights_tree.item(item, "values")
        if not values:
            return
        cat = values[0]
        self._record_state(f"Remove weight '{cat}'")
        if cat in self.kb.category_weights:
            del self.kb.category_weights[cat]
            self._save(f"Removed weight for '{cat}'.")
            self._refresh_weights()

    # Refresh helpers
    def _refresh_players(self):
        self.player_list.delete(0, "end")
        names = self.kb.list_players()
        for n in names:
            self.player_list.insert("end", n)

        self.player_combo["values"] = names
        if names and not self.player_combo.get():
            self.player_combo.set(names[0])

        fixed = [n for n in self._fixed_names() if n in names]
        self._set_fixed_names(sorted(set(fixed)))

    def _select_player_in_list(self, name: str):
        names = self.kb.list_players()
        try:
            idx = names.index(name)
            self.player_list.selection_clear(0, "end")
            self.player_list.selection_set(idx)
            self.player_list.activate(idx)
            self.player_list.see(idx)
            self.player_combo.set(name)
            self._refresh_selected_player_view()
            self._update_fixed_toggle_label()
        except ValueError:
            pass

    def _refresh_selected_player_view(self):
        for row in self.breakdown.get_children():
            self.breakdown.delete(row)

        sel = self.player_list.curselection()
        name = None
        if sel:
            name = self.player_list.get(sel[0])
        elif self.kb.list_players():
            name = self.kb.list_players()[0]

        if name and name in self.kb.players:
            p = self.kb.players[name]
            items = sorted(p.categories.items(), key=lambda kv: (-kv[1], kv[0]))
            for c, v in items:
                self.breakdown.insert("", "end", values=(c, v))

            if self.player_combo.get() != name:
                self.player_combo.set(name)

    def _refresh_weights(self):
        for row in self.weights_tree.get_children():
            self.weights_tree.delete(row)
        cats = set(self.kb.list_categories())
        for c in sorted(cats):
            w = self.kb.category_weights.get(c, 1.0)
            self.weights_tree.insert("", "end", values=(c, f"{w:g}"))

    def _save(self, status_msg):
        save_kb(self.kb)
        self.status_var.set(f"Saved. {status_msg}")

if __name__ == "__main__":
    app = KBApp()
    app.mainloop()
