 #Intelligent Urban Delivery Robot
import tkinter as tk
from tkinter import ttk, messagebox
import random, time, math, heapq
from collections import deque
#  GLOBAL CONSTANTS
GRID_SIZE      = 15          # rows and columns
CELL_PX        = 30        # pixels per cell
NUM_DELIVERIES = 5
STEP_DELAY_MS  = 100         # animation speed (ms per step)
BASE           = (0, 0)      # fixed base-station position
# Cell types
OBSTACLE = "obstacle"
ROAD     = "road"
TRAFFIC  = "traffic"
DELIVERY = "delivery"
# Cell colours
COLORS = {
    OBSTACLE : "#1a1a2e",   # dark / black
    ROAD     : "#f5f5f5",   # white / light grey
    TRAFFIC  : "#FFC0CB",   # orange
    DELIVERY : "#4caf50",   # green
    "base"   : "#9c27b0",   # purple
    "robot"  : "#2196f3",   # blue
    "path"   : "#ef5350",   # red
    "done"   : "#000000",   # black  – completed delivery
}
# Four cardinal movement directions
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

#  GRID CONSTRUCTION
def build_grid():
    cell_type = [[ROAD] * GRID_SIZE for _ in range(GRID_SIZE)]
    cost_map  = [[1]   * GRID_SIZE for _ in range(GRID_SIZE)]

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if (r, c) == BASE:          # base station is always open
                continue
            roll = random.random()
            if roll < 0.15:             # obstacle
                cell_type[r][c] = OBSTACLE
                cost_map [r][c] = 9999
            elif roll < 0.30:           # traffic zone
                cell_type[r][c] = TRAFFIC
                cost_map [r][c] = random.randint(10, 20)
            else:                       # normal road
                cell_type[r][c] = ROAD
                cost_map [r][c] = random.randint(1, 5)

    return cell_type, cost_map


def place_deliveries(cell_type):
    #Randomly choose NUM_DELIVERIES non-obstacle, non-base cells.
    chosen = []
    while len(chosen) < NUM_DELIVERIES:
        r = random.randint(0, GRID_SIZE - 1)
        c = random.randint(0, GRID_SIZE - 1)
        if cell_type[r][c] != OBSTACLE and (r, c) != BASE and (r, c) not in chosen:
            cell_type[r][c] = DELIVERY
            chosen.append((r, c))
    return chosen


def cell_reachable(cell_type, pos):
    #Return True when pos is inside the grid and not an obstacle.
    r, c = pos
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and cell_type[r][c] != OBSTACLE

#  HEURISTIC FUNCTIONS
def manhattan(a, b):
    #Sum of absolute row/col differences.
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    #Straight-line distance.
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

#  SEARCH ALGORITHMS
def bfs(cell_type, cost_map, start, goal, **_):
    """Breadth-First Search — optimal step count, ignores weights."""
    queue   = deque([(start, [start])])
    visited = {start}
    nodes   = 0
    while queue:
        cur, path = queue.popleft()
        nodes += 1
        if cur == goal:
            return path, nodes, sum(cost_map[r][c] for r, c in path[1:])
        for dr, dc in DIRS:
            nb = (cur[0]+dr, cur[1]+dc)
            if cell_reachable(cell_type, nb) and nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    return [], nodes, 0


def dfs(cell_type, cost_map, start, goal, **_):
    """Depth-First Search — explores deep first; not guaranteed optimal."""
    stack   = [(start, [start])]
    visited = set()
    nodes   = 0
    while stack:
        cur, path = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        nodes += 1
        if cur == goal:
            return path, nodes, sum(cost_map[r][c] for r, c in path[1:])
        for dr, dc in DIRS:
            nb = (cur[0]+dr, cur[1]+dc)
            if cell_reachable(cell_type, nb) and nb not in visited:
                stack.append((nb, path + [nb]))
    return [], nodes, 0


def ucs(cell_type, cost_map, start, goal, **_):
    """Uniform Cost Search — always expands the cheapest node; optimal."""
    heap  = [(0, start, [start])]   # (cost, pos, path)
    best  = {}
    nodes = 0
    while heap:
        cost, cur, path = heapq.heappop(heap)
        nodes += 1
        if cur == goal:
            return path, nodes, cost
        if cur in best and best[cur] <= cost:
            continue
        best[cur] = cost
        for dr, dc in DIRS:
            nb = (cur[0]+dr, cur[1]+dc)
            if cell_reachable(cell_type, nb):
                nc = cost + cost_map[nb[0]][nb[1]]
                if nb not in best or best[nb] > nc:
                    heapq.heappush(heap, (nc, nb, path + [nb]))
    return [], nodes, 0


def greedy(cell_type, cost_map, start, goal, heuristic=manhattan):
    """Greedy Best-First — guided purely by heuristic; fast but not optimal."""
    heap    = [(heuristic(start, goal), start, [start])]
    visited = set()
    nodes   = 0
    while heap:
        _, cur, path = heapq.heappop(heap)
        if cur in visited:
            continue
        visited.add(cur)
        nodes += 1
        if cur == goal:
            return path, nodes, sum(cost_map[r][c] for r, c in path[1:])
        for dr, dc in DIRS:
            nb = (cur[0]+dr, cur[1]+dc)
            if cell_reachable(cell_type, nb) and nb not in visited:
                heapq.heappush(heap, (heuristic(nb, goal), nb, path + [nb]))
    return [], nodes, 0


def astar(cell_type, cost_map, start, goal, heuristic=manhattan):
    """A* Search — combines actual cost g(n) with heuristic h(n); optimal."""
    heap  = [(heuristic(start, goal), 0, start, [start])]  # (f, g, pos, path)
    best  = {}
    nodes = 0
    while heap:
        f, g, cur, path = heapq.heappop(heap)
        nodes += 1
        if cur == goal:
            return path, nodes, g
        if cur in best and best[cur] <= g:
            continue
        best[cur] = g
        for dr, dc in DIRS:
            nb = (cur[0]+dr, cur[1]+dc)
            if cell_reachable(cell_type, nb):
                ng = g + cost_map[nb[0]][nb[1]]
                if nb not in best or best[nb] > ng:
                    heapq.heappush(heap, (ng + heuristic(nb, goal), ng, nb, path + [nb]))
    return [], nodes, 0


# Algorithm registry
ALGORITHMS = {
    "BFS"                : bfs,
    "DFS"                : dfs,
    "UCS"                : ucs,
    "Greedy Best First"  : greedy,
    "A* Search"          : astar,
}

HEURISTICS = {
    "Manhattan" : manhattan,
    "Euclidean" : euclidean,
}

#  MAIN APPLICATION
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Intelligent Urban Delivery Robot")
        self.root.configure(bg="#1e2a35")
        self.root.resizable(False, False)

        # Runtime state 
        self.cell_type    = []        
        self.cost_map     = []        
        self.deliveries   = []        # ordered list of delivery positions
        self.robot_pos    = BASE      # current robot position
        self.deliver_idx  = 0         
        self.running      = False     # True while animation is active
        self.rect_ids     = {}        
        self.text_ids     = {}        

        self._build_ui()
        self._new_grid()            

    #  UI CONSTRUCTION
    def _build_ui(self):
        # ── Top header bar ──
        header = tk.Frame(self.root, bg="#0d1b2a", pady=10)
        header.pack(fill="x")
        tk.Label(header,
                 text=" Intelligent Urban Delivery Robot",
                 font=("Helvetica", 16, "bold"),
                 bg="#0d1b2a", fg="#64b5f6").pack()

        # ── Main body: canvas (left) + side panel (right) ──
        body = tk.Frame(self.root, bg="#1e2a35")
        body.pack(padx=10, pady=8)

        # Grid canvas
        canvas_size = GRID_SIZE * CELL_PX + 2
        self.canvas = tk.Canvas(body,
                                width=canvas_size, height=canvas_size,
                                bg="#263238", highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=(0, 12))

        # Side panel
        side = tk.Frame(body, bg="#263238", padx=12, pady=10,
                        relief="flat", bd=0, width=220)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)
        self._build_side_panel(side)

        # ── Bottom legend ──
        legend_frame = tk.Frame(self.root, bg="#1e2a35", pady=6)
        legend_frame.pack()
        items = [
            ("#1a1a2e", "Obstacle"),
            ("#f5f5f5", "Road"),
            ("#ff9800", "Traffic"),
            ("#4caf50", "Delivery"),
            ("#9c27b0", "Base"),
            ("#2196f3", "Robot"),
            ("#ef5350", "Path"),
        ]
        for color, label in items:
            dot = tk.Frame(legend_frame, bg=color, width=13, height=13,
                           relief="solid", bd=1)
            dot.pack(side="left", padx=(8, 2))
            tk.Label(legend_frame, text=label,
                     bg="#1e2a35", fg="#cfd8dc",
                     font=("Helvetica", 8)).pack(side="left", padx=(0, 5))

    def _build_side_panel(self, parent):
        #Build controls and delivery list inside the side panel.
        def section(text):
            tk.Label(parent, text=text,
                     font=("Helvetica", 9, "bold"),
                     bg="#263238", fg="#90a4ae").pack(anchor="w", pady=(10, 2))

        # ── Algorithm selection ───
        section("Algorithm")
        self.algo_var = tk.StringVar(value="A* Search")
        algo_cb = ttk.Combobox(parent, textvariable=self.algo_var,
                               values=list(ALGORITHMS.keys()),
                               state="readonly", width=22)
        algo_cb.pack(anchor="w")

        # ── Heuristic selection ───
        section("Heuristic  (Greedy / A*)")
        self.heur_var = tk.StringVar(value="Manhattan")
        heur_cb = ttk.Combobox(parent, textvariable=self.heur_var,
                               values=list(HEURISTICS.keys()),
                               state="readonly", width=22)
        heur_cb.pack(anchor="w")

        # ── Buttons ──
        btn_kw = dict(font=("Helvetica", 10, "bold"), relief="flat",
                      cursor="hand2", padx=6, pady=5, width=20)
        tk.Button(parent, text="▶  Start Simulation",
                  bg="#43a047", fg="white",
                  command=self._start_simulation,
                  **btn_kw).pack(pady=(14, 4))
        tk.Button(parent, text="⟳  New Grid",
                  bg="#1976d2", fg="white",
                  command=self._new_grid,
                  **btn_kw).pack(pady=2)

        # ── Status label ──
        section("Status")
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(parent, textvariable=self.status_var,
                 bg="#263238", fg="#80cbc4",
                 font=("Helvetica", 9, "italic"),
                 wraplength=200, justify="left").pack(anchor="w")

        # ── Metrics ───
        section("Last Delivery Metrics")
        self.m_cost  = tk.StringVar(value="Cost    : —")
        self.m_nodes = tk.StringVar(value="Nodes   : —")
        self.m_time  = tk.StringVar(value="Time    : —")
        for v in (self.m_cost, self.m_nodes, self.m_time):
            tk.Label(parent, textvariable=v,
                     bg="#263238", fg="#ffcc80",
                     font=("Courier", 9)).pack(anchor="w")

        # ── Delivery checklist ───
        section("Delivery Queue")
        self.delivery_labels = []
        for i in range(NUM_DELIVERIES):
            lv = tk.StringVar(value=f"  D{i+1}  ·  —")
            lbl = tk.Label(parent, textvariable=lv,
                           bg="#263238", fg="#b0bec5",
                           font=("Courier", 9), anchor="w", width=22)
            lbl.pack(anchor="w")
            self.delivery_labels.append((lv, lbl))


    #  GRID MANAGEMENT
    def _new_grid(self):
        if self.running:
            return
        self.cell_type, self.cost_map = build_grid()
        self.deliveries  = place_deliveries(self.cell_type)
        self.robot_pos   = BASE
        self.deliver_idx = 0
        self._reset_metrics()
        self._update_delivery_list()
        self.status_var.set("New grid ready.")
        self._draw_full_grid()

    def _draw_full_grid(self):
        #Render every cell from scratch.
        self.canvas.delete("all")
        self.rect_ids.clear()
        self.text_ids.clear()

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self._draw_cell(r, c)

        # Base station label
        self._set_cell_color(BASE, COLORS["base"])
        self._set_cell_label(BASE, "B", "white")

        # Robot marker
        self._set_cell_color(self.robot_pos, COLORS["robot"])
        self._set_cell_label(self.robot_pos, "R", "white")

    def _draw_cell(self, r, c):
        ctype = self.cell_type[r][c]
        color = COLORS[ctype]

        x1 = c * CELL_PX + 1
        y1 = r * CELL_PX + 1
        x2 = x1 + CELL_PX - 2
        y2 = y1 + CELL_PX - 2
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        rect = self.canvas.create_rectangle(x1, y1, x2, y2,
                                            fill=color,
                                            outline="#37474f", width=1)
        self.rect_ids[(r, c)] = rect

        # Default label text
        if ctype == OBSTACLE:
            label_text  = "X"
            label_color = "#555577"
        elif ctype == DELIVERY:
            idx = self.deliveries.index((r, c)) + 1
            label_text  = f"D{idx}"
            label_color = "white"
        elif ctype == TRAFFIC:
            label_text  = str(self.cost_map[r][c])
            label_color = "#4a3000"
        else:
            label_text  = str(self.cost_map[r][c])
            label_color = "#aaaaaa"

        tid = self.canvas.create_text(cx, cy,
                                      text=label_text,
                                      font=("Helvetica", 8, "bold"),
                                      fill=label_color)
        self.text_ids[(r, c)] = tid

    def _set_cell_color(self, pos, color):
        #Change a single cell's background colour.
        rid = self.rect_ids.get(pos)
        if rid:
            self.canvas.itemconfig(rid, fill=color)

    def _set_cell_label(self, pos, text, fg="white"):
        #Change a single cell's text label.
        tid = self.text_ids.get(pos)
        if tid:
            self.canvas.itemconfig(tid, text=text, fill=fg)

    def _restore_cell(self, pos):
        #Reset a cell to its natural colour and label (after robot leaves).
        r, c   = pos
        ctype  = self.cell_type[r][c]

        if pos == BASE:
            self._set_cell_color(pos, COLORS["base"])
            self._set_cell_label(pos, "B", "white")
            return

        color = COLORS[ctype]
        self._set_cell_color(pos, color)

        if ctype == DELIVERY:
            idx  = self.deliveries.index(pos) + 1
            self._set_cell_label(pos, f"D{idx}", "white")
        elif ctype == OBSTACLE:
            self._set_cell_label(pos, "X", "#555577")
        else:
            cost_str = str(self.cost_map[r][c])
            fg       = "#4a3000" if ctype == TRAFFIC else "#aaaaaa"
            self._set_cell_label(pos, cost_str, fg)

    #  SIDE PANEL HELPERS
    def _reset_metrics(self):
        self.m_cost .set("Cost    : —")
        self.m_nodes.set("Nodes   : —")
        self.m_time .set("Time    : —")

    def _update_metrics(self, cost, nodes, elapsed):
        self.m_cost .set(f"Cost    : {cost}")
        self.m_nodes.set(f"Nodes   : {nodes}")
        self.m_time .set(f"Time    : {elapsed:.5f}s")

    def _update_delivery_list(self):
        #Refresh the D1–D5 checklist in the side panel.
        for i, pos in enumerate(self.deliveries):
            lv, lbl = self.delivery_labels[i]
            if i < self.deliver_idx:
                lv.set(f"  D{i+1}  ✓  {pos}")
                lbl.config(fg="#4caf50")          # green = done
            elif i == self.deliver_idx:
                lv.set(f"► D{i+1}  →  {pos}")
                lbl.config(fg="#ffcc80")          # yellow = current
            else:
                lv.set(f"  D{i+1}  ·  {pos}")
                lbl.config(fg="#b0bec5")          # grey  = pending


    #  SIMULATION CONTROL
    def _start_simulation(self):
        #Button handler — begins or resumes the delivery run.
        if self.running:
            messagebox.showinfo("Busy", "Simulation already running!")
            return
        if self.deliver_idx >= NUM_DELIVERIES:
            self._new_grid()            # auto-reset after full run
            return
        self.running = True
        self._next_delivery()

    def _next_delivery(self):
        #Called to start each individual delivery leg.
        if self.deliver_idx >= NUM_DELIVERIES:
            self.running = False
            self.status_var.set(" All deliveries complete!")
            self._update_delivery_list()
            print("\n" + "="*52)
            print("  ALL DELIVERIES COMPLETE")
            print("="*52)
            return

        target    = self.deliveries[self.deliver_idx]
        algo_name = self.algo_var.get()
        heur_name = self.heur_var.get()
        heuristic = HEURISTICS[heur_name]
        algo_fn   = ALGORITHMS[algo_name]

        self.status_var.set(
            f"Delivery {self.deliver_idx+1}/{NUM_DELIVERIES}\n"
            f"{self.robot_pos} → {target}\n"
            f"[{algo_name}]"
        )
        self._update_delivery_list()
        self.root.update()

        # ── Run the algorithm ───
        t0 = time.perf_counter()
        if algo_name in ("Greedy Best First", "A* Search"):
            path, nodes, cost = algo_fn(self.cell_type, self.cost_map,
                                        self.robot_pos, target,
                                        heuristic=heuristic)
        else:
            path, nodes, cost = algo_fn(self.cell_type, self.cost_map,
                                        self.robot_pos, target)
        elapsed = time.perf_counter() - t0

        # ── Update metrics ───
        self._update_metrics(cost, nodes, elapsed)

        # ── Console report ──
        print(f"\n{'─'*52}")
        print(f" Delivery {self.deliver_idx+1}  |  {self.robot_pos} → {target}")
        print(f"  Algorithm : {algo_name}  |  Heuristic: {heur_name}")
        print(f"  Path steps: {len(path)}   Cost: {cost}")
        print(f"  Nodes expl: {nodes}   Time: {elapsed:.6f}s")

        if not path:
            messagebox.showwarning(
                "No Path Found",
                f"Could not reach Delivery {self.deliver_idx+1} at {target}.\nSkipping."
            )
            print("  ⚠  No path — skipping.")
            self.deliver_idx += 1
            self._next_delivery()
            return

        # ── Colour the planned path in red ───
        for step in path[1:-1]:
            self._set_cell_color(step, COLORS["path"])
        self.root.update()

        # ── Animate step by step ──
        self._animate(path, step_index=0)

    def _animate(self, path, step_index):
        if step_index >= len(path):
            self._delivery_done(path[-1])
            return

        cur = path[step_index]

        # Restore previous cell's appearance
        if step_index > 0:
            self._restore_cell(path[step_index - 1])

        # Draw robot at new position
        self._set_cell_color(cur, COLORS["robot"])
        self._set_cell_label(cur, "R", "white")
        self.robot_pos = cur
        self.root.update()

        self.root.after(STEP_DELAY_MS, self._animate, path, step_index + 1)

    def _delivery_done(self, pos):
        #Mark a delivery as complete and move to the next.
        # Colour the delivered cell as "done"
        self._set_cell_color(pos, COLORS["done"])
        self._set_cell_label(pos, "✓", "white")

        # Update grid so cell is now traversable as a normal road
        self.cell_type[pos[0]][pos[1]] = ROAD

        print(f"  ✓ Delivery {self.deliver_idx+1} complete at {pos}")

        self.deliver_idx += 1
        self._update_delivery_list()

        # Short pause before starting the next leg
        self.root.after(350, self._next_delivery)

#  ENTRY POINT
if __name__ == "__main__":
    root = tk.Tk()
    app  = App(root)
    root.mainloop()
