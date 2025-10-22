# kaz_launcher.py
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

# ---- Try multiple import paths to handle PettingZoo version differences ----
ENV_FACTORY = None
IMPORT_ERRORS = []

def _try_imports():
    global ENV_FACTORY
    candidates = [
        # common modern path
        ("from pettingzoo.butterfly import knights_archers_zombies_v10 as kaz", "kaz"),
        # fallbacks for older versions
        ("from pettingzoo.butterfly import knights_archers_zombies_v9 as kaz",  "kaz"),
        ("from pettingzoo.butterfly import knights_archers_zombies_v8 as kaz",  "kaz"),
        ("from pettingzoo.butterfly import knights_archers_zombies_v7 as kaz",  "kaz"),
    ]
    for stmt, name in candidates:
        try:
            local = {}
            exec(stmt, {}, local)
            kaz = local[name]
            # choose human/ansi render as supported
            def factory(render_mode="human"):
                try:
                    return kaz.env(render_mode=render_mode)
                except TypeError:
                    # Some older builds use .raw_env or only support "human"
                    try:
                        return kaz.raw_env()
                    except Exception as e:
                        raise e
            ENV_FACTORY = factory
            return
        except Exception as e:
            IMPORT_ERRORS.append(f"{stmt} -> {type(e).__name__}: {e}")

_try_imports()

# ------------------------ Simple controllers (policies) ------------------------

class BaseController:
    name = "Base"
    def __call__(self, obs, action_space, agent, t):
        raise NotImplementedError

class RandomController(BaseController):
    name = "Random"
    def __call__(self, obs, action_space, agent, t):
        return action_space.sample()

class GreedyOneStepController(BaseController):
    """
    A very lightweight 'searchy' baseline:
    - Samples K candidate actions (including 'do-nothing' if available),
      asks the env for the last observed reward (proxy), and picks
      the action that historically tends to give higher immediate reward.
    NOTE: This does NOT forward-simulate the real env (no cloning). It’s a
    placeholder for plugging in a real model-based one-step lookahead.
    """
    name = "Greedy (one-step heuristic)"
    def __init__(self, K=8, seed=0):
        self.rng = np.random.default_rng(seed)
        self.K = K
        # tiny running score table per agent per action index (if Discrete)
        self.action_stats = {}

    def __call__(self, obs, action_space, agent, t):
        if not hasattr(action_space, "n"):
            # Non-discrete? fall back to random.
            return action_space.sample()

        n = action_space.n
        # lazily init
        table = self.action_stats.setdefault(agent, np.zeros(n, dtype=float))

        # Construct a candidate set (uniform + a bias toward high-score actions)
        top_k = np.argsort(-table)[: min(3, n)]
        candidates = set(top_k.tolist())
        while len(candidates) < min(self.K, n):
            candidates.add(int(self.rng.integers(0, n)))
        candidates = list(candidates)

        # Pick the currently best-scoring action
        best = int(candidates[np.argmax(table[candidates])])
        return best

    def update_scores(self, agent, action, reward):
        if isinstance(action, int):  # Discrete assumption
            self.action_stats.setdefault(agent, None)
            if self.action_stats[agent] is None:
                return
            self.action_stats[agent][action] += float(reward)

class DepthFirstStubController(BaseController):
    """
    DFS placeholder: without a forward model (env cloning) and compact state,
    we can’t do true DFS on KAZ. This stub just behaves randomly but exists so
    you can swap in your own DFS that advances a copied env for d steps.
    """
    name = "DFS (stub)"
    def __call__(self, obs, action_space, agent, t):
        return action_space.sample()

class BreadthFirstStubController(BaseController):
    name = "BFS (stub)"
    def __call__(self, obs, action_space, agent, t):
        return action_space.sample()

class AStarStubController(BaseController):
    """
    A* placeholder: you’d define h(state) (e.g., distance-to-target) and do
    graph search on a compact grid/state. KAZ observations are typically
    image-like; for a true A*, you’d need an internal map/state extractor or
    a model-based simulator to evaluate successors.
    """
    name = "A* (stub)"
    def __call__(self, obs, action_space, agent, t):
        return action_space.sample()

# Registry to populate the dropdown
CONTROLLERS = [
    RandomController(),
    GreedyOneStepController(K=8, seed=42),
    DepthFirstStubController(),
    BreadthFirstStubController(),
    AStarStubController(),
]

# -------------------------- Runner (env loop) ----------------------------------

def run_kaz(selected_controller_name: str, status_cb=print):
    if ENV_FACTORY is None:
        err = "\n".join(IMPORT_ERRORS) or "Unknown import error."
        message = (
            "Could not import PettingZoo Knights-Archers-Zombies.\n\n"
            "Tried several versions. Details:\n" + err
        )
        status_cb(message)
        messagebox.showerror("Import error", message)
        return

    # Pick controller
    controller = next((c for c in CONTROLLERS if c.name == selected_controller_name), CONTROLLERS[0])

    # Try to prefer a visible renderer
    try_modes = ["human", "ansi"]
    env = None
    last_mode_err = None
    for mode in try_modes:
        try:
            env = ENV_FACTORY(render_mode=mode)
            break
        except Exception as e:
            last_mode_err = e
    if env is None:
        status_cb(f"Render creation failed: {last_mode_err}")
        messagebox.showerror("Render error", f"Failed to create env: {last_mode_err}")
        return

    status_cb(f"Launching KAZ with controller: {controller.name}")
    try:
        env.reset(seed=123)
    except TypeError:
        env.reset()

    t = 0
    try:
        # PettingZoo agent iteration API
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last(observe=True)
            # choose action
            if termination or truncation:
                action = None  # no-op (PettingZoo convention)
            else:
                action_space = env.action_space(agent)
                action = controller(obs, action_space, agent, t)

            # step
            env.step(action)

            # update greedy table if using that controller
            if isinstance(controller, GreedyOneStepController):
                if action is not None:
                    controller.update_scores(agent, action, reward)

            # slight delay so the window is viewable / not blazing fast
            t += 1
            # If render_mode="human", the env drives its own window updates.
            # If "ansi", we could print frames, but we'll keep it minimal.
            if t % 25 == 0:
                status_cb(f"[t={t}] last={agent} r={reward:.2f}")

        status_cb("Episode finished.")
    except Exception as e:
        status_cb(f"Run error: {type(e).__name__}: {e}")
    finally:
        try:
            env.close()
        except Exception:
            pass

# ------------------------------ Tkinter UI -------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # --- DPI / scaling (nice on high-DPI displays) ---
        try:
            # 1.25 is a good starting point; tweak if needed
            self.tk.call("tk", "scaling", 1.25)
        except Exception:
            pass

        self.title("KAZ Demo Launcher")
        self.configure(bg="#0f1115")  # dark background

        # ---- Full-screen behaviors ----
        self._fullscreen = True
        # On Windows you could also do: self.state("zoomed")
        self.attributes("-fullscreen", True)
        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", self._exit_fullscreen)

        # ---- TTK theming ----
        style = ttk.Style()
        # Use a built-in theme then override colors
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TFrame", background="#0f1115")
        style.configure("TLabel", background="#0f1115", foreground="#e5e7eb", font=("Segoe UI", 12))
        style.configure("Heading.TLabel", font=("Segoe UI", 28, "bold"), foreground="#f3f4f6")
        style.configure("Subheading.TLabel", font=("Segoe UI", 13), foreground="#cbd5e1")
        style.configure("TButton", font=("Segoe UI", 12), padding=10)
        style.map("TButton", foreground=[("disabled", "#9ca3af")])
        style.configure("Play.TButton", font=("Segoe UI", 14, "bold"))

        # ---- Root grid: 3 rows (top spacer, center content, bottom bar) ----
        self.grid_rowconfigure(0, weight=1)  # spacer
        self.grid_rowconfigure(1, weight=2)  # center content
        self.grid_rowconfigure(2, weight=1)  # bottom bar (fixed height-ish)
        self.grid_columnconfigure(0, weight=1)

        # ---- Center panel (hero) ----
        center = ttk.Frame(self)
        center.grid(row=1, column=0, sticky="nsew")
        center.grid_columnconfigure(0, weight=1)
        for r in range(4):
            center.grid_rowconfigure(r, weight=1)

        title = ttk.Label(center, text="Knights–Archers–Zombies", style="Heading.TLabel")
        title.grid(row=0, column=0, sticky="s", pady=(0, 6))

        subtitle = ttk.Label(
            center,
            text="Pick a search controller and launch the interactive demo.",
            style="Subheading.TLabel"
        )
        subtitle.grid(row=1, column=0, sticky="n")

        # Selection area
        select_frame = ttk.Frame(center)
        select_frame.grid(row=2, column=0)
        select_frame.grid_columnconfigure(0, weight=1)

        ttk.Label(select_frame, text="Search method / controller:").grid(row=0, column=0, sticky="w", pady=(12, 6))
        self.controller_var = tk.StringVar(value=CONTROLLERS[0].name)
        self.controller_cb = ttk.Combobox(
            select_frame,
            textvariable=self.controller_var,
            values=[c.name for c in CONTROLLERS],
            state="readonly",
            width=34
        )
        self.controller_cb.grid(row=1, column=0, sticky="ew")

        # Small info log (optional)
        self.status = tk.Text(center, height=6, width=80, bg="#111317", fg="#e5e7eb", bd=0, highlightthickness=0)
        self.status.grid(row=3, column=0, pady=(20, 0))
        self.status.configure(state="disabled")

        if ENV_FACTORY is None:
            self._log("⚠️ Could not import KAZ yet. Try `pip install pettingzoo pygame`.\n"
                      "See errors in console if it still fails.")

        # ---- Bottom bar with Play button ----
        bottom = ttk.Frame(self)
        bottom.grid(row=2, column=0, sticky="nsew", padx=24, pady=24)
        bottom.grid_columnconfigure(0, weight=1)

        self.play_btn = ttk.Button(bottom, text="Play", style="Play.TButton", command=self.on_play)
        # keep it centered horizontally
        self.play_btn.grid(row=0, column=0)

    # -------------------- Helpers --------------------
    def _toggle_fullscreen(self, _evt=None):
        self._fullscreen = not self._fullscreen
        self.attributes("-fullscreen", self._fullscreen)

    def _exit_fullscreen(self, _evt=None):
        if self._fullscreen:
            self._fullscreen = False
            self.attributes("-fullscreen", False)

    def _log(self, msg: str):
        self.status.configure(state="normal")
        self.status.insert("end", msg + "\n")
        self.status.see("end")
        self.status.configure(state="disabled")

    # -------------------- Actions --------------------
    def on_play(self):
        name = self.controller_var.get()
        self.play_btn.configure(state="disabled")

        def _run():
            try:
                self._log(f"Starting with controller: {name}")
                run_kaz(name, status_cb=self._log)
            finally:
                self.play_btn.configure(state="normal")

        threading.Thread(target=_run, daemon=True).start()


if __name__ == "__main__":
    app = App()
    app.mainloop()