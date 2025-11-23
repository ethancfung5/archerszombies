# kaz_launcher.py
import sys
import os
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

# local imports from controllers package
from controllers import CONTROLLERS, get_controller_by_name

# ----------------------- PettingZoo import handling ----------------------------

ENV_FACTORY = None
ENV_LABEL = None
IMPORT_ERRORS = []


def _make_factory(kaz_module):
    """Return an env factory preferring env(render_mode=...), with safe fallbacks."""
    def factory(render_mode="human"):
        if hasattr(kaz_module, "env"):
            try:
                return kaz_module.env(render_mode=render_mode)
            except TypeError:
                # older builds may not accept render_mode kwarg
                return kaz_module.env()
        if hasattr(kaz_module, "raw_env"):
            return kaz_module.raw_env()
        if hasattr(kaz_module, "parallel_env"):
            return kaz_module.parallel_env()
        raise RuntimeError("No env/raw_env/parallel_env on KAZ module")
    return factory


def _try_imports():
    """
    Prefer v10; try butterfly first (current home), then sisl.
    As a last resort, allow v9 (deprecated) so the app can still run
    but we'll warn loudly in the UI.
    """
    global ENV_FACTORY, ENV_LABEL

    candidates = [
        # Preferred modern locations
        ("from pettingzoo.butterfly import knights_archers_zombies_v10 as kaz", "KAZ v10 (butterfly)"),
        ("from pettingzoo.sisl import knights_archers_zombies_v10 as kaz",      "KAZ v10 (sisl)"),
        # Last-resort deprecated fallback (shows warning in UI)
        ("from pettingzoo.butterfly import knights_archers_zombies_v9 as kaz",  "KAZ v9 (deprecated)"),
    ]

    for stmt, label in candidates:
        try:
            ns = {}
            exec(stmt, {}, ns)
            kaz = ns["kaz"]
            ENV_FACTORY = _make_factory(kaz)
            ENV_LABEL = label
            return
        except Exception as e:
            IMPORT_ERRORS.append(f"{stmt} -> {type(e).__name__}: {e}")

    ENV_FACTORY = None
    ENV_LABEL = None


_try_imports()


# ------------------------------ Game runner ------------------------------------

def run_kaz(selected_controller_name: str, status_cb=print):
    """
    Run a single KAZ episode using the PettingZoo AEC API.
    NOTE: This runs in a separate process (spawned by the Tk launcher).
    """
    import time

    if ENV_FACTORY is None:
        err = "\n".join(IMPORT_ERRORS) or "Unknown import error."
        msg = (
            "Could not import Knights–Archers–Zombies v10.\n"
            "Install in your venv:\n"
            "  pip install 'pettingzoo[butterfly]>=1.24' 'gymnasium==0.29.1' pygame numpy\n\n"
            "Details:\n" + err
        )
        status_cb(msg)
        return

    # Let users know if the fallback hit v9
    if "deprecated" in (ENV_LABEL or "").lower():
        status_cb("⚠️ Using deprecated KAZ v9. Please upgrade to v10:\n"
                  "   pip install --upgrade 'pettingzoo[butterfly]>=1.24'")

    controller = get_controller_by_name(selected_controller_name)
    status_cb(f"Using env: {ENV_LABEL}")
    status_cb(f"Launching KAZ with controller: {controller.name}")

    # Prefer visible window; fallback to headless construction if needed
    env = None
    last_err = None
    for mode in ("human", None):
        try:
            env = ENV_FACTORY(render_mode=mode)
            break
        except Exception as e:
            last_err = e
    if env is None:
        status_cb(f"Render creation failed: {last_err}")
        return

    try:
        try:
            env.reset(seed=123)
        except TypeError:
            env.reset()

        t = 0
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last(observe=True)

            if termination or truncation:
                action = None  # PettingZoo convention when done
            else:
                action_space = env.action_space(agent)
                action = controller(obs, action_space, agent, t)

            env.step(action)

            if action is not None and hasattr(controller, "update_scores"):
                try:
                    controller.update_scores(agent, action, reward)
                except Exception:
                    pass

            t += 1
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
    """
    Full-screen Tk launcher. On Play, start a **separate Python process**
    that runs `run_kaz(controller_name)`. This avoids Tk + SDL/pygame collisions
    on macOS and is robust on Windows.
    """
    def __init__(self):
        super().__init__()

        # DPI / scaling
        try:
            self.tk.call("tk", "scaling", 1.25)
        except Exception:
            pass

        self.title("KAZ Demo Launcher")
        self.configure(bg="#0f1115")

        # Full-screen behaviors
        self._fullscreen = True
        self.attributes("-fullscreen", True)
        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", self._exit_fullscreen)

        # Theming
        style = ttk.Style()
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

        # Root grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=2)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Center panel
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
            style="Subheading.TLabel",
        )
        subtitle.grid(row=1, column=0, sticky="n")

        # Selection area
        select_frame = ttk.Frame(center)
        select_frame.grid(row=2, column=0)
        select_frame.grid_columnconfigure(0, weight=1)

        ttk.Label(select_frame, text="Search method / controller:").grid(
            row=0, column=0, sticky="w", pady=(12, 6)
        )

        self.controller_var = tk.StringVar(value=CONTROLLERS[0].name)
        self.controller_cb = ttk.Combobox(
            select_frame,
            textvariable=self.controller_var,
            values=[c.name for c in CONTROLLERS],
            state="readonly",
            width=34,
        )
        self.controller_cb.grid(row=1, column=0, sticky="ew")

        # Status log
        self.status = tk.Text(
            center,
            height=6,
            width=80,
            bg="#111317",
            fg="#e5e7eb",
            bd=0,
            highlightthickness=0,
        )
        self.status.grid(row=3, column=0, pady=(20, 0))
        self.status.configure(state="disabled")

        if ENV_FACTORY is None:
            self._log(
                "⚠️ KAZ v10 not found. In your venv run:\n"
                "   pip install 'pettingzoo[butterfly]>=1.24' 'gymnasium==0.29.1' pygame numpy\n"
                "Then relaunch."
            )

        # Bottom bar with Play
        bottom = ttk.Frame(self)
        bottom.grid(row=2, column=0, sticky="nsew", padx=24, pady=24)
        bottom.grid_columnconfigure(0, weight=1)

        self.play_btn = ttk.Button(bottom, text="Play", style="Play.TButton", command=self.on_play)
        self.play_btn.grid(row=0, column=0)

        self._child_proc = None

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

    def _poll_child(self):
        if self._child_proc is None:
            return
        code = self._child_proc.poll()
        if code is None:
            self.after(500, self._poll_child)
            return
        self._log(f"Game exited with code {code}.")
        self.play_btn.configure(state="normal")
        self._child_proc = None

    # -------------------- Action --------------------
    def on_play(self):
        if ENV_FACTORY is None:
            err = "\n".join(IMPORT_ERRORS) or "Unknown import error."
            messagebox.showerror(
                "KAZ v10 required",
                "KAZ v10 not available in this environment.\n\n"
                "Run:\n  pip install 'pettingzoo[butterfly]>=1.24' 'gymnasium==0.29.1' pygame numpy\n\n"
                "Details:\n" + err,
            )
            return

        if self._child_proc is not None:
            self._log("A game is already running.")
            return

        name = self.controller_var.get()
        self.play_btn.configure(state="disabled")

        # Launch separate process (works on Windows/macOS/Linux)
        cmd = [sys.executable, os.path.abspath(__file__), "--run-game", name]
        try:
            self._child_proc = subprocess.Popen(cmd)
            self._log(f"Starting with controller: {name}")
            self._log(f"Using env: {ENV_LABEL or 'unknown'}")
            self.after(500, self._poll_child)
        except Exception as e:
            self._log(f"Failed to start game: {type(e).__name__}: {e}")
            self.play_btn.configure(state="normal")


# ---------------------------- CLI entrypoint -----------------------------------

def _run_game_cli():
    # invoked as: python kaz_launcher.py --run-game "<controller name>"
    if len(sys.argv) < 3:
        print("Usage: python kaz_launcher.py --run-game \"<controller name>\"")
        sys.exit(2)
    controller_name = " ".join(sys.argv[2:])
    run_kaz(controller_name, status_cb=print)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--run-game":
        _run_game_cli()
    else:
        app = App()
        app.mainloop()
