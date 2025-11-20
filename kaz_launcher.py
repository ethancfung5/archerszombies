# kaz_launcher.py
import threading
import tkinter as tk
from tkinter import ttk, messagebox

# local imports from controllers package
from controllers import CONTROLLERS, get_controller_by_name

# ---- Try multiple import paths to handle PettingZoo version differences ----
ENV_FACTORY = None
IMPORT_ERRORS = []


def _try_imports():
    global ENV_FACTORY
    candidates = [
        # common modern path
        ("from pettingzoo.butterfly import knights_archers_zombies_v10 as kaz", "kaz"),
        # fallbacks for older versions
        ("from pettingzoo.butterfly import knights_archers_zombies_v9 as kaz", "kaz"),
        ("from pettingzoo.butterfly import knights_archers_zombies_v8 as kaz", "kaz"),
        ("from pettingzoo.butterfly import knights_archers_zombies_v7 as kaz", "kaz"),
    ]
    for stmt, name in candidates:
        try:
            local = {}
            exec(stmt, {}, local)
            kaz = local[name]

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
    controller = get_controller_by_name(selected_controller_name)

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

            if termination or truncation:
                action = None  # no-op (PettingZoo convention)
            else:
                action_space = env.action_space(agent)
                action = controller(obs, action_space, agent, t)

            env.step(action)

            # If the controller exposes an update_scores method, call it
            if action is not None and hasattr(controller, "update_scores"):
                controller.update_scores(agent, action, reward)

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
    def __init__(self):
        super().__init__()

        # --- DPI / scaling ---
        try:
            self.tk.call("tk", "scaling", 1.25)
        except Exception:
            pass

        self.title("KAZ Demo Launcher")
        self.configure(bg="#0f1115")  # dark background

        # ---- Full-screen behaviors ----
        self._fullscreen = True
        self.attributes("-fullscreen", True)
        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", self._exit_fullscreen)

        # ---- TTK theming ----
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

        # ---- Root grid ----
        self.grid_rowconfigure(0, weight=1)  # spacer
        self.grid_rowconfigure(1, weight=2)  # center content
        self.grid_rowconfigure(2, weight=1)  # bottom bar
        self.grid_columnconfigure(0, weight=1)

        # ---- Center panel ----
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

        # Small info log
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
                "⚠️ Could not import KAZ yet. Try `pip install pettingzoo pygame`.\n"
                "See errors in console if it still fails."
            )

        # ---- Bottom bar with Play button ----
        bottom = ttk.Frame(self)
        bottom.grid(row=2, column=0, sticky="nsew", padx=24, pady=24)
        bottom.grid_columnconfigure(0, weight=1)

        self.play_btn = ttk.Button(bottom, text="Play", style="Play.TButton", command=self.on_play)
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