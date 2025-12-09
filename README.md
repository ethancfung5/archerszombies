# Knights–Archers–Zombies (KAZ) Search Demo

*A cross-platform Tkinter launcher for the PettingZoo KAZ environment with pluggable search controllers*

This project provides a **full-screen launcher UI** for the **Knights–Archers–Zombies (KAZ)** environment from the [PettingZoo](https://pettingzoo.farama.org) library.
It allows users to:

* Select a search controller (Random, Greedy, DFS, BFS, A*, etc.)
* Launch the KAZ game in a **separate process** (required for macOS stability)
* View status logs during gameplay
* Play the simulation through a clean fullscreen menu

The launcher works on **Windows**, **macOS**, and **Linux**.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ethancfung5/archerszombies.git
cd archerszombies
```

### 2. Prerequisites: Python 3

This project requires Python 3.10+.

Check your Python version:
macOS / Linux
```
python3 --version
```
Windows
```
python --version
```

If Python 3 is not installed, download it from:
https://www.python.org/downloads/
### 3. Create & activate a virtual environment

#### macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Your shell prompt should now show:

```
(.venv)
```

### 4. Install required dependencies

PettingZoo’s KAZ environment (v10) lives in the **butterfly** family and requires Gymnasium + pygame.

Install everything with:

```bash
pip install --upgrade pip
pip install "pettingzoo[butterfly]>=1.24" pygame==2.6.1 numpy screeninfo
```

### 5. Run the launcher

```bash
python kaz_launcher.py
```

## Controllers

Controllers live in the `controllers/` package and follow a simple interface:

```python
action = controller(observation, action_space, agent_name, timestep)
```

Included stubs:

* Random
* Greedy Heuristic
* Depth-First Search (DFS)
* Breadth-First Search (BFS)
* Uniform-Cost / Dijkstra (stub)
* A* Search (stub)

You can easily add new controllers by creating new modules and registering them in
`controllers/__init__.py`.

---

## 🖼️ Screenshots

### Launcher Home Screen

![Launcher](images/HomeScreen.png)

### In-Game KAZ Environment

![KAZ](images/KAZ_pic.png)

---

## Goals

This project aims to:

* How to work with **PettingZoo**’s multi-agent API
* How to adapt **search algorithms** (DFS, BFS, A*, greedy) to multi-agent game environments
* How to integrate **Tkinter UI** with external simulations
* How to build safe, cross-platform GUI pipelines using process isolation

---

## License

This project is open-source under the MIT License.
Feel free to fork, modify, and use it for coursework or research projects.

---