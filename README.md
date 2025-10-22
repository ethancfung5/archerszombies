# Knights–Archers–Zombies (KAZ) Demo with Search Controllers  

This project demonstrates the **Knights–Archers–Zombies (KAZ)** environment from [PettingZoo](https://www.pettingzoo.ml/) with different search algorithms and controllers.  
It includes a **full-screen launcher menu** built with Tkinter that allows you to select a search method and launch the game with a single click.  

---

## 🎮 Features  

- **Full-screen home menu** with dark theme and clean layout.  
- Dropdown to select the search controller (Random, Greedy heuristic, DFS, BFS, A* stubs).  
- **Play button** launches the KAZ game in a new window.  
- Status box logs events (episode progress, errors, etc.).  
- Keyboard shortcuts:  
  - `F11` → Toggle full-screen  
  - `Esc` → Exit full-screen  

---

## 🚀 Getting Started  

### 1. Clone the repository  

```bash
git clone https://github.com/yourusername/kaz-search-demo.git
cd kaz-search-demo
```

### 2. Create a virtual environment  

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies  

```bash
pip install pettingzoo pygame numpy
```

### 4. Run the launcher  

```bash
python kaz_launcher.py
```

---

## 🧠 Controllers  

The project will support several **controllers/policies**:  

---

## 📸 Screenshots  

![alt text](images/HomeScreen.png)

![alt text](/images/KAZ_pic.png)

---


## 📚 Learning Goals  

- Understand the basics of **PettingZoo environments**.  
- Learn how **search algorithms** (DFS, BFS, A*, greedy) can be adapted to a multi-agent environment.  
- Explore **UI integration** (Tkinter) for game simulations.  

---

## 📜 License  

This project is open-source under the MIT License.  
