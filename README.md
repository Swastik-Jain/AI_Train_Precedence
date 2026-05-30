# RailMind 🚆🧠
**AI-Powered Train Precedence & Dispatch System**

RailMind is an advanced multi-agent reinforcement learning (RL) system built to optimize train precedence, dispatching, and movement within complex railway networks (such as the CSMT-Manmad corridor). By leveraging a custom Gymnasium environment and Proximal Policy Optimization (PPO), RailMind addresses scheduling complexities, resolves high-density traffic conflicts, and ensures dead-lock-free operations using a deterministic safety shield.

## ✨ Core Features

- **Reinforcement Learning Engine**: Utilizes Stable-Baselines3 (PPO) to learn and evolve optimal train dispatch strategies.
- **High-Density Curriculum Training**: Scales from simple to complex scenarios, managing up to 15+ concurrent trains under strict network constraints.
- **Deterministic Safety Shield**: A robust rule-based layer that intercepts AI-suggested actions to guarantee collision avoidance and capacity constraint adherence.
- **Digital Twin Real-Time Backend**: A high-performance FastAPI backend leveraging a database to track train telemetry and network states dynamically.
- **Topological Simulation**: Faithfully models realistic constraints, including bidirectional token-block bottlenecks (e.g., Ghat sections), varying speed limits, and junction layouts.
- **Live Frontend Dashboard**: A sleek React + Vite frontend integrating dynamic map visualizations for tracking real-time fleet movement.

## 🏗️ Architecture & Project Structure

The repository is divided into two primary sub-projects:

- **`/backend`**: Contains the core Python simulation, RL environment (`train_env.py`), FastAPI server (`main.py`), database connectors, and training pipelines.
- **`/frontend`**: Contains the React dashboard built with TypeScript, Vite, and Tailwind CSS to visualize live train tracking and network topology.

### Backend Overview
- `backend/train_env.py`: The Gymnasium-based physical simulation of the train tracks, speed limits, and signalling.
- `backend/ai/`: Houses the reinforcement learning model setup, weights, and agent interfaces.
- `backend/main.py`: The FastAPI server that interacts with the frontend and serves the state.
- `backend/or_tools/`: Constraint programming models for baseline and fallback evaluation.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+ 
- Node.js 18+ and npm

### 1. Backend Setup

Navigate to the `backend` directory and set up your Python environment:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirement.txt
```

**Running the System**:
You can run the FastAPI server to start the system backend:
```bash
uvicorn main:app --reload --port 8000
```
*(Check `backend/scripts` and `backend/run_training.sh` for specific curriculum training execution).*

### 2. Frontend Setup

Navigate to the `frontend` directory:

```bash
cd frontend
npm install
npm run dev
```

The React dashboard will be accessible at `http://localhost:5173`.

## 🧪 Training & Evaluation

To train the RL dispatcher, the project uses a curriculum learning approach:
1. Progressively increase train density (from 2 up to 15+).
2. Utilize TensorBoard to analyze stability, collision rates, and reward distributions.
3. Model checkpoints are auto-saved to the `backend/results/` or `backend/models/` directories.

Run TensorBoard to view metrics:
```bash
tensorboard --logdir backend/logs
```

## 🤝 Contributing
1. Fork the repository
2. Create a new feature branch (`git checkout -b feature/awesome-feature`)
3. Commit your changes (`git commit -m 'Add awesome feature'`)
4. Push to the branch (`git push origin feature/awesome-feature`)
5. Open a Pull Request

## 📜 License
This project is licensed under the MIT License.