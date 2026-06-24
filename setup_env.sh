#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "=== Franka RL Isaac Lab — Environment Setup ==="

# --- 1. Create virtual environment ---
if [ ! -d "${VENV_DIR}" ]; then
    echo "[1/5] Creating Python 3.10 virtual environment..."
    python3.10 -m venv "${VENV_DIR}"
else
    echo "[1/5] Virtual environment already exists, skipping."
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip

# --- 2. Install Isaac Sim (pip) ---
echo "[2/5] Installing Isaac Sim via pip (this may take a while)..."
pip install isaacsim[all] --extra-index-url https://pypi.nvidia.com

# --- 3. Clone and install Isaac Lab ---
echo "[3/5] Installing Isaac Lab..."
if [ ! -d "${SCRIPT_DIR}/isaaclab" ]; then
    git clone https://github.com/isaac-sim/IsaacLab.git "${SCRIPT_DIR}/isaaclab"
fi
cd "${SCRIPT_DIR}/isaaclab"
pip install -e .

# --- 4. Install RL frameworks ---
echo "[4/5] Installing RL libraries..."
pip install "rsl-rl>=2.0" "rl-games>=1.6" "stable-baselines3>=2.0" "sb3-contrib"

# --- 5. Install this project extension ---
echo "[5/5] Installing franka_rl extension..."
cd "${SCRIPT_DIR}"
pip install -e exts/franka_rl

echo ""
echo "=== Setup complete! ==="
echo "Activate with:  source ${VENV_DIR}/bin/activate"
echo "Train with:     python scripts/train.py --task Franka-Reach-v0"
