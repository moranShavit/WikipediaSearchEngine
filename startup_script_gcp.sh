#!/bin/bash
set -e

APP_USER="shhara"
APP_HOME="/home/${APP_USER}"
VENV_DIR="${APP_HOME}/venv"
REQ_FILE="${APP_HOME}/requirements.txt"

apt-get update
apt-get install -y python3-venv python3-pip

# Create venv as the normal user (not root)
sudo -u "${APP_USER}" bash -lc "
python3 -m venv '${VENV_DIR}'
source '${VENV_DIR}/bin/activate'
pip install --upgrade pip
pip install -r '${REQ_FILE}'
