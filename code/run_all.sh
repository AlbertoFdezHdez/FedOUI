#!/usr/bin/env bash
set -euo pipefail

python code/main.py --config code/config/cifar10_dirichlet_0.1.yaml --method fedavg
python code/main.py --config code/config/cifar10_dirichlet_0.1.yaml --method fedprox
python code/main.py --config code/config/cifar10_dirichlet_0.1.yaml --method fedoui
python code/main.py --config code/config/cifar10_dirichlet_0.1.yaml --method fedalign
python code/main.py --config code/config/cifar10_dirichlet_0.1.yaml --method fedoui_align

