"""Настройка тестов фазы D: CPU-JAX, float64, путь к пакету ribbon_sim.

D0 — CPU-only (N≤64, fp64; fp64 на 4070 Ti ~1/64 скорости). x64 включается ДО
первого использования jax. src/ добавляется в путь для импорта ribbon_sim.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
