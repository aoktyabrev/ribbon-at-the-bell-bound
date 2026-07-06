"""Настройка окружения тестов: аллокация GPU-памяти по требованию."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
