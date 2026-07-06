#!/usr/bin/env python
"""CLI прогона экспериментов (SPEC §6).

    python run.py R0                # полный прогон
    python run.py R0 --smoke        # smoke-версия
    python run.py R0 --seed 2       # переопределить сид
"""

import argparse
import os
import sys
from pathlib import Path

# Не преаллоцировать 75% VRAM: наши массивы — единицы МБ, а GPU делится с
# другими процессами. Аллокация по требованию убирает ложные OOM-откаты.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# src-layout: делаем ribbon_sim импортируемым без установки пакета.
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from ribbon_sim import experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="имя эксперимента, напр. R0")
    ap.add_argument("--smoke", action="store_true", help="smoke-версия (SPEC CLAUDE.md)")
    ap.add_argument("--seed", type=int, default=None, help="переопределить сид")
    args = ap.parse_args()

    cfg_path = ROOT / "experiments" / f"{args.name}.yaml"
    cfg = experiment.load_config(cfg_path)
    mode = "smoke" if args.smoke else "full"

    print(f"[{args.name}] режим={mode}  seed_override={args.seed}")
    res = experiment.run_sweep(cfg, mode=mode, seed_override=args.seed)

    subdir = args.name if mode == "full" else f"{args.name}/smoke"
    outdir = ROOT / "results" / subdir
    rep = experiment.write_report(cfg, res, outdir)

    print(f"[{args.name}] прогон {res['elapsed_s']:.1f} с; "
          f"вердикт {'PASS' if rep['verdict_pass'] else 'FAIL'}; "
          f"max|E|={rep['max_absE']:.4f} (порог {rep['E_tol']:.4f})")
    print(f"[{args.name}] отчёт: {rep['path']}")
    if not rep["verdict_pass"] and mode == "full":
        sys.exit(1)


if __name__ == "__main__":
    main()
