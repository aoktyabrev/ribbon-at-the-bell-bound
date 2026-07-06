#!/usr/bin/env python
"""CLI прогона экспериментов (SPEC §6).

    python run.py R1                # полный прогон
    python run.py R1 --smoke        # smoke-версия
    python run.py R1 --seed 2       # переопределить сид
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
    ap.add_argument("name", help="имя эксперимента, напр. R1")
    ap.add_argument("--smoke", action="store_true", help="smoke-версия (CLAUDE.md)")
    ap.add_argument("--seed", type=int, default=None, help="переопределить сид")
    args = ap.parse_args()

    cfg = experiment.load_config(ROOT / "experiments" / f"{args.name}.yaml")
    mode = "smoke" if args.smoke else "full"

    print(f"[{args.name}] режим={mode}  seed_override={args.seed}")
    exp = experiment.run_experiment(cfg, mode=mode, seed_override=args.seed)

    subdir = args.name if mode == "full" else f"{args.name}/smoke"
    outdir = ROOT / "results" / subdir
    rep = experiment.write_report(exp, outdir)

    print(f"[{args.name}] готово за {rep['total_elapsed']:.1f} с суммарно; отчёт: {rep['path']}")
    for res, an in zip(exp["cells"], rep["analyses"]):
        print(f"    {res['cell']['label']}: амплитуда max|E|={an['amp']:.3f}, "
              f"лучшая={an['best']}, p̂={an['cmp']['p_hat']:.3f}")


if __name__ == "__main__":
    main()
