from __future__ import annotations

import csv
import os
import random
from time import perf_counter
from typing import List, Tuple

from .classic import mul_classic
from .strassen import mul_strassen, StrassenStats
from .matrix import Matrix


def gerar_matriz(n: int, rng: random.Random, minimo: int, maximo: int) -> Matrix:
    """Gera matriz n x n com inteiros uniformes em [minimo, maximo]."""
    return [[rng.randint(minimo, maximo) for _ in range(n)] for _ in range(n)]


def medir_tempo(fn, *args, **kwargs) -> Tuple[float, object]:
    """Mede tempo total de execução da função (segundos)."""
    t0 = perf_counter()
    result = fn(*args, **kwargs)
    t1 = perf_counter()
    return (t1 - t0), result


def parse_sizes_env(default: List[int]) -> List[int]:
    """
    Lê SIZES="64,128,256,512" do ambiente.
    Se não existir, usa default.
    """
    sizes_env = os.getenv("SIZES", "").strip()
    if not sizes_env:
        return default
    return [int(x.strip()) for x in sizes_env.split(",") if x.strip()]


def main() -> None:
    
    sizes = parse_sizes_env([64, 128, 256, 512])  
    seed_base = int(os.getenv("SEED", "42"))
    minimo = int(os.getenv("VAL_MIN", "0"))
    maximo = int(os.getenv("VAL_MAX", "10"))
    repeats = int(os.getenv("REPEATS", "3"))
    cutoff = int(os.getenv("CUTOFF", "64"))
    out_csv = os.getenv("OUT_CSV", "benchmark_results.csv")

    print("=== Benchmark: Clássico vs Strassen ===")
    print(f"Tamanhos: {sizes}")
    print(f"Seed base: {seed_base} | Intervalo: [{minimo}, {maximo}]")
    print(f"Repetições: {repeats} | Cutoff Strassen: {cutoff}")
    print(f"Saída CSV: {out_csv}\n")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n",
            "algoritmo",
            "repeticao",
            "tempo_total_seg",
            "strassen_calls",
            "strassen_split_combine_seg",
            "seed_base",
            "val_min",
            "val_max",
            "cutoff"
        ])

        for n in sizes:
            print(f"--- n = {n} ---")

            for r in range(1, repeats + 1):
                
                local_seed = seed_base + (n * 1000) + r
                rng = random.Random(local_seed)

                A = gerar_matriz(n, rng, minimo, maximo)
                B = gerar_matriz(n, rng, minimo, maximo)

                # 1) Clássico
                t_classic, _ = medir_tempo(mul_classic, A, B)
                writer.writerow([n, "classic", r, f"{t_classic:.9f}", "", "", seed_base, minimo, maximo, cutoff])
                print(f"[{r}/{repeats}] classic  : {t_classic:.6f} s")

                # 2) Strassen (com métricas)
                stats = StrassenStats()
                t_strassen, (C, stats_out) = medir_tempo(mul_strassen, A, B, cutoff, stats)

                writer.writerow([
                    n, "strassen", r, f"{t_strassen:.9f}",
                    stats_out.calls, f"{stats_out.split_combine_time:.9f}",
                    seed_base, minimo, maximo, cutoff
                ])

                print(
                    f"[{r}/{repeats}] strassen : {t_strassen:.6f} s | "
                    f"calls={stats_out.calls} | split+combine={stats_out.split_combine_time:.6f} s"
                )

            print()

    print(f"✅ Benchmark concluído. CSV gerado em: {out_csv}")


if __name__ == "__main__":
    main()
