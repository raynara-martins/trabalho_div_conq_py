# src/strassen.py
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Tuple

from .matrix import (
    Matrix,
    add,
    sub,
    split,
    combine,
    next_power_of_two,
    pad_to_size,
    crop,
    assert_square,
)
from .classic import mul_classic


@dataclass
class StrassenStats:
    """
    Estatísticas exigidas no trabalho:
    - calls: número de chamadas recursivas (quantas vezes strassen_rec foi chamada)
    - split_combine_time: tempo gasto SOMENTE em particionar (split) e combinar (combine),
      além de somas/subtrações necessárias ao Strassen (overhead estrutural).
    """
    calls: int = 0
    split_combine_time: float = 0.0


def mul_strassen(A: Matrix, B: Matrix, cutoff: int = 64, stats: Optional[StrassenStats] = None) -> Tuple[Matrix, StrassenStats]:
    """
    Multiplicação de matrizes usando Strassen.

    Requisitos atendidos:
    - Divide recursivamente em quadrantes
    - Faz as 7 multiplicações de Strassen
    - Combina resultados
    - Trata tamanhos que NÃO são potência de 2 (padding + crop)
    - Mede estatísticas: chamadas recursivas + tempo de particionamento/combinação

    Retorna: (C, stats)
    """
    assert_square(A, "A")
    assert_square(B, "B")

    n = len(A)
    if len(B) != n:
        raise ValueError("A e B devem ter o mesmo tamanho (n x n).")

    if stats is None:
        stats = StrassenStats()

    # Padding caso n não seja potência de 2
    m = next_power_of_two(n)
    if m != n:
        A_pad = pad_to_size(A, m)
        B_pad = pad_to_size(B, m)
        C_pad = _strassen_rec(A_pad, B_pad, cutoff, stats)
        C = crop(C_pad, n)
        return C, stats

    # Já é potência de 2
    C = _strassen_rec(A, B, cutoff, stats)
    return C, stats


def _strassen_rec(A: Matrix, B: Matrix, cutoff: int, stats: StrassenStats) -> Matrix:
    """
    Parte recursiva do Strassen.
    Aqui contamos calls e acumulamos split_combine_time.

    cutoff:
      - Se n <= cutoff, usamos a multiplicação clássica (reduz overhead em Python).
    """
    stats.calls += 1
    n = len(A)

    # Caso base: 1x1 ou limiar
    if n == 1:
        return [[A[0][0] * B[0][0]]]

    if n <= cutoff:
        return mul_classic(A, B)

    # Medir somente o "overhead estrutural": split + add/sub + combine
    t0 = perf_counter()

    # 1) Divide em quadrantes
    A11, A12, A21, A22 = split(A)
    B11, B12, B21, B22 = split(B)

    # 2) Preparar as somas/subtrações (Strassen)
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    A11_plus_A22 = add(A11, A22)
    B11_plus_B22 = add(B11, B22)

    A21_plus_A22 = add(A21, A22)
    B12_minus_B22 = sub(B12, B22)

    B21_minus_B11 = sub(B21, B11)
    A11_plus_A12 = add(A11, A12)

    A21_minus_A11 = sub(A21, A11)
    B11_plus_B12 = add(B11, B12)

    A12_minus_A22 = sub(A12, A22)
    B21_plus_B22 = add(B21, B22)

    stats.split_combine_time += perf_counter() - t0

    # 3) As 7 multiplicações recursivas (contabilizam calls dentro delas)
    M1 = _strassen_rec(A11_plus_A22, B11_plus_B22, cutoff, stats)
    M2 = _strassen_rec(A21_plus_A22, B11, cutoff, stats)
    M3 = _strassen_rec(A11, B12_minus_B22, cutoff, stats)
    M4 = _strassen_rec(A22, B21_minus_B11, cutoff, stats)
    M5 = _strassen_rec(A11_plus_A12, B22, cutoff, stats)
    M6 = _strassen_rec(A21_minus_A11, B11_plus_B12, cutoff, stats)
    M7 = _strassen_rec(A12_minus_A22, B21_plus_B22, cutoff, stats)

    # 4) Combinar quadrantes do resultado:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    t1 = perf_counter()

    C11 = add(sub(add(M1, M4), M5), M7)
    C12 = add(M3, M5)
    C21 = add(M2, M4)
    C22 = add(sub(add(M1, M3), M2), M6)

    C = combine(C11, C12, C21, C22)

    stats.split_combine_time += perf_counter() - t1

    return C
