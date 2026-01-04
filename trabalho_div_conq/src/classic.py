# src/classic.py
from __future__ import annotations

from typing import List

from .matrix import Matrix, zeros, assert_square


def mul_classic(A: Matrix, B: Matrix) -> Matrix:
    """
    Multiplicação clássica de matrizes quadradas (O(n^3)).
    Implementação "baseline" com 3 laços aninhados.

    C[i][j] = sum(A[i][k] * B[k][j] for k in 0..n-1)
    """
    assert_square(A, "A")
    assert_square(B, "B")

    n = len(A)
    if len(B) != n:
        raise ValueError("A e B devem ter o mesmo tamanho (n x n).")

    C = zeros(n)

    # Laços aninhados (i, j, k) - forma mais direta
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for j in range(n):
            s = 0
            for k in range(n):
                s += Ai[k] * B[k][j]
            Ci[j] = s

    return C


def matrices_equal(A: Matrix, B: Matrix) -> bool:
    """Comparação exata (útil para validar Strassen com pequenos n)."""
    if len(A) != len(B):
        return False
    n = len(A)
    for i in range(n):
        if A[i] != B[i]:
            return False
    return True
