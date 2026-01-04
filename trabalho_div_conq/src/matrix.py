# src/matrix.py
from __future__ import annotations

from typing import List, Tuple

Matrix = List[List[int]]


def zeros(n: int) -> Matrix:
    """Cria uma matriz n x n preenchida com 0."""
    return [[0 for _ in range(n)] for _ in range(n)]


def add(A: Matrix, B: Matrix) -> Matrix:
    """Retorna A + B (mesmo tamanho)."""
    n = len(A)
    C = zeros(n)
    for i in range(n):
        Ai = A[i]
        Bi = B[i]
        Ci = C[i]
        for j in range(n):
            Ci[j] = Ai[j] + Bi[j]
    return C


def sub(A: Matrix, B: Matrix) -> Matrix:
    """Retorna A - B (mesmo tamanho)."""
    n = len(A)
    C = zeros(n)
    for i in range(n):
        Ai = A[i]
        Bi = B[i]
        Ci = C[i]
        for j in range(n):
            Ci[j] = Ai[j] - Bi[j]
    return C


def split(A: Matrix) -> Tuple[Matrix, Matrix, Matrix, Matrix]:
    """
    Divide uma matriz n x n (n par) em 4 quadrantes:
    A11 A12
    A21 A22
    """
    n = len(A)
    mid = n // 2

    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]

    return A11, A12, A21, A22


def combine(C11: Matrix, C12: Matrix, C21: Matrix, C22: Matrix) -> Matrix:
    """Combina 4 quadrantes em uma matriz única."""
    mid = len(C11)
    n = mid * 2
    C = zeros(n)

    for i in range(mid):
        C[i][:mid] = C11[i]
        C[i][mid:] = C12[i]

    for i in range(mid):
        C[i + mid][:mid] = C21[i]
        C[i + mid][mid:] = C22[i]

    return C


def next_power_of_two(n: int) -> int:
    """Retorna a menor potência de 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def pad_to_size(A: Matrix, new_n: int) -> Matrix:
    """
    Faz padding com zeros para transformar A (n x n) em (new_n x new_n).
    Assumimos new_n >= n.
    """
    n = len(A)
    P = zeros(new_n)

    for i in range(n):
        Pi = P[i]
        Ai = A[i]
        for j in range(n):
            Pi[j] = Ai[j]

    return P


def crop(A: Matrix, n: int) -> Matrix:
    """Corta a matriz A para o tamanho n x n (usado após padding)."""
    return [row[:n] for row in A[:n]]


def is_square(A: Matrix) -> bool:
    """Validação simples: matriz quadrada."""
    n = len(A)
    return all(len(row) == n for row in A)


def assert_square(A: Matrix, name: str = "A") -> None:
    """Lança erro se não for quadrada."""
    if not is_square(A):
        raise ValueError(f"Matriz {name} não é quadrada.") 
