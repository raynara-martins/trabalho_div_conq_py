from __future__ import annotations

import random

from .classic import mul_classic, matrices_equal
from .strassen import mul_strassen, StrassenStats
from .matrix import Matrix


def gerar_matriz(n: int, seed: int, minimo: int = -5, maximo: int = 5) -> Matrix:
    """Gera uma matriz n x n com inteiros aleatórios usando seed fixa."""
    rng = random.Random(seed)
    return [[rng.randint(minimo, maximo) for _ in range(n)] for _ in range(n)]


def garantir_iguais(C1: Matrix, C2: Matrix, msg: str) -> None:
    """Se forem diferentes, mostra a primeira posição onde diverge."""
    if matrices_equal(C1, C2):
        return

    n = len(C1)
    for i in range(n):
        for j in range(n):
            if C1[i][j] != C2[i][j]:
                raise AssertionError(
                    f"{msg}\nDiferença em ({i},{j}): clássico={C1[i][j]} strassen={C2[i][j]}"
                )
    raise AssertionError(msg)


def comparar_classico_vs_strassen(A: Matrix, B: Matrix, cutoff: int = 4) -> None:
    """Calcula pelos dois métodos e garante que o resultado é igual."""
    C_classico = mul_classic(A, B)
    C_strassen, _ = mul_strassen(A, B, cutoff=cutoff, stats=StrassenStats())
    garantir_iguais(C_classico, C_strassen, "Clássico != Strassen")


# test 1: Caso  pequeno
def teste_fixo_2x2() -> None:
    A = [
        [1, 2],
        [3, 4],
    ]
    B = [
        [5, 6],
        [7, 8],
    ]
    comparar_classico_vs_strassen(A, B, cutoff=1)
    print("OK - teste_fixo_2x2 (resultado do Strassen bate com o clássico)")


# test 2: Vários aleatórios em tamanhos pequenos

def teste_aleatorios_pequenos() -> None:
    tamanhos = [1, 2, 4, 8, 16]
    for n in tamanhos:
        for s in range(10):
            A = gerar_matriz(n, seed=1000 + n * 10 + s)
            B = gerar_matriz(n, seed=2000 + n * 10 + s)
            comparar_classico_vs_strassen(A, B, cutoff=4)

    print("OK - teste_aleatorios_pequenos (vários casos aleatórios passaram)")


# test 3: Tamanhos que NÃO são potência de 2

def teste_nao_potencia_de_2_padding() -> None:
    tamanhos = [3, 5, 6, 10, 12]
    for n in tamanhos:
        A = gerar_matriz(n, seed=3000 + n)
        B = gerar_matriz(n, seed=4000 + n)
        comparar_classico_vs_strassen(A, B, cutoff=4)

    print("OK - teste_nao_potencia_de_2_padding (padding/crop funcionando)")


def main() -> None:
    print("Rodando testes (validação do Strassen comparando com o clássico)...\n")
    teste_fixo_2x2()
    teste_aleatorios_pequenos()
    teste_nao_potencia_de_2_padding()
    print("\nOK - Todos os testes passaram! Strassen e Clássico estão consistentes.")


if __name__ == "__main__":
    main()
