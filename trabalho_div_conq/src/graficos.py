from __future__ import annotations

import csv
from collections import defaultdict
from statistics import mean
import matplotlib.pyplot as plt

ARQUIVO_CSV = "benchmark_results.csv"


def to_float(value):
    if value in ("", None):
        return None
    return float(value)


def to_int(value):
    if value in ("", None):
        return None
    return int(value)


def carregar_dados():
    """
    Lê o CSV gerado pelo benchmark e padroniza tipos.
    Espera colunas:
      n, algoritmo, repeticao, tempo_total_seg, strassen_calls, strassen_split_combine_seg, ...
    """
    dados = []
    with open(ARQUIVO_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["n"] = int(row["n"])
            row["repeticao"] = int(row["repeticao"])
            row["tempo_total_seg"] = to_float(row["tempo_total_seg"])
            row["strassen_calls"] = to_int(row["strassen_calls"])
            row["strassen_split_combine_seg"] = to_float(row["strassen_split_combine_seg"])
            dados.append(row)
    return dados


def media_por_n_e_algoritmo(dados, campo):
    """
    Retorna:
      tamanhos (lista ordenada)
      medias_classic (lista)
      medias_strassen (lista)
    """
    bucket = defaultdict(list)

    for d in dados:
        n = d["n"]
        alg = d["algoritmo"]
        val = d.get(campo)
        if val is None:
            continue
        bucket[(n, alg)].append(val)

    tamanhos = sorted(set(d["n"] for d in dados))

    classic = []
    strassen = []
    for n in tamanhos:
        classic_vals = bucket.get((n, "classic"), [])
        strassen_vals = bucket.get((n, "strassen"), [])
        classic.append(mean(classic_vals) if classic_vals else None)
        strassen.append(mean(strassen_vals) if strassen_vals else None)

    return tamanhos, classic, strassen


def usar_escala_log(valores1, valores2=None):
    """
    Decide se usa escala log:
    - se houver valores > 0
    - e a razão max/min >= 100 (igual ideia do seu outro trabalho)
    """
    vals = [v for v in (valores1 or []) if v is not None and v > 0]
    if valores2:
        vals += [v for v in valores2 if v is not None and v > 0]
    if len(vals) < 2:
        return False
    mn = min(vals)
    mx = max(vals)
    return mn > 0 and (mx / mn) >= 100


def grafico_tempo_medio(dados, nome_arquivo="tempo_medio.png"):
    tamanhos, classic, strassen = media_por_n_e_algoritmo(dados, "tempo_total_seg")

    log = usar_escala_log(classic, strassen)

    plt.figure()
    plt.plot(tamanhos, classic, marker="o", label="Clássico")
    plt.plot(tamanhos, strassen, marker="o", label="Strassen")
    plt.xlabel("Tamanho da matriz (n)")
    plt.ylabel("Tempo médio (s)" + (" (escala log)" if log else ""))
    plt.title("Tempo médio de execução: Clássico vs Strassen")
    if log:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(nome_arquivo)
    plt.close()


def grafico_calls_strassen(dados, nome_arquivo="calls_strassen.png"):
   
    bucket = defaultdict(list)
    for d in dados:
        if d["algoritmo"] == "strassen" and d["strassen_calls"] is not None:
            bucket[d["n"]].append(d["strassen_calls"])

    tamanhos = sorted(bucket.keys())
    medias = [mean(bucket[n]) for n in tamanhos]

    log = usar_escala_log(medias)

    plt.figure()
    plt.plot(tamanhos, medias, marker="o")
    plt.xlabel("Tamanho da matriz (n)")
    plt.ylabel("Chamadas recursivas (média)" + (" (escala log)" if log else ""))
    plt.title("Número de chamadas recursivas do Strassen por tamanho")
    if log:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(nome_arquivo)
    plt.close()


def grafico_split_combine_strassen(dados, nome_arquivo="split_combine_strassen.png"):
    bucket = defaultdict(list)
    for d in dados:
        if d["algoritmo"] == "strassen" and d["strassen_split_combine_seg"] is not None:
            bucket[d["n"]].append(d["strassen_split_combine_seg"])

    tamanhos = sorted(bucket.keys())
    medias = [mean(bucket[n]) for n in tamanhos]

    log = usar_escala_log(medias)

    plt.figure()
    plt.plot(tamanhos, medias, marker="o")
    plt.xlabel("Tamanho da matriz (n)")
    plt.ylabel("Tempo médio split+combine (s)" + (" (escala log)" if log else ""))
    plt.title("Tempo de particionamento/combinação do Strassen por tamanho")
    if log:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(nome_arquivo)
    plt.close()


if __name__ == "__main__":
    dados = carregar_dados()

    grafico_tempo_medio(dados, "tempo_medio.png")
    grafico_calls_strassen(dados, "calls_strassen.png")
    grafico_split_combine_strassen(dados, "split_combine_strassen.png")

    print(" OK - Gráficos gerados: tempo_medio.png, calls_strassen.png, split_combine_strassen.png")
