# Import library
import json
import random
from statsmodels.stats.contingency_tables import mcnemar
from pathlib import Path

PASTA = Path("json/results")

ambos = 0
nenhum = 0
llm = 0
sasrec = 0

resultados = []

with open(PASTA / "métricas_avaliadas.jsonl", "r", encoding="utf-8") as g:
    for row in g:
        resultados.append(json.loads(row))

for r in resultados:
    if (r["llm_hit@3"] == 1 and r["sasrec_hit@3"] == 1):
        ambos += 1
    elif (r["llm_hit@3"] == 1 and r["sasrec_hit@3"] == 0):
        llm += 1
    elif (r["llm_hit@3"] == 0 and r["sasrec_hit@3"] == 1):
        sasrec += 1
    else:
        nenhum += 1

# Create a dataset
data = [[ambos, llm],
        [sasrec, nenhum]]

print(ambos)
print(llm)
print(sasrec)
print(nenhum)

# 'exact = true' utiliza os números inteiros
print(mcnemar(data, exact=True))

# Bootstrap
def diferencas(lista):
    dif_hit = sum(r["llm_hit@3"] for r in lista) / len(lista) - sum(r["sasrec_hit@3"] for r in lista) / len(lista)
    dif_ndcg = sum(r["llm_ndcg@3"] for r in lista) / len(lista) - sum(r["sasrec_ndcg@3"] for r in lista) / len(lista)
    return dif_hit, dif_ndcg

random.seed(42)

hit_observado, ndcg_observado = diferencas(resultados)

hit = []
ndcg = []

for _ in range(10000):
    amostra = random.choices(resultados, k=len(resultados))
    dif_hit, dif_ndcg = diferencas(amostra)
    hit.append(dif_hit)
    ndcg.append(dif_ndcg)

hit = sorted(hit)
ndcg = sorted(ndcg)

print("Hit@3:", hit_observado, "IC 95%:", hit[250], hit[9750])
print("NDCG@3:", ndcg_observado, "IC 95%:", ndcg[250], ndcg[9750])