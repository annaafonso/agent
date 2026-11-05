import math

def hit_at_k(topk, gabarito, k):
    return 1.0 if any(i in gabarito for i in topk[:k]) else 0.0

def recall_at_k(topk, gabarito, k):
    if not gabarito:
        return 0.0
    acertos = sum(1 for i in topk[:k] if i in gabarito)
    return acertos / len(gabarito)

def ndcg_at_k(topk, gabarito, k):
    dcg = sum(1 / math.log2(pos + 2)          
              for pos, i in enumerate(topk[:k]) if i in gabarito)
    ideais = min(len(gabarito), k)
    idcg = sum(1 / math.log2(pos + 2) for pos in range(ideais))
    return dcg / idcg if idcg > 0 else 0.0

if __name__ == "__main__":
    recomendados = [10, 20, 30]
    relevantes = {20, 40}
    k = 3

    print(hit_at_k(recomendados, relevantes, k))
    print(recall_at_k(recomendados, relevantes, k))
    print(ndcg_at_k(recomendados, relevantes, k))