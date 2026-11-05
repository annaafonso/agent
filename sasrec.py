import torch
import json
import prepare_split
import define_metrics
from pathlib import Path
from recbole.data.interaction import Interaction
from recbole.quick_start import load_data_and_model

PASTA = Path("json/results")

def load_model():
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file="saved/SASRec-Sep-22-2026_18-54-53.pth"
    )
    model.eval()
    return config, model, dataset

def return_topk(historico, config, model, dataset):
    vocabulario_itens = set(dataset.field2token_id[dataset.iid_field].keys())
    historico_movie_ids = [mid for mid in historico if str(mid) in vocabulario_itens]

    if len(historico_movie_ids) == 0:
        return None

    vistos_internos = dataset.token2id(dataset.iid_field, [str(mid) for mid in historico_movie_ids])

    max_len = config['MAX_ITEM_LIST_LENGTH']
    if len(historico_movie_ids) > max_len:
        historico_movie_ids = historico_movie_ids[-max_len:]

    item_seq_internos = dataset.token2id(dataset.iid_field, [str(mid) for mid in historico_movie_ids])

    item_seq_tensor = torch.zeros(max_len, dtype=torch.long).to(config['device'])
    item_seq_tensor[:len(item_seq_internos)] = torch.tensor(item_seq_internos).to(config['device'])
    item_seq_len_tensor = torch.tensor([len(item_seq_internos)]).to(config['device'])

    with torch.no_grad():
        interaction = Interaction({
            model.ITEM_SEQ: item_seq_tensor.unsqueeze(0),
            model.ITEM_SEQ_LEN: item_seq_len_tensor
        })

        scores = model.full_sort_predict(interaction)
        scores[0, 0] = -float('inf')
        scores[0, torch.as_tensor(vistos_internos, dtype=torch.long, device=scores.device)] = -float('inf')
        topk_scores, topk_indices = torch.topk(scores, k=30)

    topk_movie_ids = dataset.id2token(dataset.iid_field, topk_indices.cpu())
    return [int(mid) for mid in topk_movie_ids[0]]

if __name__ == '__main__':
    config, model, dataset = load_model()

    entrada, gabarito = prepare_split.load_json()

    resultados = []

    for userId, lista in entrada.items():
        historico = [movieId for timestamp, movieId in lista]
        resultado = return_topk(historico, config, model, dataset)

        relevantes = {movieId for timestamp, movieId in gabarito[userId]}

        hit = define_metrics.hit_at_k(resultado, relevantes, 30)
        ndcg = define_metrics.ndcg_at_k(resultado, relevantes, 30)
        recall = define_metrics.recall_at_k(resultado, relevantes, 30)

        resultados.append({
            "userId": userId,
            "topk": resultado,
            "hit": hit,
            "ndcg": ndcg,
            "recall": recall
        })

    medias = {
        "hit@30": sum(r["hit"] for r in resultados) / len(resultados),
        "ndcg@30": sum(r["ndcg"] for r in resultados) / len(resultados),
        "recall@30": sum(r["recall"] for r in resultados) / len(resultados),
        "usuários": len(resultados)
    }

    print(medias)

    PASTA.mkdir(parents=True, exist_ok=True)

    with open(PASTA / "métricas_por_usuário_30.jsonl", "w", encoding="utf-8") as r:
        for linha in resultados:
            r.write(json.dumps(linha, ensure_ascii=False) + "\n")

    with open(PASTA / "métricas_médias_30.json", "w", encoding="utf-8") as m:
        json.dump(medias, m, indent=4, ensure_ascii=False)
    
    # resultado2 = return_topk([], config, model, dataset)
    # print("Usuário sem histórico:", resultado2)

    # print(gabarito[1])