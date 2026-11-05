import sasrec
import prepare_split
import json
import agent
import time
import requests
import random
import define_metrics
# from google.genai import errors
from pydantic import ValidationError
from pathlib import Path

PASTA = Path("json/results")

if __name__ == '__main__':

    config, model, dataset = sasrec.load_model() # Retorna 'config, model, dataset'

    entrada, gabarito = prepare_split.load_json() # Retorna 'entrada, gabarito'

    avaliados = set() # Conjunto vazio

    if (Path.exists(PASTA / "métricas_avaliadas_30.jsonl")): # Se existir o arquivo de 'resultados'
        with open(PASTA / "métricas_avaliadas_30.jsonl", "r", encoding="utf-8") as g:
            for row in g:
                user_dict = json.loads(row)
                avaliados.add(user_dict["userId"]) # Adicionar somente os valores de userId no conjunto

    entrada = list(entrada.items())[:608] # Transforma em lista [(userId, [timestamp, movieId])]

    PASTA.mkdir(parents=True, exist_ok=True)

    for userId, lista in entrada:
        if userId in avaliados:
            continue
        historico = [movieId for timestamp, movieId in lista]
        resultado_k = sasrec.return_topk(historico, config, model, dataset)

        random.seed(userId)
        resultado_3 = random.sample(resultado_k, k=3)

        prompt = agent.agent_instruction(userId, historico, resultado_k, [])

        llm = None

        for tentativa in range(5):
            try:
                inicio = time.perf_counter()
                llm, uso = agent.ask_agent(prompt, "Oi")
                latencia = time.perf_counter() - inicio
                break
            except requests.exceptions.RequestException as erro:
                print(f"Usuário {userId}, Tentativa {tentativa + 1}: {erro}")
                if erro.response is None or erro.response.status_code != 503:
                    break
                time.sleep(10 * 2 ** tentativa)
            except ValidationError as erro:
                print(f"Usuário {userId}, Resposta Malformada: {erro}")
                break

            """
            except errors.APIError as erro:
                print(f"Usuário {userId}, tentativa {tentativa + 1}: {erro}")
                if erro.code != 503:
                    break
                time.sleep(10 * 2 ** tentativa)
            """

        if llm is None:
            print(f"Usuário {userId} não foi avaliado")
            continue

        # try:
        #     llm = agent.ask_agent(prompt, "Oi")
        # except Exception as erro:
        #     print(f"Usuário {userId} falhou: {erro}")
        #     print(type(erro))
        #     time.sleep(10)
        #     continue

        candidatos_llm = llm.recommendations
        respostas_llm = llm.response

        # print(userId, len(historico))

        violacao_llm = int(not all(mid in resultado_k for mid in candidatos_llm) or len(candidatos_llm) != 3 or len(candidatos_llm) != len(set(candidatos_llm)))

        relevantes = {movieId for timestamp, movieId in gabarito[userId]}

        sasrec_hit_30 = define_metrics.hit_at_k(resultado_k, relevantes, 30)
        sasrec_ndcg_30 = define_metrics.ndcg_at_k(resultado_k, relevantes, 30)
        sasrec_recall_30 = define_metrics.recall_at_k(resultado_k, relevantes, 30)

        sasrec_hit_3 = define_metrics.hit_at_k(resultado_k, relevantes, 3)
        sasrec_ndcg_3 = define_metrics.ndcg_at_k(resultado_k, relevantes, 3)

        random_hit_3 = define_metrics.hit_at_k(resultado_3, relevantes, 3)
        random_ndcg_3 = define_metrics.ndcg_at_k(resultado_3, relevantes, 3)

        llm_hit_3 = define_metrics.hit_at_k(candidatos_llm, relevantes, 3)
        llm_ndcg_3 = define_metrics.ndcg_at_k(candidatos_llm, relevantes, 3)

        acertos_possiveis = relevantes & set(resultado_k)
        print(userId, acertos_possiveis, candidatos_llm)

        metrics_per_user_dict = {
            "userId": userId, 
            "topk": resultado_k, 
            "sasrec_hit@30": sasrec_hit_30, 
            "sasrec_ndcg@30": sasrec_ndcg_30, 
            "sasrec_recall@30": sasrec_recall_30,
            "sasrec_hit@3": sasrec_hit_3, 
            "sasrec_ndcg@3": sasrec_ndcg_3,
            "random_top3": resultado_3,
            "random_hit@3": random_hit_3,
            "random_ndcg@3": random_ndcg_3,
            "llm_top3": candidatos_llm,
            "llm_resposta": respostas_llm,
            "llm_hit@3": llm_hit_3,
            "llm_ndcg@3": llm_ndcg_3,
            "llm_violação": violacao_llm,
            "llm_custo": uso["cost"],
            "llm_reasoning_tokens": uso["completion_tokens_details"]["reasoning_tokens"],
            "llm_latência": latencia
        }

        with open(PASTA / "métricas_avaliadas_30.jsonl", "a", encoding="utf-8") as m:
            m.write(json.dumps(metrics_per_user_dict, ensure_ascii=False) + "\n")

    resultados = []

    with open(PASTA / "métricas_avaliadas_30.jsonl", "r", encoding="utf-8") as g:
        for row in g:
            resultados.append(json.loads(row))

    medias = {
        "sasrec_hit@30": sum(r["sasrec_hit@30"] for r in resultados) / len(resultados),
        "sasrec_ndcg@30": sum(r["sasrec_ndcg@30"] for r in resultados) / len(resultados),
        "sasrec_recall@30": sum(r["sasrec_recall@30"] for r in resultados) / len(resultados),
        "sasrec_hit@3": sum(r["sasrec_hit@3"] for r in resultados) / len(resultados),
        "sasrec_ndcg@3": sum(r["sasrec_ndcg@3"] for r in resultados) / len(resultados),
        "random_hit@3": sum(r["random_hit@3"] for r in resultados) / len(resultados),
        "random_ndcg@3": sum(r["random_ndcg@3"] for r in resultados) / len(resultados),
        "llm_hit@3": sum(r["llm_hit@3"] for r in resultados) / len(resultados),
        "llm_ndcg@3": sum(r["llm_ndcg@3"] for r in resultados) / len(resultados),
        "llm_violação": sum(r["llm_violação"] for r in resultados) / len(resultados),
        "llm_custo": sum(r["llm_custo"] for r in resultados) / len(resultados),
        "llm_reasoning_tokens": sum(r["llm_reasoning_tokens"] for r in resultados) / len(resultados),
        "llm_latência": sum(r["llm_latência"] for r in resultados) / len(resultados),
        "usuários": len(resultados)
    }

    print(medias)

    




