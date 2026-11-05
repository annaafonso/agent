import csv
import json
import prepare_split, sasrec, define_metrics, agent
from pathlib import Path

PASTA = Path("json/split")

def load_movies():
    dados = {}

    with open("data/movies.csv", "r", encoding="utf-8") as f:
        data = csv.DictReader(f)

        for linha in data:           
            movieId = int(linha["movieId"])
            title = linha["title"]
            genres = linha["genres"].split('|')
            dados[movieId] = (title, genres)

    return dados

def return_split(userId, entrada, gabarito, filmes):
    lista_entrada = entrada[userId][-20:] # ID
    titulos_entrada = [filmes[movieId][0] for timestamp, movieId in lista_entrada]

    lista_gabarito = gabarito[userId] # ID
    titulos_gabarito = [filmes[movieId][0] for timestamp, movieId in lista_gabarito]

    return titulos_entrada, titulos_gabarito

def return_candidates(userId, gabarito, filmes, topk):
    relevantes = {movieId for timestamp, movieId in gabarito[userId]}

    resultados = []

    for pos, movieId in enumerate(topk, start=1):
        filme = filmes[movieId][0]
        acerto = movieId in relevantes
        resultados.append((pos, movieId, filme, acerto))

    return resultados

def return_choices(userId, recomendacoes, topk, gabarito, filmes):
    relevantes = {movieId for timestamp, movieId in gabarito[userId]}

    resultados = []

    for ordem, movieId in enumerate(recomendacoes, start=1):
        if movieId in topk:
            posicao = topk.index(movieId) + 1
        else:
            posicao = None

        if movieId in filmes:
            filme = filmes[movieId][0]
        else:
            filme = None

        acerto = movieId in relevantes
        resultados.append((ordem, movieId, filme, posicao, acerto))

    return resultados

if __name__ == "__main__":
    userId = 3
    filmes = load_movies()
    entrada, gabarito = prepare_split.load_json()
    lista_entrada, lista_gabarito = return_split(userId, entrada, gabarito, filmes)
    print(lista_entrada)
    print() # Espaço em branco
    print(lista_gabarito)

    config, model, dataset = sasrec.load_model()

    historico = [movieId for timestamp, movieId in entrada[userId]]
    topk = sasrec.return_topk(historico, config, model, dataset)
    
    candidatos = return_candidates(userId, gabarito, filmes, topk)
    for c in candidatos:
        print(c)

    print()
    prompt = agent.agent_instruction(userId, historico, topk, [])
    print(prompt)

    recomendacao, uso = agent.ask_agent(prompt, "Oi")

    print(recomendacao.response)
    print()
    print(uso)
    print()

    escolhas = return_choices(userId, recomendacao.recommendations, topk, gabarito, filmes)
    for e in escolhas:
        print(e)