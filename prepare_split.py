import json
import csv
from collections import defaultdict, Counter
from pathlib import Path

PASTA = Path("json/split")

def create_dict():
    dados = defaultdict(list)

    with open("data/ratings.csv", "r", encoding="utf-8") as f:
        data = csv.DictReader(f)

        for linha in data:       
            rating = float(linha["rating"])
            if rating < 3:
                continue

            userId = int(linha["userId"])        
            movieId = int(linha["movieId"])
            timestamp = int(linha["timestamp"])
            dados[userId].append((timestamp, movieId))

    # Ordenar por Timestamp
    for lista in dados.values():
        lista.sort()

    return dados

def filter_count(dados):
    while True:
        contagem_filmes = Counter(movieId for lista in dados.values() for timestamp, movieId in lista)

        filtrado = {}
        for userId, lista in dados.items():
            nova_lista = [(timestamp, movieId) for timestamp, movieId in lista if contagem_filmes[movieId] >= 5]
            if len(nova_lista) >= 5:
                filtrado[userId] = nova_lista

        if filtrado == dados:
            return filtrado

        dados = filtrado

def split_data(dados):
    entrada = {}
    gabarito = {}

    for userId, lista in dados.items():
        tamanho_teste = max(1, int(0.1 * (len(lista) - 1)))
        corte = len(lista) - tamanho_teste

        entrada[userId] = lista[:corte]
        gabarito[userId] = lista[corte:]

    return entrada, gabarito

def create_json(entrada, gabarito):
    PASTA.mkdir(parents=True, exist_ok=True)

    with open(PASTA / "entrada.json", "w", encoding="utf-8") as e:
        json.dump(entrada, e, indent=4, ensure_ascii=False)

    with open(PASTA / "gabarito.json", "w", encoding="utf-8") as g:
        json.dump(gabarito, g, indent=4, ensure_ascii=False)

def load_json():
    with open(PASTA / "entrada.json", "r", encoding="utf-8") as e:
        entrada_bruta = json.load(e)

    with open(PASTA / "gabarito.json", "r", encoding="utf-8") as g:
        gabarito_bruto = json.load(g)

    # Chave: texto -> int | Valor: cada par [timestamp, filme] -> (timestamp, filme)
    entrada = {int(chave): [tuple(par) for par in lista] for chave, lista in entrada_bruta.items()}
    gabarito = {int(chave): [tuple(par) for par in lista] for chave, lista in gabarito_bruto.items()}

    return entrada, gabarito


if __name__ == "__main__":
    dados = filter_count(create_dict())
    print("Usuários:", len(dados))
    print("Interações:", sum(len(lista) for lista in dados.values()))
    entrada, gabarito = split_data(dados)

    create_json(entrada, gabarito)

    print("Total gabarito:", sum(len(lista) for lista in gabarito.values()))

    usuario = 2

    # Teste 1: Tamanhos
    print("Total: ", len(dados[usuario]))
    print("Entrada: ", len(entrada[usuario]))
    print("Gabarito: ", len(gabarito[usuario]))
    print("Soma confere?: ", len(entrada[usuario]) + len(gabarito[usuario]) == len(dados[usuario]))

    # Teste 2: Fronteira Temporal
    ultimo_entrada = entrada[usuario][-1][0]
    primeiro_gabarito = gabarito[usuario][0][0]
    print("Último timestamp da entrada: ", ultimo_entrada)
    print("Primeiro timestamp do gabarito: ", primeiro_gabarito)
    print("Ordem confere?: ", ultimo_entrada <= primeiro_gabarito)
