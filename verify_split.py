import prepare_split

def verify_disjunction(entrada, gabarito):
    falhas = []

    for userId, lista in entrada.items():
        filmes_entrada = {movieId for timestamp, movieId in lista}
        filmes_gabarito = {movieId for timestamp, movieId in gabarito[userId]}

        if (filmes_entrada & filmes_gabarito):
            falhas.append((userId, filmes_entrada & filmes_gabarito))
        
    return falhas

def verify_temporal(entrada, gabarito):
    falhas = []

    for userId, lista in entrada.items():
        timestamp_entrada = max(timestamp for timestamp, movieId in lista)
        timestamp_gabarito = min(timestamp for timestamp, movieId in gabarito[userId])

        if (timestamp_entrada > timestamp_gabarito):
            falhas.append((userId, timestamp_entrada, timestamp_gabarito))

    return falhas

def verify_count(entrada, gabarito, contagem):
    falhas = []

    usuarios_entrada = set(entrada)
    usuarios_gabarito = set(gabarito)
    usuarios_csv = set(contagem)

    if not (usuarios_entrada == usuarios_gabarito == usuarios_csv):
        falhas.append(("Usuários diferentes", usuarios_csv - usuarios_entrada, usuarios_csv - usuarios_gabarito, usuarios_entrada - usuarios_csv))    
        return falhas   

    for userId, total in contagem.items():
        soma = len(entrada[userId]) + len(gabarito[userId])
        if soma != total:
            falhas.append(("Contagem diferente", userId, soma, total))

    return falhas

if __name__ == "__main__":
    entrada, gabarito = prepare_split.load_json()

    falhas = verify_disjunction(entrada, gabarito)
    print("Falhas de disjunção:", len(falhas))

    falhas_temporal = verify_temporal(entrada, gabarito)
    print("Falhas de ordem temporal:", len(falhas_temporal))

    dados = prepare_split.filter_count(prepare_split.create_dict())
    contagem = {userId: len(lista) for userId, lista in dados.items()}
    print("Total de interações filtradas:", sum(contagem.values()))
    falhas_contagem = verify_count(entrada, gabarito, contagem)
    print("Falhas de contagem:", len(falhas_contagem))

    gabaritos_vazios = sum(1 for lista in gabarito.values() if not lista)
    print("Gabaritos vazios:", gabaritos_vazios)