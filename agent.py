import os
import json
import sqlite3
import sasrec
import requests
# from google import genai
# from google.genai import types
from pydantic import BaseModel, ConfigDict, Field
from typing import List

class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    recommendations: List[int] = Field(description="MovieIDs dos 3 filmes escolhidos entre os candidatos, do mais indicado para o menos.")
    response: str = Field(description="Mensagem amigável para o usuário.")

# client = genai.Client()

def load_history(userId):
    try:
        with sqlite3.connect('database.db') as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT papel, mensagem FROM history WHERE userId = ? ORDER BY id DESC LIMIT 20", (userId,))
            historico = cursor.fetchall()
            historico.reverse()
            return historico
    except sqlite3.Error as e:
        print(f"Erro ao resgatar histórico de conversas: {e}")
        return []

def load_history_movie_ids(userId):
    try:
        with sqlite3.connect('database.db') as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT movieId FROM ratings
                WHERE userId = ? AND rating >= 3
                ORDER BY timestamp ASC, movieId ASC
            """, (userId,))
            return [row[0] for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"Erro ao resgatar filmes assistidos: {e}")
        return []

def load_return_topk(historico_ids, config, model, dataset):
    lista = sasrec.return_topk(historico_ids, config, model, dataset)

    if lista is None:
        return []

    return lista

def get_titles_map(movies_ids):
    if movies_ids is None or len(movies_ids) == 0:
        return {}

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    ids = [int(mid) for mid in movies_ids]
    placeholders = ','.join('?' * len(ids))
    cursor.execute(f"SELECT movieId, title FROM movies WHERE movieId IN ({placeholders})", ids)
    mapa_titulos = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()

    return mapa_titulos

def get_titles(movies_ids):
    mapa_titulos = get_titles_map(movies_ids)
    return [mapa_titulos[int(mid)] for mid in movies_ids if int(mid) in mapa_titulos]

def save_history(userId, papel, mensagem):
    try:
        with sqlite3.connect('database.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO history (userId, papel, mensagem) VALUES (?, ?, ?)",
                (userId, papel, mensagem)
            )
            conn.commit()
    except sqlite3.Error as e:
        print(f"Erro ao salvar mensagem: {e}")

def agent_instruction(userId, historico_ids, candidatos_ids, conversa):
    historico_format = '\n'.join(f'{papel}: {mensagem}' for papel, mensagem in conversa)

    recentes = list(reversed(historico_ids[-20:]))
    historico_filmes_format = ', '.join(get_titles(recentes))

    mapa_candidatos = get_titles_map(candidatos_ids)
    candidatos_format = '\n'.join(f'ID {mid}: {mapa_candidatos[mid]}' for mid in candidatos_ids if mid in mapa_candidatos)

    agent_prompt = f"""
    Você é um Sistema de Recomendação Conversacional de filmes baseado no dataset MovieLens.
    Você está conversando com o Usuário ID {userId}.

    Aqui está o histórico de filmes que este usuário já assistiu e gostou no passado (vindo do banco de dados): {historico_filmes_format}

    Aqui está o histórico da conversa com esse Usuário até agora: {historico_format}

    Aqui está uma lista de até 30 candidatos gerada por um modelo sequencial de recomendação (SASRec), treinado sobre o histórico de interações de todos os usuários, no formato "ID <movieId>: Título". O número após "ID" é o identificador do filme, não a sua posição na lista:
    {candidatos_format}

    Regras:
    1. Recomende 3 filmes, com o filme mais indicado sendo o primeiro a ser informado.
    2. A recomendação final que você der ao usuário tem que ser um filme que está na lista de candidatos do SASRec acima.
    3. Nunca recomende um filme que já esteja na lista de histórico acima, mesmo que ele apareça por engano entre os candidatos do SASRec.
    4. Use o histórico do usuário e o que ele disser na conversa apenas para escolher qual candidato da lista é o mais adequado e para explicar o porquê da escolha.
    5. Se a lista de candidatos do SASRec estiver vazia, informe isso ao usuário de forma transparente, não tente compensar recomendando um filme por conta própria e deixe a lista de recomendações vazia.
    6. Mantenha um tom amigável e conversacional. Se o usuário fizer perguntas de acompanhamento, responda de forma contextualizada, sempre respeitando as regras acima.
    7. No campo de recomendações, informe os movieId exatamente como aparecem após "ID" na lista de candidatos (nunca a posição do filme na lista), na mesma ordem em que os filmes aparecem na sua mensagem.
    """

    return agent_prompt

def ask_agent(prompt, mensagem):

    resposta = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "model": "google/gemini-3.5-flash",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": mensagem}
            ],
            "max_tokens": 8000,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "recommendation",
                    "strict": True,
                    "schema": Recommendation.model_json_schema()
                }
            },
            "provider": {
                "require_parameters": True
            }
        }),
        timeout=100
    )

    resposta.raise_for_status()

    conteudo = resposta.json()["choices"][0]["message"]["content"]
    uso = resposta.json()["usage"]

    return Recommendation.model_validate_json(conteudo), uso

    """
    resposta = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=mensagem,
        config=types.GenerateContentConfig(
            system_instruction=prompt,
            response_mime_type="application/json",
            response_schema=Recommendation,
        ),
    )
    """

    # print(resposta.text)

    # return Recommendation.model_validate_json(resposta.text)

if __name__ == "__main__":
    import prepare_split

    config, model, dataset = sasrec.load_model()
    entrada, gabarito = prepare_split.load_json()

    userId = 1
    historico_ids = [movieId for timestamp, movieId in entrada[userId]]
    candidatos_ids = load_return_topk(historico_ids, config, model, dataset)

    prompt = agent_instruction(userId, historico_ids, candidatos_ids, [])
    recomendacao, uso = ask_agent(prompt, "Oi")

    print("Candidatos:", candidatos_ids)
    print("Recomendações:", recomendacao.recommendations)
    print("Resposta:", recomendacao.response)
    print("Todas nos candidatos?", all(mid in candidatos_ids for mid in recomendacao.recommendations))