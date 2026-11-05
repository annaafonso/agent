import os
import requests
import agent
import sasrec
from pydantic import ValidationError

if not os.environ.get("OPENROUTER_API_KEY"):
    raise RuntimeError("Variável de ambiente OPENROUTER_API_KEY não encontrada.")

def main():
    config, model, dataset = sasrec.load_model()

    userId = int(input("Digite o ID do usuário: "))
    historico_ids = agent.load_history_movie_ids(userId)
    candidatos_ids = agent.load_return_topk(historico_ids, config, model, dataset)

    print("Conversa iniciada. Digite 'Sair' para encerrar.\n")

    while True:
        mensagem = input("Você: ")

        if mensagem.lower() == 'sair':
            print("Conversa encerrada.")
            break

        conversa = agent.load_history(userId)
        prompt = agent.agent_instruction(userId, historico_ids, candidatos_ids, conversa)

        try:
            recomendacao, uso = agent.ask_agent(prompt, mensagem)
        except requests.exceptions.RequestException as erro:
            print(f"Erro ao contatar o agente: {erro}\n")
            continue
        except ValidationError as erro:
            print(f"Resposta malformada do agente: {erro}\n")
            continue

        resposta = recomendacao.response
        agent.save_history(userId, 'usuario', mensagem)
        agent.save_history(userId, 'assistente', resposta)

        print(f"Agente: {resposta}\n")

if __name__ == '__main__':
    main()