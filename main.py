import os
from google import genai
from google.genai import types
import agent, sasrec
 
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "Variável de ambiente GEMINI_API_KEY não encontrada. "
    )
client = genai.Client(api_key=api_key)

def main():
    # Carrega o modelo SASRec
    config, model, dataset = sasrec.load_model()

    userId = int(input("Digite o ID do usuário: "))
    print("Conversa iniciada. Digite 'Sair' para encerrar.\n")

    while True:
        mensagem = input("Você: ")

        if mensagem.lower() == 'sair':
            print("Conversa encerrada.")
            break

        agent.save_history(userId, 'usuario', mensagem)
        prompt = agent.agent_instruction(userId, config, model, dataset)

        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=mensagem,
            config=types.GenerateContentConfig(
                system_instruction=prompt
            )
        )

        resposta= response.text
        agent.save_history(userId, 'assistente', resposta)

        print(f"Agente: {resposta}\n")

if __name__ == '__main__':
    main()