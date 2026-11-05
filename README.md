# Recomendador Conversacional de Filmes (SASRec + LLM)

Projeto de Iniciação Científica (PIBIC) sobre um **Sistema de Recomendação Conversacional** baseado em LLMs e Engenharia de Contexto.

A ideia é juntar duas etapas:

1. **Retrieval (busca de candidatos):** o modelo **SASRec** olha a sequência de filmes que o usuário assistiu e sugere uma lista com os 30 filmes mais prováveis (Top-K).
2. **Ranking (escolha final):** um **LLM** (Gemini 3.5 Flash, acessado pelo OpenRouter) recebe essa lista e o histórico do usuário e escolhe **3 filmes**, explicando o motivo em uma mensagem amigável.

O cenário avaliado é simples: o usuário manda apenas **"Oi"**, e o sistema recomenda com base só no histórico dele.

---

## Dados

O projeto usa o **MovieLens Small** (`ml-latest-small`), com cerca de 100 mil avaliações de 610 usuários.

Os dados **não estão no repositório**, porque a licença do MovieLens não permite redistribuí-los. Para usar o projeto:

1. Baixe o arquivo `ml-latest-small.zip` em: https://grouplens.org/datasets/movielens/latest/
2. Descompacte e coloque `movies.csv` e `ratings.csv` dentro de uma pasta `data/` na raiz do projeto.

Regras aplicadas aos dados:

- Só contam as avaliações com **nota ≥ 3** (o usuário "gostou").
- Filtro **5-core**: ficam só filmes e usuários com pelo menos 5 interações. Depois disso, restam **608 usuários**.
- Divisão **80% treino / 10% validação / 10% teste**, respeitando a ordem do tempo (os filmes de teste são sempre os mais recentes de cada usuário).

---

## Instalação

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

O LLM é acessado pelo OpenRouter. Antes de rodar os scripts que usam o LLM, defina a chave como variável de ambiente (ela **nunca** deve ser escrita no código):

```bash
export OPENROUTER_API_KEY="sua-chave-aqui"
```

---

## Ordem de execução

| Passo | Comando | O que faz |
|---|---|---|
| 1 | `python database.py` | Cria o banco `database.db` |
| 2 | `python prepare_atomic_files.py` | Prepara os dados no formato do RecBole |
| 3 | `python train_model.py` | Treina o SASRec |
| 4 | `python prepare_split.py` | Separa histórico e gabarito de cada usuário |
| 5 | `python verify_split.py` e `python compare_recbole.py` | Confere se a divisão está correta |
| 6 | `python sasrec.py` | Avalia só o SASRec |
| 7 | `python evaluate_turns.py` | Avalia o sistema completo (SASRec + LLM) |
| 8 | `python analyze_results.py` | Testes estatísticos |
| 9 | `python trace_users.py` | Análise detalhada de um usuário |

---

## O que cada arquivo faz

### Preparação dos dados

**`database.py`**
Cria o banco de dados SQLite (`database.db`) com três tabelas: `movies` (filmes), `ratings` (avaliações) e `history` (mensagens das conversas). Lê os dados de `data/movies.csv` e `data/ratings.csv`.

**`prepare_atomic_files.py`**
Lê as avaliações do banco e salva no formato que o RecBole entende (arquivo `.inter`), dentro de `dataset/ml-latest-small/`. As avaliações são ordenadas pelo tempo.

**`config.yaml`**
Configurações do RecBole: onde estão os dados, os filtros (nota ≥ 3 e 5-core), a divisão 80-10-10 em ordem temporal, o tamanho máximo do histórico (50 filmes) e as métricas do treino.

**`prepare_split.py`**
Monta, para cada usuário, duas listas:
- **entrada:** os filmes que o sistema pode usar como histórico;
- **gabarito:** os filmes mais recentes, que o sistema deveria acertar.

Aplica os mesmos filtros do RecBole e salva tudo em `json/split/` (`entrada.json` e `gabarito.json`). O tamanho do teste segue a regra do RecBole: 10% das interações do usuário, sem contar a primeira (que não tem histórico antes dela), com no mínimo 1 filme.

**`verify_split.py`**
Confere se a divisão está correta:
- nenhum filme aparece ao mesmo tempo na entrada e no gabarito;
- o gabarito vem sempre **depois** da entrada no tempo;
- a soma de entrada + gabarito bate com o total de cada usuário;
- nenhum usuário ficou com gabarito vazio.

**`compare_recbole.py`**
Compara o gabarito montado em `prepare_split.py` com o conjunto de teste que o próprio RecBole separou. Mostra quantos usuários têm diferença (o esperado é zero).

### Modelo SASRec

**`train_model.py`**
Treina o SASRec com o RecBole, usando o `config.yaml`. O modelo treinado fica salvo na pasta `saved/`, e o resultado do teste vai para `json/split/test_result.json`.

**`sasrec.py`**
- `load_model()`: carrega o modelo treinado.
- `return_topk()`: recebe o histórico de um usuário e devolve os **30 filmes** mais recomendados, ignorando os que ele já assistiu.

Ao ser executado direto, calcula Hit@30, NDCG@30 e Recall@30 para todos os usuários e salva os resultados em `json/results/`.

> Atenção: o nome do arquivo do modelo está escrito dentro de `load_model()`. Se o modelo for treinado de novo, é preciso atualizar esse nome.

**`define_metrics.py`**
Funções das métricas de avaliação:
- **Hit@K:** acertou pelo menos um filme entre os K primeiros? (1 ou 0)
- **Recall@K:** quantos filmes do gabarito apareceram entre os K primeiros?
- **NDCG@K:** além de acertar, leva em conta a **posição** do acerto (acertar em 1º vale mais que em 10º).

### Agente (LLM)

**`agent.py`**
O "cérebro" conversacional do sistema:
- `agent_instruction()`: monta o **prompt** com os 20 filmes mais recentes do usuário, o histórico da conversa, os 30 candidatos do SASRec e as regras (recomendar 3 filmes, só da lista de candidatos, nunca repetir filmes já vistos etc.).
- `ask_agent()`: envia o prompt ao Gemini pelo OpenRouter e recebe a resposta em formato estruturado: os 3 `movieId` escolhidos e a mensagem para o usuário.
- Outras funções buscam títulos de filmes e salvam ou carregam o histórico da conversa no banco.

A mesma função de prompt é usada na avaliação e no uso real, para garantir que os dois funcionem do mesmo jeito.

**`main.py`**
Programa de conversa no terminal. O usuário digita o seu ID e conversa com o agente:
1. busca no banco os filmes que o usuário já avaliou bem;
2. o SASRec gera os 30 candidatos (uma vez, no início da conversa);
3. a cada mensagem, monta o prompt com o histórico da conversa e envia ao LLM pelo OpenRouter;
4. mostra a resposta e salva as duas mensagens no banco, para o agente lembrar da conversa.

Digite `Sair` para encerrar. Precisa da variável `OPENROUTER_API_KEY` definida.

### Avaliação e análise

**`evaluate_turns.py`**
Avaliação principal do sistema completo, no cenário do "Oi". Para cada usuário:
1. o SASRec gera os 30 candidatos;
2. o LLM escolhe 3;
3. compara três estratégias de escolha dos 3 filmes:
   - **SASRec:** os 3 primeiros da lista;
   - **Aleatória:** 3 sorteados da lista;
   - **LLM:** os 3 escolhidos pelo modelo.

Registra também se o LLM violou alguma regra, o custo, os tokens de raciocínio e o tempo de resposta. Os resultados vão para `json/results/`. Se o script parar no meio, ele continua de onde parou, pulando os usuários já avaliados.

**`analyze_results.py`**
Verifica se a diferença entre LLM e SASRec é real ou pode ser acaso:
- **Teste de McNemar:** compara os acertos dos dois, usuário por usuário.
- **Bootstrap:** calcula o intervalo de confiança (95%) da diferença de Hit@3 e NDCG@3.

**Nomes dos arquivos de resultados (`json/results/`)**
O final do nome indica quantos candidatos o SASRec gerou (o K do Top-K):
- sem final (ex.: `métricas_avaliadas.jsonl`): **Top-20**;
- com `_30` (ex.: `métricas_avaliadas_30.jsonl`): **Top-30**;
- com `_40`: **Top-40**.

**`trace_users.py`**
Análise detalhada de **um usuário**, do "Oi" até a recomendação final. Mostra:
1. os 20 filmes recentes do histórico e os filmes do gabarito;
2. os 30 candidatos do SASRec, com a posição e se cada um está no gabarito;
3. o prompt completo enviado ao LLM;
4. a resposta do LLM e o uso de tokens;
5. os 3 filmes escolhidos, com a posição de cada um na lista do SASRec e se acertou.

Para analisar outro usuário, basta mudar a variável `userId` no início do bloco principal.

### Outros

**`requirements.txt`**
Bibliotecas necessárias. Algumas têm versão limitada (`numpy<2.0`, `scipy<1.14`, `torch<2.6`) para funcionar com o RecBole.

**`.gitignore`**
Lista o que não vai para o GitHub: ambiente virtual, modelo treinado, dados do MovieLens, banco de dados, logs e o arquivo `.env`.