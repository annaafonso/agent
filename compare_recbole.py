import prepare_split
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed
from collections import defaultdict

if __name__ == '__main__':
    # Configurations initialization
    config = Config(model='SASRec', dataset='ml-latest-small', config_file_list=['config.yaml'])
    # Init random seed
    init_seed(config['seed'], config['reproducibility'])

    init_logger(config)

    # Dataset creating
    dataset = create_dataset(config)

    # Dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    usuarios_teste = test_data.dataset.inter_feat["userId"]
    filmes_teste = test_data.dataset.inter_feat["movieId"]
    
    usuarios = dataset.id2token("userId", usuarios_teste)
    filmes = dataset.id2token("movieId", filmes_teste)

    gabarito_recbole = defaultdict(set)

    for userId, movieId in zip(usuarios, filmes):
        gabarito_recbole[int(userId)].add(int(movieId))

    entrada, gabarito = prepare_split.load_json()

    falhas = []

    for userId, lista in gabarito.items():
        filmes_gabarito = {movieId for timestamp, movieId in lista}

        if (gabarito_recbole[userId] != filmes_gabarito):
            falhas.append(userId)

    print(len(falhas))