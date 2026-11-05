import json
from pathlib import Path
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed

PASTA = Path("json/split")

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

    # Model loading and initialization
    model = SASRec(config, train_data.dataset).to(config['device'])

    # Trainer loading and initialization
    trainer = Trainer(config, model)

    # Model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # Model evaluation
    test_result = trainer.evaluate(test_data)
    print(test_result)

    test_result["modelo"] = trainer.saved_model_file

    PASTA.mkdir(parents=True, exist_ok=True)
    
    with open(PASTA / "test_result.json", "w", encoding="utf-8") as t:
        json.dump(test_result, t, indent=4, ensure_ascii=False)