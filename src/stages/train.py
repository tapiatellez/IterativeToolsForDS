import yaml
import argparse
import pandas as pd
import joblib
from typing import Text

from src.utils.logs import get_logger

from src.train.train import train


def train_model(config_path: Text) -> None:
    """Train model
    Params
    ------
    config_path : Text
        Path to configuration file
    """
    # Get the configuration file
    config = yaml.safe_load(open(config_path))
    logger = get_logger("TRAIN_MODEL", log_level=config["base"]["log_level"])
    # Get the training dataset
    logger.info("Get the training dataset")
    train_dataset = pd.read_csv(config["data_split"]["trainset_path"])

    # Train (Use function from separate module for reusability)
    model = train(
        df=train_dataset,
        target_column=config["featurize"]["target_column"],
        estimator_name=config["train"]["estimator_name"],
        param_grid=config["train"]["estimators"]["logreg"]["param_grid"],
        cv=config["train"]["cv"],
    )
    logger.info(f"Best score: {model.best_score_}")
    logger.info("Save model")
    model_path = config["train"]["model_path"]
    joblib.dump(model, model_path)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)
