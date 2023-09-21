import yaml
import argparse
import pandas as pd
from typing import Text

from src.utils.logs import get_logger


def feature_engineering(config_path: Text) -> None:
    """Feature engineering over iris dataset
    Params
    ------
    config_path : Text
        Path to configuration file
    """
    config = yaml.safe_load(open(config_path))
    logger = get_logger("FEATURE_ENGINEERING", log_level=config["base"]["log_level"])
    # Get the data
    logger.info("Load dataset locally")
    dataset = pd.read_csv(config["data_load"]["dataset_csv"])
    # Perform feature engineering
    logger.info("Create features")
    dataset["sepal_length_to_sepal_width"] = (
        dataset["sepal_length"] / dataset["sepal_width"]
    )
    dataset["petal_length_to_petal_width"] = (
        dataset["petal_length"] / dataset["petal_width"]
    )

    dataset = dataset[
        [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            #     'sepal_length_in_square', 'sepal_width_in_square', 'petal_length_in_square', 'petal_width_in_square',
            "sepal_length_to_sepal_width",
            "petal_length_to_petal_width",
            "target",
        ]
    ]
    # Save dataset with new features
    logger.info("Save dataset with new features")
    dataset.to_csv(config["featurize"]["features_path"], index=False)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    feature_engineering(config_path=args.config)
