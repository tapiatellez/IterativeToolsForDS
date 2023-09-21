import yaml
import argparse
import pandas as pd
from typing import Text

from src.utils.logs import get_logger
from sklearn.model_selection import train_test_split


def split_data(config_path: Text) -> None:
    """Split data into training and testing
    Params
    ------
    config_path : Text
        Path to configuration file
    """
    config = yaml.safe_load(open(config_path))
    logger = get_logger("SPLIT_DATA", log_level=config["base"]["log_level"])
    # Get the data
    logger.info("Load dataset with features")
    dataset = pd.read_csv(config["featurize"]["features_path"])
    # Split the data
    logger.info("Split the dataset")
    random_state = config["base"]["seed"]
    test_size = config["data_split"]["test_size"]

    train_dataset, test_dataset = train_test_split(
        dataset, test_size=test_size, random_state=random_state
    )

    # Save the data
    logger.info("Save train and test dataset")
    train_dataset.to_csv(config["data_split"]["trainset_path"])
    test_dataset.to_csv(config["data_split"]["testset_path"])


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    split_data(config_path=args.config)
