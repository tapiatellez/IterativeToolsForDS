import yaml
import argparse
from sklearn.datasets import load_iris
from typing import Text

from src.utils.logs import get_logger


def data_load(config_path: Text) -> None:
    config = yaml.safe_load(open(config_path))
    logger = get_logger("DATA_LOAD", log_level=config["base"]["log_level"])
    # Get the data
    logger.info("Get Dataset")
    data = load_iris(as_frame=True)
    dataset = data.frame
    # Rename the rows
    dataset.columns = [
        colname.strip(" (cm)").replace(" ", "_") for colname in dataset.columns.tolist()
    ]
    # Save raw data
    logger.info("Save raw data.")
    dataset.to_csv(config["data_load"]["dataset_csv"], index=False)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)
