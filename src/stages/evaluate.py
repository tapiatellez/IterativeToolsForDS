import yaml
import argparse
import pandas as pd
import joblib
import json
from typing import Text
from pathlib import Path
from src.utils.logs import get_logger
from sklearn.metrics import confusion_matrix, f1_score
from src.report.visualize import plot_confusion_matrix
from sklearn.datasets import load_iris


def evaluate_model(config_path: Text) -> None:
    """Evaluate the model
    Params
    ------
    config_path : Text
        Path to configuration file
    """
    # Get the configuration file
    config = yaml.safe_load(open(config_path))
    logger = get_logger("TRAIN_MODEL", log_level=config["base"]["log_level"])
    # Get the training dataset
    logger.info("Get the test dataset")
    test_dataset = pd.read_csv(config["data_split"]["testset_path"])
    # Load the model
    logger.info("Load the trained model")
    model = joblib.load(config["train"]["model_path"])
    # Get predictions and metrics
    logger.info("Get the predictions and metrics")
    y_test = test_dataset.loc[:, "target"].values.astype("int32")
    X_test = test_dataset.drop("target", axis=1).values.astype("float32")
    prediction = model.predict(X_test)
    cm = confusion_matrix(prediction, y_test)
    f1 = f1_score(y_true=y_test, y_pred=prediction, average="macro")
    report = {
        "f1": f1,
        "confusion_matrix": cm,
        "actual_values": y_test,
        "predicted": prediction,
    }
    # Save the metrics
    logger.info("Save the metrics")
    reports_folder = Path(config["evaluate"]["reports_dir"])
    metrics_path = reports_folder / config["evaluate"]["metrics_file"]

    json.dump(obj={"f1_score": report["f1"]}, fp=open(metrics_path, "w"))

    logger.info(f"F1 metrics file saved to : {metrics_path}")

    logger.info("Save confusion matrix")
    # save confusion_matrix.png
    plt = plot_confusion_matrix(
        cm=report["confusion_matrix"],
        target_names=load_iris(as_frame=True).target_names.tolist(),
        normalize=False,
    )
    confusion_matrix_png_path = (
        reports_folder / config["evaluate"]["confusion_matrix_image"]
    )
    plt.savefig(confusion_matrix_png_path)
    logger.info(f"Confusion matrix saved to : {confusion_matrix_png_path}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)
