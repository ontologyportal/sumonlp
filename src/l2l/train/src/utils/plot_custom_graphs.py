import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

sumo_nlp_home = os.environ.get('SUMO_NLP_HOME')
l2l_home = os.path.join(sumo_nlp_home, 'src', 'l2l','train')

csv_path = l2l_home+"/out/7-12mil-10-epochs/lightning_logs/version_0/metrics.csv"
log_dir = l2l_home+"/src/utils/custom_plots"

def plot_metrics(log_dir: str, csv_path, max_y: float = 0.002):
    metrics_csv = csv_path

    if not os.path.exists(metrics_csv):
        print(f"Metrics file not found: {metrics_csv}")
        return

    metrics_df = pd.read_csv(metrics_csv)
    metrics_df = metrics_df[metrics_df["epoch"] >= 0]

    if "loss" in metrics_df.columns and "val_loss" in metrics_df.columns:
        # Filter out rows where epoch is 0
        metrics_df = metrics_df[metrics_df["epoch"] >= 0]

        trn_loss_df = metrics_df[["epoch", "loss"]].dropna()
        val_loss_df = metrics_df[["epoch", "val_loss"]].dropna()

        plt.figure()
        plt.plot(trn_loss_df["epoch"], trn_loss_df["loss"], label="Training")
        plt.plot(val_loss_df["epoch"], val_loss_df["val_loss"], label="Validation")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

                # Set maximum Y value if specified
        if max_y is not None:
            plt.ylim(top=max_y)

        plt.savefig(os.path.join(log_dir, "loss.png"))
        plt.close()

    print(f"Metrics plots saved in {log_dir}")

plot_metrics(log_dir,csv_path)