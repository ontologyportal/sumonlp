import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os



def plot_metrics(log_dir: str):
    metrics_csv = os.path.join(log_dir, "metrics.csv")

    if not os.path.exists(metrics_csv):
        print(f"Metrics file not found: {metrics_csv}")
        return

    metrics_df = pd.read_csv(metrics_csv)
    metrics_df = metrics_df[metrics_df["epoch"] > 0]

    # Plot Loss
    if "loss" in metrics_df.columns and "val_loss" in metrics_df.columns:
        trn_loss_df = metrics_df[["epoch", "loss"]].dropna()
        val_loss_df = metrics_df[["epoch", "val_loss"]].dropna()

        plt.figure()
        plt.plot(trn_loss_df["epoch"], trn_loss_df["loss"], label="Training")
        plt.plot(val_loss_df["epoch"], val_loss_df["val_loss"], label="Validation")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(log_dir, "loss.png"))
        plt.close()

        # Plot Loss
    if "train_sumo_penalty" in metrics_df.columns and "val_sumo_penalty" in metrics_df.columns:
        trn_loss_df = metrics_df[["epoch", "train_sumo_penalty"]].dropna()
        val_loss_df = metrics_df[["epoch", "val_sumo_penalty"]].dropna()

        plt.figure()
        plt.plot(trn_loss_df["epoch"], trn_loss_df["train_sumo_penalty"], label="Training")
        plt.plot(val_loss_df["epoch"], val_loss_df["val_sumo_penalty"], label="Validation")
        plt.title("Training and Validation SUMO Penalties")
        plt.xlabel("Epoch")
        plt.ylabel("Penalty")
        plt.legend()
        plt.savefig(os.path.join(log_dir, "sumo-term-penalties.png"))
        plt.close()

    # Plot BLEU Score
    if "val_bleu" in metrics_df.columns:
        bleu_df = metrics_df[["epoch", "val_bleu"]].dropna()

        plt.figure()
        plt.plot(bleu_df["epoch"], bleu_df["val_bleu"], marker="o", linestyle="-", label="BLEU Score")
        plt.title("Validation BLEU Score")
        plt.xlabel("Epoch")
        plt.ylabel("BLEU Score")
        plt.legend()
        plt.savefig(os.path.join(log_dir, "bleu_score.png"))
        plt.close()

    # Automatically find all ROUGE metrics in the CSV
    rouge_columns = [col for col in metrics_df.columns if "val_rouge" in col]

    if rouge_columns:
        plt.figure(figsize=(8, 6))

        for col in rouge_columns:
            rouge_df = metrics_df[["epoch", col]].dropna()
            plt.plot(rouge_df["epoch"], rouge_df[col], marker="o", linestyle="-", label=col)

        plt.title("Validation ROUGE Scores")
        plt.xlabel("Epoch")
        plt.ylabel("ROUGE Score")
        plt.legend()
        plt.savefig(os.path.join(log_dir, "rouge_scores.png"))  # Updated filename
        plt.close()

        print(f"Saved ROUGE score plots: {rouge_columns}")

    # Plot Exact Match Score
    if "val_exact_match" in metrics_df.columns:
        em_df = metrics_df[["epoch", "val_exact_match"]].dropna()

        plt.figure()
        plt.plot(em_df["epoch"], em_df["val_exact_match"], marker="^", linestyle="-", label="Exact Match Score")
        plt.title("Validation Exact Match Score")
        plt.xlabel("Epoch")
        plt.ylabel("Exact Match Score")
        plt.legend()
        plt.savefig(os.path.join(log_dir, "exact_match.png"))
        plt.close()

    # Plot Syntax Validity
    if "val_syntax_validity" in metrics_df.columns:
        syntax_df = metrics_df[["epoch", "val_syntax_validity"]].dropna()

        plt.figure()
        plt.plot(syntax_df["epoch"], syntax_df["val_syntax_validity"], marker="d", linestyle="-", label="Syntax Validity")
        plt.title("Validation Syntax Validity Score")
        plt.xlabel("Epoch")
        plt.ylabel("Syntax Validity Score")
        plt.legend()
        plt.savefig(os.path.join(log_dir, "syntax_validity.png"))
        plt.close()

        # Plot Syntax Validity
    if "sumo_terms_valid" in metrics_df.columns:
        syntax_df = metrics_df[["epoch", "sumo_terms_valid"]].dropna()

        plt.figure()
        plt.plot(syntax_df["epoch"], syntax_df["sumo_terms_valid"], marker="d", linestyle="-", label="Generated SUMO terms Validity")
        plt.title("Validation SUMO Generation Score")
        plt.xlabel("Epoch")
        plt.ylabel("SUMO Term Generation Validity Score")
        plt.legend()
        plt.savefig(os.path.join(log_dir, "sumo_term_validity.png"))
        plt.close()

    # Plot Learning Rate
    if "learning_rate" in metrics_df.columns:
        lr_df = metrics_df[["epoch", "learning_rate"]].dropna()

        plt.figure()
        plt.plot(lr_df["epoch"], lr_df["learning_rate"], marker="o", linestyle="-", label="Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.savefig(os.path.join(log_dir, "learning_rate.png"))
        plt.close()

    print(f"Metrics plots saved in {log_dir}")