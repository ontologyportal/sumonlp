import torch
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
import lightning.pytorch as pl
import re
from transformers import T5ForConditionalGeneration, AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig
from peft import LoraConfig, get_peft_model
import src.plot_utils as plot_utils
import subprocess

class L2LModel(pl.LightningModule):

    def __init__(self, **config):

        super().__init__()

        self.save_hyperparameters()

        model_name = config["model_name"]

        self.lr = config["lr"]
        self.weight_decay = config.get("weight_decay",0)
        sumo_terms_path = config.get("sumo_terms",'src/utils/sumo_terms.txt')
        self.warm_up_step = config.get("warm_up_step",0)
        self.sumo_term_penalty_weight = config.get("sumo_term_penalty_weight",0)
        self.use_pretrained = config["use_pretrained"]

        with open(sumo_terms_path, "r") as file:
            self.sumo_terms = {line.strip() for line in file if line.strip()}


        # LoRa
        # lora_r = config["lora_r"]
        # lora_alpha = config["lora_alpha"]
        # lora_dropout = config["lora_dropout"]

        # Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.use_pretrained:
          # Load a pretrained model configuration
          print(f"Loading pretrained model: {model_name}")
          self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
          # Load a model without weights.
          print(f"Loading model without weights: {model_name}")
          config = AutoConfig.from_pretrained(model_name)
          self.model = T5ForConditionalGeneration(config)

        self.bleu = BLEUScore(n_gram=4)
        # self.rouge = ROUGEScore(accumulate="avg")

        # Apply LoRA
        # peft_config = LoraConfig(
        #   r=lora_r,
        #   lora_alpha=lora_alpha,
        #   lora_dropout=lora_dropout,
        #   task_type="SEQ_2_SEQ_LM")

        # self.model = get_peft_model(self.model, peft_config)


    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def compute_exact_match(self, preds, targets):
        """Computes exact match (EM) accuracy."""
        return sum([p.strip() == t.strip() for p, t in zip(preds, targets)]) / len(preds)

    def check_valid_suo_kif(self, logic_str):
        """Checks if generated logic is well-formed (e.g., balanced parentheses)."""
        return logic_str.count("(") == logic_str.count(")")

    # def check_valid_suo_kif(self, logic_str):
    #     """Runs the SUO-KIF syntax checker and returns whether the statement is valid."""
    #     try:
    #         process = subprocess.run(
    #             ["bash", "src/utils/check_SUOKIF_syntax.sh"],
    #             input=logic_str.encode(),
    #             capture_output=True,
    #             text=True
    #         )
    #         return "Valid syntax" in process.stdout  # Check if output contains "Valid syntax"
    #     except Exception as e:
    #         print(f"Error running SUO-KIF checker: {e}")
    #         return False  # Assume invalid if script execution fails

    def evaluate_sumo_terms(self, sentence):
        """Extracts SUMO terms from a sentence using regex and filters only known SUMO terms."""
        word_pattern = re.compile(r"(?<!\?)\b[a-zA-Z_][a-zA-Z_0-9]*\b")
        extracted_terms = set(word_pattern.findall(sentence))
        return extracted_terms.issubset(self.sumo_terms)


    def penaltize_based_on_sumo_terms(self, sentence):
        """Extracts SUMO terms from a sentence and penalizes incorrect usage."""
        word_pattern = re.compile(r"(?<!\?)\b[a-zA-Z_][a-zA-Z_0-9]*\b")
        extracted_terms = set(word_pattern.findall(sentence))

        unknown_terms = extracted_terms - self.sumo_terms  # Find terms not in SUMO
        penalty = len(unknown_terms) / len(extracted_terms) if extracted_terms else 0

        return penalty  # 0 if all terms are correct, closer to 1 if many are wrong


    def training_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        loss = outputs.loss


        pred_texts = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        target_texts = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        # Compute SUMO term penalties
        sumo_penalties = torch.tensor([self.penaltize_based_on_sumo_terms(p) for p in pred_texts],
                                      dtype=torch.float32,
                                      device=self.device)

        sumo_penalty_loss = sumo_penalties.mean()  # Average penalty

        # Combine losses with a weighting factor (adjust factor as needed)
        loss = loss + (self.sumo_term_penalty_weight * sumo_penalty_loss)  # Penalize incorrect SUMO terms

        # Compute metrics
        # bleu_score = self.bleu(pred_texts, [[t] for t in target_texts])
        # rouge_score = self.rouge(pred_texts, target_texts)
        # em_score = self.compute_exact_match(pred_texts, target_texts)
        # valid_syntax = sum([self.check_valid_suo_kif(p) for p in pred_texts]) / len(pred_texts)

        # Log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        # Log metrics
        self.log_dict(
            {
                "loss": loss,
                # "train_bleu": bleu_score,
                # "train_rouge2_f1": rouge_score["rouge2_fmeasure"].item(),
                # "train_rougeL_f1": rouge_score["rougeL_fmeasure"].item(),
                # "train_exact_match": em_score,
                # "train_syntax_validity": valid_syntax,
                "train_sumo_penalty": sumo_penalty_loss.item(),
                "learning_rate": lr
            },
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        loss = outputs.loss

        pred_texts = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        target_texts = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        # Print only one sample per epoch (first batch, first sample)
        if batch_idx == 0:
            print(f"\n[VAL] Epoch {self.current_epoch+1}:")
            print(f"  Input: {self.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)}")
            print(f"  Target: {target_texts[0]}")
            print(f"  Prediction: {pred_texts[0]}")

        # Compute metrics
        bleu_score = self.bleu(pred_texts, [[t] for t in target_texts])
        # rouge_score = self.rouge(pred_texts, target_texts)
        em_score = self.compute_exact_match(pred_texts, target_texts)
        valid_syntax = sum([self.check_valid_suo_kif(p) for p in pred_texts]) / len(pred_texts)
        sumo_terms_valid = sum([self.evaluate_sumo_terms(p) for p in pred_texts]) / len(pred_texts)

        # Compute SUMO term penalties
        sumo_penalties = torch.tensor([self.penaltize_based_on_sumo_terms(p) for p in pred_texts],
                                      dtype=torch.float32,
                                      device=self.device)
        sumo_penalty_loss = sumo_penalties.mean()  # Average penalty

        # Log metrics
        self.log_dict(
            {
                "val_loss": loss,
                "val_bleu": bleu_score,
                # "val_rouge2_f1": rouge_score["rouge2_fmeasure"].item(),
                # "val_rougeL_f1": rouge_score["rougeL_fmeasure"].item(),
                "val_exact_match": em_score,
                "val_syntax_validity": valid_syntax,
                "sumo_terms_valid": sumo_terms_valid,
                "val_sumo_penalty": sumo_penalty_loss.item(),
            },
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return loss

    def on_validation_epoch_end(self):
        plot_utils.plot_metrics(self.logger.log_dir)

    def on_test_epoch_end(self):
        plot_utils.plot_metrics(self.logger.log_dir)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
          self.parameters(),
          lr=self.lr,
          weight_decay = self.weight_decay
        )

        if self.trainer.datamodule is not None:
            train_loader = self.trainer.datamodule.train_dataloader()
        else:
            raise ValueError("Trainer datamodule is not initialized. Ensure it is properly set in Trainer.")

        # Get total training steps
        num_training_steps = len(train_loader) * self.trainer.max_epochs
        print(f"Number of training steps per epoch are: {num_training_steps}")

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warm_up_step*num_training_steps,
            num_training_steps=num_training_steps
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
