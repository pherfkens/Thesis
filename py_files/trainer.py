import torch.nn as nn
from transformers import Trainer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.MSELoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    mse_value = mean_squared_error(labels, logits)
    rmse_value = np.sqrt(mse_value)
    mae_value = mean_absolute_error(labels, logits)
    r2_value = r2_score(labels, logits)
    return {
        "mse": mse_value,
        "rmse": rmse_value,
        "mae": mae_value,
        "r2": r2_value
    }
