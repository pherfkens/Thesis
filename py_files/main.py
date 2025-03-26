from config import TRAINING_ARGS, PRETRAINED_MODEL, PAQS
from data import get_dataset
from model import get_model
from trainer import RegressionTrainer, compute_metrics

# Load dataset and model
dataset = get_dataset(PAQS, pretrained_model=PRETRAINED_MODEL)
print('function get_dataset ran')
model = get_model(PRETRAINED_MODEL, num_outputs=len(PAQS))
print('It is running now')
# Set up and run trainer
print('Starting trainer')
trainer = RegressionTrainer(
    model=model,
    args=TRAINING_ARGS,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)
print('Trainer started')
trainer.train()
trainer.predict(dataset["test"])
