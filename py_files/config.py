from transformers import TrainingArguments

PAQS = ["pleasant", "chaotic", "vibrant", "uneventful", "calm", "annoying", "eventful", "monotonous"]
PRETRAINED_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

TRAINING_ARGS = TrainingArguments(
    output_dir="./runs/ast_regressor",
    logging_dir="./logs/ast_regressor",
    report_to="tensorboard",
    learning_rate=5e-5,
    warmup_ratio=0.1,
    push_to_hub=False,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="rmse",
    greater_is_better=False,
    logging_strategy="steps",
    logging_steps=5,
    remove_unused_columns=False,
    seed=0
)
