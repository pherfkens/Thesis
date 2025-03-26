import torch.nn as nn
from transformers import ASTConfig, ASTForAudioClassification

def get_model(pretrained_model, num_outputs=8):
    config = ASTConfig.from_pretrained(pretrained_model)
    model = ASTForAudioClassification.from_pretrained(pretrained_model, config=config, ignore_mismatched_sizes=True)
    model.classifier = nn.Linear(config.hidden_size, num_outputs)
    model.init_weights()
    return model
