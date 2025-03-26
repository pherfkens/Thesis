import pandas as pd
import torch
import torchaudio
from datasets import Dataset, Audio
from transformers import ASTFeatureExtractor

def get_dataset(label_keys, pretrained_model):
    source_dir = "C:/Users/pepij/OneDrive - Delft University of Technology/THESIS/data/csv"
    metadata = pd.read_csv(f"{source_dir}/first_try.csv")
    metadata[label_keys] = metadata[label_keys] / 5.0
    # metadata = metadata.iloc[0:100] # Only first 100 for testing
    print('data loaded')

    # def load_audio(row):
    #     waveform, sr = torchaudio.load(row['audio_path'])
    #     return {
    #         'filename': f"{row['GroupID']}.wav",
    #         'labels': {k: row[k] for k in label_keys},
    #         'audio': {
    #             'path': row['audio_path'],
    #             'array': waveform.squeeze().numpy().reshape(-1),
    #             'sampling_rate': sr
    #         }
    #     }
    
    def load_audio(row):
        return {
            'filename': f"{row['GroupID']}.wav",
            'labels': {k: row[k] for k in label_keys},
            'audio': row['audio_path']  # just the path, no loading
        }



    # records = metadata.apply(load_audio, axis=1)
    records = [load_audio(row) for _, row in metadata.iterrows()]
    df = pd.DataFrame(records)
    print('start creating dataset')
    # dataset = Dataset.from_pandas(df).cast_column("audio", Audio(sampling_rate=16000))
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    print('dataset created and casted')

    feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)
    SAMPLING_RATE = feature_extractor.sampling_rate
    model_input_name = feature_extractor.model_input_names[0]

    def preprocess_audio(batch):
        wavs = [audio["array"] for audio in batch["input_values"]]
        inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        labels = [[row[k] for k in label_keys] for row in batch["labels"]]
        return {model_input_name: inputs[model_input_name], "labels": torch.tensor(labels, dtype=torch.float32)}

    if "test" not in dataset:
        dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=0)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    dataset = dataset.rename_column("audio", "input_values")
    dataset["train"].set_transform(preprocess_audio, output_all_columns=False)
    dataset["test"].set_transform(preprocess_audio, output_all_columns=False)

    return dataset
