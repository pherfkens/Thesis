import os
import pickle
import numpy as np

def load_lines(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_feature_npy(feature_dir, audio_ids):
    feature_dict = {}
    for audio_id in audio_ids:
        npy_path = os.path.join(feature_dir, f"{audio_id}.npy")
        if os.path.exists(npy_path):
            data = np.load(npy_path, allow_pickle=True)

            # Handle case where data is string with comma as decimal separator
            if isinstance(data, np.ndarray) and data.dtype == object:
                try:
                    if data.ndim == 2 and data.shape[1] == 1:
                        # Convert and preserve (N, 1) shape
                        float_data = [[float(x[0].replace(',', '.'))] for x in data]
                        data = np.array(float_data, dtype=float)
                    elif data.ndim == 1:
                        # Convert and reshape to (N, 1)
                        float_data = [[float(x.replace(',', '.'))] for x in data]
                        data = np.array(float_data, dtype=float)
                except Exception as e:
                    print(f"Warning: Failed to convert string to float for {audio_id}: {e}")
                    continue

            feature_dict[audio_id] = data
        else:
            print(f"Warning: Missing feature for {audio_id} in {feature_dir}")
    return feature_dict


def generate_feature_pickle(audio_id_path, npy_dir, out_pickle_path):
    audio_ids = load_lines(audio_id_path)
    feature_dict = load_feature_npy(npy_dir, audio_ids)
    with open(out_pickle_path, 'wb') as f:
        pickle.dump(feature_dict, f)
    print(f"Saved: {out_pickle_path}")

# ==== MAIN EXECUTION ====
# Paths for each dataset split
DATASET_DIR = r'C:\Users\pepij\OneDrive - Delft University of Technology\SoundSCaper\Other_AD_CNN\application\Dataset' 
# MEL_FEATURE_DIR = r'C:\Users\pepij\OneDrive - Delft University of Technology\SoundSCaper\Feature_log_mel\Dataset_mel'
# MEL_FEATURE_DIR = r'C:\Users\pepij\OneDrive - Delft University of Technology\hop\Dataset_mel_hop_320'


LOUDNESS_FEATURE_DIR = r'C:\Users\pepij\OneDrive - Delft University of Technology\SoundSCaper\Feature_loudness_ISO532_1\Dataset_wav_loudness'

splits = ['test']   # training, validation, test ADDED BY ME
for split in splits:
    split_dir = os.path.join(DATASET_DIR)
    audio_id_path = os.path.join(split_dir, f"{split}_set_audio_file_ids.txt")

    # mel_out = os.path.join(split_dir, f"{split}_log_mel.pickle")
    # mel_out = os.path.join(split_dir, f"{split}_log_mel_hop_320.pickle")
    # loud_out = os.path.join(split_dir, f"{split}_loudness_hop_320.pickle")
    loud_out = os.path.join(split_dir, f"{split}_loudness.pickle")

    # generate_feature_pickle(audio_id_path, MEL_FEATURE_DIR, mel_out)
    generate_feature_pickle(audio_id_path, LOUDNESS_FEATURE_DIR, loud_out)
