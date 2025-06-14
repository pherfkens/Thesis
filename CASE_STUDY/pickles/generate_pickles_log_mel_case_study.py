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

            if isinstance(data, np.ndarray) and data.dtype == object:
                try:
                    if data.ndim == 2 and data.shape[1] == 1:
                        float_data = [[convert_to_float_safe(x[0])] for x in data]
                        data = np.array(float_data, dtype=float)
                    elif data.ndim == 1:
                        float_data = [[convert_to_float_safe(x)] for x in data]
                        data = np.array(float_data, dtype=float)
                except Exception as e:
                    print(f"Warning: Failed to convert data for {audio_id}: {e}")
                    continue

            feature_dict[audio_id] = data
        else:
            print(f"Warning: Missing feature for {audio_id} in {feature_dir}")
    return feature_dict

import numpy as np

def convert_to_float_safe(val):
    """
    Converts a value to float:
    - Handles strings with comma
    - Handles Python int/float
    - Handles NumPy numeric types
    """
    if isinstance(val, str):
        return float(val.replace(',', '.'))
    elif isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    else:
        raise ValueError(f"Unsupported data type: {type(val)}")

def generate_feature_pickle(audio_id_path, npy_dir, out_pickle_path):
    audio_ids = load_lines(audio_id_path)
    feature_dict = load_feature_npy(npy_dir, audio_ids)
    with open(out_pickle_path, 'wb') as f:
        pickle.dump(feature_dict, f)
    print(f"Saved: {out_pickle_path}")

# ==== MAIN EXECUTION ====
# Paths for each dataset split
DATASET_DIR = r"C:\Users\pepij\OneDrive - Delft University of Technology\SoundSCaper\ISD_SOUNDSCAPY"
# MEL_FEATURE_DIR = 'Feature_log_mel/Dataset_hop_320_case_study_npy'
# MEL_FEATURE_DIR = 'Feature_log_mel/Dataset_hop_160_case_study_npy'

# LOUDNESS_FEATURE_DIR = r'C:\Users\pepij\OneDrive - Delft University of Technology\SoundSCaper\Feature_loudness_ISO532_1\Dataset_wav_loudness'
LOUDNESS_FEATURE_DIR = r'C:\Users\pepij\OneDrive - Delft University of Technology\SoundSCaper\Feature_loudness_ISO532_1\Dataset_wav_loudness_case_study'

splits = ['']   # training, validation, test ADDED BY ME
for split in splits:
    split_dir = os.path.join(DATASET_DIR)
    audio_id_path = os.path.join(split_dir, f"{split}groupids_case_study.txt")

    # mel_out = os.path.join(split_dir, f"{split}_hop_160_npy.pickle")
    # mel_out = os.path.join(split_dir, f"{split}_hop_320_npy.pickle")
    # loud_out = os.path.join(split_dir, f"{split}_loudness_hop_320.pickle")
    # loud_out = os.path.join(split_dir, f"{split}_loudness.pickle")
    loud_out = os.path.join(split_dir, f"{split}_loudness_case_study.pickle")

    # generate_feature_pickle(audio_id_path, MEL_FEATURE_DIR, mel_out)
    generate_feature_pickle(audio_id_path, LOUDNESS_FEATURE_DIR, loud_out)
