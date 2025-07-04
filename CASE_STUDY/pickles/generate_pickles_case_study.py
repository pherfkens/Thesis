# import os
# import pickle
# import numpy as np
# from collections import defaultdict

# # Define your master event label list from config
# ALL_EVENTS = [
#     'Silence', 'Human sounds', 'Wind', 'Water', 'Natural sounds',
#     'Traffic', 'Sounds of things', 'Vehicle', 'Bird',
#     'Outside, rural or natural', 'Environment and background',
#     'Speech', 'Music', 'Noise', 'Animal'
# ]

# def load_lines(file_path):
#     with open(file_path, 'r') as f:
#         return [line.strip() for line in f if line.strip()]

# def load_paq_attributes(paq_path):
#     data = np.loadtxt(paq_path)
#     return [data[:, i].tolist() for i in range(8)]

# def load_isopl_isoe(path):
#     data = np.loadtxt(path)
#     return data[:, 0].tolist(), data[:, 1].tolist()

# def build_event_labels(event_lines, all_events):
#     label_matrix = np.zeros((len(event_lines), len(all_events)))
#     for i, line in enumerate(event_lines):
#         labels = line.split('\t') if '\t' in line else line.split()
#         for label in labels:
#             if label in all_events:
#                 label_matrix[i, all_events.index(label)] = 1
#     return label_matrix.tolist()

# def generate_pickle(split_name, base_path, has_events=True):
#     split_dir = os.path.join(base_path)

#     audio_ids = load_lines(os.path.join(split_dir, f"{split_name}_set_audio_file_ids.txt"))
#     scene_labels = load_lines(os.path.join(split_dir, f"{split_name}_set_acoustic_scene_labels.txt"))
#     isopl, isoev = load_isopl_isoe(os.path.join(split_dir, f"{split_name}_set_ISOP_ISOE.txt"))
#     paq_attrs = load_paq_attributes(os.path.join(split_dir, f"{split_name}_set_PAQ_8D_AQs.txt"))

#     pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous = paq_attrs

#     if has_events:
#         event_lines = load_lines(os.path.join(split_dir, f"{split_name}_set_audio_event_labels.txt"))
#         event_labels = build_event_labels(event_lines, ALL_EVENTS)
#     else:
#         event_labels = [[0]*len(ALL_EVENTS) for _ in audio_ids]

#     scene_label_dict = {audio_id.split('_44100')[0]: label.strip() for audio_id, label in zip(audio_ids, scene_labels)}

#     output_dict = {
#         'soundscape': audio_ids,
#         'feature_names': audio_ids,
#         'USotW_acoustic_scene_labels': scene_label_dict,
#         'event_labels': event_labels,
#         'all_events': ALL_EVENTS,
#         'pleasant': pleasant,
#         'eventful': eventful,
#         'chaotic': chaotic,
#         'vibrant': vibrant,
#         'uneventful': uneventful,
#         'calm': calm,
#         'annoying': annoying,
#         'monotonous': monotonous
#     }

#     out_path = os.path.join(split_dir, f"{split_name}_scene_event_PAQs.pickle")
#     with open(out_path, 'wb') as f:
#         pickle.dump(output_dict, f)
#     print(f"Saved: {out_path}")

# # ==== MAIN EXECUTION ====
# BASE_DATASET_PATH = r'C:\Users\pepij\OneDrive - Delft University of Technology\SoundSCaper\Dataset_training_validation_test'

# # Training, Validation, Testing (all have event labels now)
# generate_pickle('training', BASE_DATASET_PATH, has_events=True)
# generate_pickle('validation', BASE_DATASET_PATH, has_events=True)
# generate_pickle('test', BASE_DATASET_PATH, has_events=True)

##################################################################################################3

import os
import pickle
import numpy as np



MEL_FEATURE_DIR = 'Feature_log_mel/Dataset_mel'
# LOUDNESS_FEATURE_DIR = 'Feature_loudness_ISO532_1/Dataset_wav_loudness'

# Define your master event label list from config
ALL_EVENTS = [
    'Silence', 'Human sounds', 'Wind', 'Water', 'Natural sounds',
    'Traffic', 'Sounds of things', 'Vehicle', 'Bird',
    'Outside, rural or natural', 'Environment and background',
    'Speech', 'Music', 'Noise', 'Animal'
]

def load_lines(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_paq_attributes(paq_path):
    data = np.loadtxt(paq_path)
    return [data[:, i].tolist() for i in range(data.shape[1])]

scene_label_dict = {}
with open (r"C:\Users\pepij\OneDrive - Delft University of Technology\SoundSCaper\Dataset_training_validation_test\dictionary_acoustic_scene_labels.txt", 'r') as f: data = f.read()
scene_label_dict = eval(data)



def generate_paq_pickle(split_name, base_path):
    split_dir = os.path.join(base_path)

    audio_ids = load_lines(os.path.join(split_dir, f"{split_name}groupids_case_study.txt"))
    # scene_labels = load_lines(os.path.join(split_dir, f"{split_name}_set_acoustic_scene_labels.txt"))
    paq_attrs = load_paq_attributes(os.path.join(split_dir, f"{split_name}paqs_case_study.txt"))
    # isopl, isoev = load_isopl_isoe(os.path.join(split_dir, f"{split_name}_set_ISOP_ISOE.txt"))
    # masker_lines = load_lines(os.path.join(split_dir, f"{split_name}_set_masker.txt"))
    # soundscapes_lines = load_lines(os.path.join(split_dir, f"{split_name}_set_soundscapes.txt"))
    # app_leql_leqr = load_appropriate_leql_leqr(os.path.join(split_dir, f"{split_name}_set_appropriate_leql_leqr.txt"))
    # print(app_leql_leqr)
    pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous = paq_attrs
    # appropriate, leq_l_r, leq_r_r = app_leql_leqr
    # masker  = [masker.strip() for masker in masker_lines]
    # soundscape_line = [soundscape.strip() for soundscape in soundscapes_lines]



    output_dict = {
        'USotW_acoustic_scene_labels': scene_label_dict,
        'all_events': ALL_EVENTS,
        'event_labels': [None] * len(audio_ids),  # Placeholder for event labels
        'feature_names': [f"{aid}.npy" for aid in audio_ids],
        'soundscape': [None] * len(audio_ids),  # Placeholder for soundscapes
        'masker': [None] * len(audio_ids),  # Placeholder for maskers
        'pleasant': pleasant,
        'eventful': eventful,
        'chaotic': chaotic,
        'vibrant': vibrant,
        'uneventful': uneventful,
        'calm': calm,
        'annoying': annoying,
        'monotonous': monotonous,
        'appropriate': [None] * len(audio_ids),
        'Leq_L_r': [None] * len(audio_ids),
        'Leq_R_r': [None] * len(audio_ids)
    }

    out_path = os.path.join(split_dir, f"{split_name}_audio_id_PAQs.pickle")
    with open(out_path, 'wb') as f:
        pickle.dump(output_dict, f)
    print(f"Saved: {out_path}")

# ==== MAIN EXECUTION ====
BASE_DATASET_PATH = r'C:\Users\pepij\OneDrive - Delft University of Technology\SoundSCaper\ISD_SOUNDSCAPY'

# generate_paq_pickle('training', BASE_DATASET_PATH)
# generate_paq_pickle('validation', BASE_DATASET_PATH)
generate_paq_pickle('', BASE_DATASET_PATH)
