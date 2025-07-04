import numpy as np
import os, pickle
import time
from framework.utilities import calculate_scalar, scale, create_folder
import framework.config as config
# from functools import lru_cache


class DataGenerator_Mel_loudness_no_graph(object):

    # _pickle_cache = {}

    def __init__(self, Dataset_path, batch_size=config.batch_size, seed=42, normalization=True, overwrite=False,
                 using_mel=True, using_loudness=False):
        self.Dataset_path = Dataset_path
        self.batch_size = batch_size
        self.overwrite = overwrite
        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)
        self.test_random_state = np.random.RandomState(0)

        self.using_mel = using_mel
        self.using_loudness = using_loudness

        # Load data
        load_time = time.time()
        if config.training:
            file_path = os.path.join(Dataset_path, 'Training_set', 'training_scene_event_PAQs.pickle')
            all_data = self.load_pickle(file_path)

            self.train_features, self.train_scene_labels, self.train_sound_maskers, self.train_ISOPls, self.train_ISOEvs, \
            self.train_pleasant, self.train_eventful, self.train_chaotic, self.train_vibrant, \
            self.train_uneventful, self.train_calm, self.train_annoying,  self.train_monotonous, _ = self.get_input_output(all_data)

            lengths = list(map(len, [
                self.train_features,
                self.train_scene_labels,
                self.train_sound_maskers,
                self.train_ISOPls,
                self.train_ISOEvs,
                self.train_pleasant,
                self.train_eventful,
                self.train_chaotic,
                self.train_vibrant,
                self.train_uneventful,
                self.train_calm,
                self.train_annoying,
                self.train_monotonous,
            ]))
            print("Lengths:", lengths)

        
            if using_mel:
                if not config.hop:
                    all_feature_file_path = os.path.join(Dataset_path, 'Training_set', 'training_log_mel.pickle')
                else:
                    all_feature_file_path = os.path.join(Dataset_path, 'Training_set', 'training_log_mel_hop_320.pickle') # ON IF HOP adjusted spectrograms
                print('using: ', all_feature_file_path)
                
                self.train_all_feature_data = self.load_pickle(all_feature_file_path)
                # print(all_feature_data.keys())
                self.train_x = np.array([self.train_all_feature_data[name.replace(".npy", "")] for name in self.train_features])
                print('self.train_x: ', self.train_x.shape)
                # self.train_x:  (19152, 3001, 64)

                self.val_all_feature_data = 0
                self.val_x = 0

            if using_loudness:
                all_feature_file_path = os.path.join(Dataset_path, 'Training_set', 'training_loudness.pickle')
                self.train_all_feature_data_loudness = self.load_pickle(all_feature_file_path)
                # print(all_feature_data.keys())
                self.train_x_loudness = np.array([self.train_all_feature_data_loudness[name.replace(".npy", "")] for name in self.train_features])
                print('self.train_x_loudness: ', self.train_x_loudness.shape)
                # self.train_x_loudness:  (19152, 15000, 1)

                self.val_all_feature_data_loudness = 0
                self.val_x_loudness = 0

        self.normal = normalization
        if not config.hop:
            output_dir = os.path.join(Dataset_path, '0_normalization_files')
        else:
            output_dir = os.path.join(Dataset_path, 'hop_normalization_files')  # ADDED BY ME
        # print('output_dir', output_dir)
        create_folder(output_dir)
        normalization_log_mel_file = os.path.join(output_dir, 'norm_log_mel.pickle')
        normalization_loudness_file = os.path.join(output_dir, 'norm_loudness.pickle')

        if self.normal and not os.path.exists(normalization_log_mel_file) or overwrite:
            norm_pickle = {}
            if using_mel:
                (self.mean_log_mel, self.std_log_mel) = calculate_scalar(np.concatenate(self.train_x))
                norm_pickle['mean'] = self.mean_log_mel
                norm_pickle['std'] = self.std_log_mel
                self.save_pickle(norm_pickle, normalization_log_mel_file)

            if using_loudness:
                norm_pickle = {}
                (self.mean_loudness, self.std_loudness) = calculate_scalar(np.concatenate(self.train_x_loudness))
                norm_pickle['mean'] = self.mean_loudness
                norm_pickle['std'] = self.std_loudness
                self.save_pickle(norm_pickle, normalization_loudness_file)
        else:
            if using_mel:
                print('using: ', normalization_log_mel_file)
                norm_pickle = self.load_pickle(normalization_log_mel_file)
                self.mean_log_mel = norm_pickle['mean']
                self.std_log_mel = norm_pickle['std']
                print('Log Mel Mean: ', self.mean_log_mel)
                print('Log Mel STD: ', self.std_log_mel)

            if using_loudness:
                print('using: ', normalization_loudness_file)
                norm_pickle = self.load_pickle(normalization_loudness_file)
                self.mean_loudness = norm_pickle['mean']
                self.std_loudness = norm_pickle['std']
                print(self.mean_loudness)
                print(self.std_loudness)
        if using_mel:
            print("norm: ", self.mean_log_mel.shape, self.std_log_mel.shape)
        if using_loudness:
            print("norm: ", self.mean_loudness.shape, self.std_loudness.shape)

        print('Loading data time: {:.3f} s'.format(time.time() - load_time))


    def get_input_output(self, all_data):
        # ISOPls, ISOEvs = self.get_ISOPl_ISOEv(all_data)
        # scene_labels = self.load_scene_labels(all_data)
        # ['park' 'park' 'street_traffic' ... 'public_square' 'park' 'public_square']
        # (20160,)
        # print(set(scene_labels))  # {0, 1, 2}

        # attributes = ['pleasant', 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying',  'monotonous']
        audio_names, features, sound_maskers, \
        pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying,  monotonous = all_data['soundscape'], all_data['feature_names'], all_data['masker'], \
                                                             all_data['pleasant'], all_data['eventful'], all_data['chaotic'], \
                                                             all_data['vibrant'], all_data['uneventful'], all_data['calm'], \
                                                             all_data['annoying'], all_data['monotonous']
        
    
        # print('audio_names: ', audio_names, '---', 'features: ', features, '---')
        audio_names = all_data['feature_names']

        assert all_data['all_events'] == config.event_labels
        sound_maskers_labels = all_data['event_labels']

        event_labels = np.zeros((len(sound_maskers_labels), len(config.event_labels)))
        # for i, each in enumerate(sound_maskers_labels):
        #     for sub_each in each:
        #         event_labels[i, config.event_labels.index(sub_each)] = 1

        pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous = np.array(pleasant)[:, None], \
                                                                                       np.array(eventful)[:, None], \
                                                                                       np.array(chaotic)[:, None], \
                                                                                       np.array(vibrant)[:, None], \
                                                                                       np.array(uneventful)[:, None], \
                                                                                       np.array(calm)[:, None], \
                                                                                       np.array(annoying)[:, None], \
                                                                                       np.array(monotonous)[:, None]
        print(pleasant.shape)

        return features, None, event_labels, None, None, \
               pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying,  monotonous, np.array(audio_names)


    def get_ISOPl_ISOEv(self, all_data):
        attributes = ['pleasant', 'eventful', 'chaotic', 'vibrant', 'uneventful', 'calm', 'annoying', 'monotonous']  # Define attributes to extract from dataframes
        ISOPl_weights = [1, 0, -np.sqrt(2) / 2, np.sqrt(2) / 2, 0, np.sqrt(2) / 2, -1, -np.sqrt(2) / 2]  # Define weights for each attribute in attributes in computation of ISO Pleasantness
        ISOEv_weights = [0, 1, np.sqrt(2) / 2, np.sqrt(2) / 2, -1, -np.sqrt(2) / 2, 0, -np.sqrt(2) / 2]

        emotion_values = [all_data[each] for each in attributes]
        # for each in emotion_values:
        #     print(len(each), each)
        # #     20160 [4 4 1 ... 2 2 2]
        # # 20160 [2 2 4 ... 4 1 4]
        # # 20160 [1 1 2 ... 4 1 4]
        # # 20160 [2 2 4 ... 4 1 4] .....

        emotion_values = np.array(emotion_values).transpose((1, 0))
        # print('emotion_values: ', emotion_values.shape)
        # emotion_values:  (19152, 8)

        ISOPls = ((emotion_values * ISOPl_weights).sum(axis=1) / (4 + np.sqrt(32)))

        ISOEvs = ((emotion_values * ISOEv_weights).sum(axis=1) / (4 + np.sqrt(32)))
        # print(ISOPls.shape, ISOEvs.shape)
        # # (20160,) (20160,)
        ISOPls, ISOEvs = ISOPls[:, None], ISOEvs[:, None]
        # print(ISOPls.shape, ISOEvs.shape)
        # # (20160, 1) (20160, 1)
        return ISOPls, ISOEvs

    def load_scene_labels(self, all_data):
        USotW_acoustic_scene_laebls = all_data['USotW_acoustic_scene_labels']
        clips = all_data['soundscape']

        scenes = [USotW_acoustic_scene_laebls[each.split('_44100')[0]] for each in clips]
        correct_scene = []
        for each in scenes:
            if each == 'park ':
                correct_scene.append('park')
            else:
                correct_scene.append(each)

        scene_labels = np.array([config.scene_labels.index(each) for each in correct_scene])

        return scene_labels

    def load_pickle(self, file):
        # if (not self.overwrite) and (file in self._pickle_cache):
        #     return self._pickle_cache[file]
        

        with open(file, 'rb') as f:
            data = pickle.load(f)
        # self._pickle_cache[file] = data
        return data

    def save_pickle(self, data, file):
        with open(file, 'wb') as f:
            pickle.dump(data, f)

    def generate_train(self):

        audios_num = len(self.train_scene_labels)

        audio_indexes = [i for i in range(audios_num)]

        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        while True:
            if pointer >= audios_num:
                pointer = 0
                self.random_state.shuffle(audio_indexes)

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1

            if self.using_mel:
                batch_x = self.train_x[batch_audio_indexes]
            if self.using_loudness:
                batch_x_loudness = self.train_x_loudness[batch_audio_indexes]
            if self.normal:
                if self.using_mel:
                    batch_x = self.transform(batch_x, self.mean_log_mel, self.std_log_mel)
                if self.using_loudness:
                    batch_x_loudness = self.transform(batch_x_loudness, self.mean_loudness, self.std_loudness)

            # ----------------------- emotions ------------------------------------------------------------------------
            batch_scene = self.train_scene_labels[batch_audio_indexes]
            batch_event = self.train_sound_maskers[batch_audio_indexes]

            batch_ISOPls = self.train_ISOPls[batch_audio_indexes]
            batch_ISOEvs = self.train_ISOEvs[batch_audio_indexes]

            batch_pleasant = self.train_pleasant[batch_audio_indexes]
            batch_eventful = self.train_eventful[batch_audio_indexes]
            batch_chaotic = self.train_chaotic[batch_audio_indexes]
            batch_vibrant = self.train_vibrant[batch_audio_indexes]
            batch_uneventful = self.train_uneventful[batch_audio_indexes]
            batch_calm = self.train_calm[batch_audio_indexes]
            batch_annoying = self.train_annoying[batch_audio_indexes]
            batch_monotonous = self.train_monotonous[batch_audio_indexes]

            # if self.using_mel:
            #     yield batch_x, batch_event, \
            #           batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant, \
            #           batch_uneventful, batch_calm, batch_annoying, batch_monotonous

            # if self.using_loudness:
            #     yield batch_x_loudness, batch_event, \
            #           batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant, \
            #           batch_uneventful, batch_calm, batch_annoying, batch_monotonous

            # if self.using_loudness and self.using_mel:
            #     yield batch_x, batch_x_loudness, batch_event, \
            #           batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant, \
            #           batch_uneventful, batch_calm, batch_annoying, batch_monotonous
                

            if self.using_loudness and self.using_mel:
                yield batch_x, batch_x_loudness, batch_event, \
                    batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant, \
                    batch_uneventful, batch_calm, batch_annoying, batch_monotonous
            elif self.using_mel:
                yield batch_x, batch_event, \
                    batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant, \
                    batch_uneventful, batch_calm, batch_annoying, batch_monotonous
            elif self.using_loudness:
                yield batch_x_loudness, batch_event, \
                    batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant, \
                    batch_uneventful, batch_calm, batch_annoying, batch_monotonous



    def generate_validate(self, data_type, max_iteration=None):
        # load
        # ------------------ validation --------------------------------------------------------------------------------
        file_path = os.path.join(self.Dataset_path, 'Validation_set', 'validation_scene_event_PAQs.pickle')
        all_data = self.load_pickle(file_path)

        self.val_features, self.val_scene_labels, self.val_sound_maskers, self.val_ISOPls, self.val_ISOEvs, \
        self.val_pleasant, self.val_eventful, self.val_chaotic, self.val_vibrant, \
        self.val_uneventful, self.val_calm, self.val_annoying, self.val_monotonous, _ = self.get_input_output(
            all_data)

        if self.using_mel:
            if not config.hop:
                all_feature_file_path = os.path.join(self.Dataset_path, 'Validation_set', 'validation_log_mel.pickle')
            else:
                all_feature_file_path = os.path.join(self.Dataset_path, 'Validation_set', 'validation_log_mel_hop_320.pickle')
            print('using: ', all_feature_file_path)
            
            self.val_all_feature_data = self.load_pickle(all_feature_file_path)
            self.val_x = np.array([self.val_all_feature_data[name.replace(".npy", "")] for name in self.val_features])
            print('self.val_x: ', self.val_x.shape)
            # self.val_x:  (2520, 3001, 64)

        if self.using_loudness:
            all_feature_file_path = os.path.join(self.Dataset_path, 'Validation_set', 'validation_loudness.pickle')
            self.val_all_feature_data_loudness = self.load_pickle(all_feature_file_path)
            self.val_x_loudness = np.array([self.val_all_feature_data_loudness[name.replace(".npy", "")] for name in self.val_features])
            print('self.val_x_loudness: ', self.val_x_loudness.shape)
            # self.val_x_loudness:  (2520, 15000, 1)

        audios_num = len(self.val_scene_labels)

        audio_indexes = [i for i in range(audios_num)]

        self.validate_random_state.shuffle(audio_indexes)

        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1

            if self.using_mel:
                batch_x = self.val_x[batch_audio_indexes]
            if self.using_loudness:
                batch_x_loudness = self.val_x_loudness[batch_audio_indexes]
            if self.normal:
                if self.using_mel:
                    batch_x = self.transform(batch_x, self.mean_log_mel, self.std_log_mel)
                if self.using_loudness:
                    batch_x_loudness = self.transform(batch_x_loudness, self.mean_loudness, self.std_loudness)

            # ----------------------- emotions ------------------------------------------------------------------------
            batch_scene = self.val_scene_labels[batch_audio_indexes]
            batch_event = self.val_sound_maskers[batch_audio_indexes]

            batch_ISOPls = self.val_ISOPls[batch_audio_indexes]
            batch_ISOEvs = self.val_ISOEvs[batch_audio_indexes]

            batch_pleasant = self.val_pleasant[batch_audio_indexes]
            batch_eventful = self.val_eventful[batch_audio_indexes]
            batch_chaotic = self.val_chaotic[batch_audio_indexes]
            batch_vibrant = self.val_vibrant[batch_audio_indexes]
            batch_uneventful = self.val_uneventful[batch_audio_indexes]
            batch_calm = self.val_calm[batch_audio_indexes]
            batch_annoying = self.val_annoying[batch_audio_indexes]
            batch_monotonous = self.val_monotonous[batch_audio_indexes]


            if self.using_mel:
                yield batch_x, batch_event, \
                      batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant, \
                      batch_uneventful, batch_calm, batch_annoying, batch_monotonous

            if self.using_loudness:
                yield batch_x_loudness, batch_event, \
                      batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant, \
                      batch_uneventful, batch_calm, batch_annoying, batch_monotonous

            if self.using_loudness and self.using_mel:
                yield batch_x, batch_x_loudness, batch_event, \
                      batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant, \
                      batch_uneventful, batch_calm, batch_annoying, batch_monotonous


    def generate_testing(self, data_type, max_iteration=None):

        # load
        # file_path = os.path.join(self.Dataset_path, 'Testing_set', 'testing_scene_event_PAQs.pickle')
        file_path = os.path.join(self.Dataset_path, 'CASE_STUDY', '_audio_id_PAQs.pickle')
        all_data = self.load_pickle(file_path)

        self.test_features, self.test_scene_labels, self.test_sound_maskers, self.test_ISOPls, self.test_ISOEvs, \
        self.test_pleasant, self.test_eventful, self.test_chaotic, self.test_vibrant, \
        self.test_uneventful, self.test_calm, self.test_annoying, self.test_monotonous, self.test_audio_nams = self.get_input_output(
            all_data)

        if self.using_mel:
            if not config.hop:
                all_feature_file_path = os.path.join(self.Dataset_path, 'Testing_set', 'testing_log_mel.pickle')
            else:
                all_feature_file_path = os.path.join(self.Dataset_path, 'CASE_STUDY', '_hop_320_npy.pickle')
            print('using: ', all_feature_file_path)
            self.test_all_feature_data = self.load_pickle(all_feature_file_path)
            self.test_x = np.array([self.test_all_feature_data[name.replace(".npy", "")] for name in self.test_features])
            print('self.test_x: ', self.test_x.shape)
            # # self.test_x:  (5040, 3001, 64)

        if self.using_loudness:
            all_feature_file_path = os.path.join(self.Dataset_path, 'Testing_set', 'testing_loudness.pickle')
            self.test_all_feature_data_loudness = self.load_pickle(all_feature_file_path)
            self.test_x_loudness = np.array([self.test_all_feature_data_loudness[name.replace(".npy", "")] for name in self.test_features])
            print('self.test_x_loudness: ', self.test_x_loudness.shape)
            # # self.test_x:  (5040, 3001, 64)

        audios_num = len(self.test_audio_nams)

        audio_indexes = [i for i in range(audios_num)]

        self.test_random_state.shuffle(audio_indexes)

        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1

            if self.using_mel:
                batch_x = self.test_x[batch_audio_indexes]
            if self.using_loudness:
                batch_x_loudness = self.test_x_loudness[batch_audio_indexes]
            if self.normal:
                if self.using_mel:
                    batch_x = self.transform(batch_x, self.mean_log_mel, self.std_log_mel)
                if self.using_loudness:
                    batch_x_loudness = self.transform(batch_x_loudness, self.mean_loudness, self.std_loudness)

            # ----------------------- emotions ------------------------------------------------------------------------
            # batch_scene = self.test_scene_labels[batch_audio_indexes]
            batch_event = self.test_sound_maskers[batch_audio_indexes]

            # batch_ISOPls = self.test_ISOPls[batch_audio_indexes]
            # batch_ISOEvs = self.test_ISOEvs[batch_audio_indexes]

            batch_pleasant = self.test_pleasant[batch_audio_indexes]
            print('batch_pleasant: ', batch_pleasant.shape)
            batch_eventful = self.test_eventful[batch_audio_indexes]
            batch_chaotic = self.test_chaotic[batch_audio_indexes]
            batch_vibrant = self.test_vibrant[batch_audio_indexes]
            batch_uneventful = self.test_uneventful[batch_audio_indexes]
            batch_calm = self.test_calm[batch_audio_indexes]
            batch_annoying = self.test_annoying[batch_audio_indexes]
            batch_monotonous = self.test_monotonous[batch_audio_indexes]

            if config.return_names:
                names = self.test_audio_nams[batch_audio_indexes]

                if self.using_mel:
                    yield batch_x, batch_event, \
                        batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant, \
                        batch_uneventful, batch_calm, batch_annoying, batch_monotonous, names
            else:
                if self.using_mel:
                    yield batch_x, batch_event, \
                          batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant, \
                          batch_uneventful, batch_calm, batch_annoying, batch_monotonous

            if self.using_loudness:
                yield batch_x_loudness, batch_event, \
                      batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant, \
                      batch_uneventful, batch_calm, batch_annoying, batch_monotonous

            if self.using_loudness and self.using_mel:
                yield batch_x, batch_x_loudness, batch_event, \
                      batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant, \
                      batch_uneventful, batch_calm, batch_annoying, batch_monotonous


    def transform(self, x, mean, std):
        """Transform data.

        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)

        Returns:
          Transformed data.
        """

        return scale(x, mean, std)



