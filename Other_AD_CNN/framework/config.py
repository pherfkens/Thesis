import torch, os

####################################################################################################

cuda = 1

training = 0
testing = 1

if cuda:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device('cpu')

mel_bins = 64
batch_size = 64 # ADDED BY ME
lr_init = 1e-4
epochs = 10   # ADDED BY ME

endswith = '.pth'

event_labels = ['Silence', 'Human sounds', 'Wind', 'Water', 'Natural sounds', 'Traffic', 'Sounds of things', 'Vehicle',
                'Bird', 'Outside, rural or natural', 'Environment and background', 'Speech', 'Music', 'Noise', 'Animal']
scene_labels = ['public_square', 'park', 'street_traffic',]
each_emotion_class_num = 1


