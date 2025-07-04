import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import framework.config as config

def move_data_to_gpu(x, cuda, using_float=False):
    if using_float:
        x = torch.Tensor(x)
    else:
        if 'float' in str(x.dtype):
            x = torch.Tensor(x)

        elif 'int' in str(x.dtype):
            x = torch.LongTensor(x)

        else:
            raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x

def init_layer(layer):
    if layer.weight.ndimension() == 4:
        n_out, n_in, h, w = layer.weight.size()
        n = n_in * h * w
    elif layer.weight.ndimension() == 2:
        n_out, n = layer.weight.size()
    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)
    if layer.bias is not None:
        layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class AD_CNN_decreased_conv_layers(nn.Module):            # IN REPORT: AD_CNN_decreased_filters
    def __init__(self, event_class=15):

        super(AD_CNN_decreased_conv_layers, self).__init__()

        in_channels = 1
        out_channels = 8
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        in_channels = 8
        out_channels = 8
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(3, 3), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        in_channels = 8
        out_channels = 16
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(3, 3), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.linear_layer = nn.Linear(2400, 100, bias=True)
        # Dense layer #1

        each_emotion_class = 1

        self.fc_final_event = nn.Linear(100, event_class, bias=True)

        self.fc_final_pleasant = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_eventful = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_chaotic = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_vibrant = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_uneventful = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_calm = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_annoying = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_monotonous = nn.Linear(100, each_emotion_class, bias=True)

    def forward(self, input):

        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        # print(x.size())
        # torch.Size([64, 1, 1501, 64])

        # CNN layer #1
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.size())
        # torch.Size([64, 16, 2997, 60])

        # CNN layer #2
        x = F.relu(self.bn2(self.conv2(x)))
        pool_size = (5, 5)
        x = F.max_pool2d(x, kernel_size=pool_size)
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())
        # torch.Size([64, 16, 598, 11])

        # CNN layer #3
        x = F.relu(self.bn3(self.conv3(x)))
        pool_size = (4, 10)
        x = F.max_pool2d(x, kernel_size=pool_size)
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())
        # torch.Size([64, 32, 5, 1])

        x = x.flatten(1)
        # print(x.size())
        # torch.Size([64, 160])

        x = F.relu(self.linear_layer(x))
        x = F.dropout(x, p=0.3, training=self.training)

        event = self.fc_final_event(x)

        pleasant = self.fc_final_pleasant(x)
        eventful = self.fc_final_eventful(x)
        chaotic = self.fc_final_chaotic(x)
        vibrant = self.fc_final_vibrant(x)
        uneventful = self.fc_final_uneventful(x)
        calm = self.fc_final_calm(x)
        annoying = self.fc_final_annoying(x)
        monotonous = self.fc_final_monotonous(x)

        return event, pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous
    

##### Decrease hop_lenth makes seq_len smaller, makes input smaller #####

class AD_CNN_hop_length(nn.Module):
    def __init__(self, event_class=len(config.event_labels)):

        super(AD_CNN_hop_length, self).__init__()

        in_channels = 1
        out_channels = 16
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        in_channels = 16
        out_channels = 16
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(3, 3), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        in_channels = 16
        out_channels = 32
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(3, 3), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.linear_layer = nn.Linear(2400, 100, bias=True)
        # Dense layer #1

        each_emotion_class = 1

        self.fc_final_event = nn.Linear(100, event_class, bias=True)

        self.fc_final_pleasant = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_eventful = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_chaotic = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_vibrant = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_uneventful = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_calm = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_annoying = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_monotonous = nn.Linear(100, each_emotion_class, bias=True)

    def forward(self, input):

        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        # print(x.size())
        # torch.Size([64, 1, 1501, 64])

        # CNN layer #1
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.size())
        # torch.Size([64, 16, 1501, 64])

        # CNN layer #2
        x = F.relu(self.bn2(self.conv2(x)))
        pool_size = (5, 5)
        x = F.max_pool2d(x, kernel_size=pool_size)
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())
        # torch.Size([64, 16, 300, 12])

        # CNN layer #3
        x = F.relu(self.bn3(self.conv3(x)))
        pool_size = (4, 10)
        x = F.max_pool2d(x, kernel_size=pool_size)
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())
        # torch.Size([64, 32, 75, 1])

        x = x.flatten(1)
        # print(x.size())
        # torch.Size([64, 2400])

        x = F.relu(self.linear_layer(x))
        x = F.dropout(x, p=0.3, training=self.training)

        event = self.fc_final_event(x)

        pleasant = self.fc_final_pleasant(x)
        eventful = self.fc_final_eventful(x)
        chaotic = self.fc_final_chaotic(x)
        vibrant = self.fc_final_vibrant(x)
        uneventful = self.fc_final_uneventful(x)
        calm = self.fc_final_calm(x)
        annoying = self.fc_final_annoying(x)
        monotonous = self.fc_final_monotonous(x)

        return event, pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous
    

class AD_CNN_linear_layer(nn.Module):
    def __init__(self, event_class=len(config.event_labels)):

        super(AD_CNN_linear_layer, self).__init__()

        in_channels = 1
        out_channels = 16
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        in_channels = 16
        out_channels = 16
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(3, 3), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        in_channels = 16
        out_channels = 32
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(3, 3), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.linear_layer = nn.Linear(4800, 50, bias=True)
        # Dense layer #1

        each_emotion_class = 1

        self.fc_final_event = nn.Linear(50, event_class, bias=True)

        self.fc_final_pleasant = nn.Linear(50, each_emotion_class, bias=True)
        self.fc_final_eventful = nn.Linear(50, each_emotion_class, bias=True)
        self.fc_final_chaotic = nn.Linear(50, each_emotion_class, bias=True)
        self.fc_final_vibrant = nn.Linear(50, each_emotion_class, bias=True)
        self.fc_final_uneventful = nn.Linear(50, each_emotion_class, bias=True)
        self.fc_final_calm = nn.Linear(50, each_emotion_class, bias=True)
        self.fc_final_annoying = nn.Linear(50, each_emotion_class, bias=True)
        self.fc_final_monotonous = nn.Linear(50, each_emotion_class, bias=True)

    def forward(self, input):

        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        # print(x.size())
        # torch.Size([64, 1, 3001, 64])

        # CNN layer #1
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.size())
        # torch.Size([64, 16, 3001, 64])

        # CNN layer #2
        x = F.relu(self.bn2(self.conv2(x)))
        pool_size = (5, 5)
        x = F.max_pool2d(x, kernel_size=pool_size)
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())
        # torch.Size([64, 16, 600, 12])

        # CNN layer #3
        x = F.relu(self.bn3(self.conv3(x)))
        pool_size = (4, 10)
        x = F.max_pool2d(x, kernel_size=pool_size)
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())
        # torch.Size([64, 32, 150, 1])

        x = x.flatten(1)
        # print(x.size())
        # torch.Size([64, 4800])

        x = F.relu(self.linear_layer(x))
        x = F.dropout(x, p=0.3, training=self.training)

        event = self.fc_final_event(x)

        pleasant = self.fc_final_pleasant(x)
        eventful = self.fc_final_eventful(x)
        chaotic = self.fc_final_chaotic(x)
        vibrant = self.fc_final_vibrant(x)
        uneventful = self.fc_final_uneventful(x)
        calm = self.fc_final_calm(x)
        annoying = self.fc_final_annoying(x)
        monotonous = self.fc_final_monotonous(x)

        return event, pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous

class AD_CNN_harder_max_pooling(nn.Module):
    def __init__(self, event_class=len(config.event_labels)):

        super(AD_CNN_harder_max_pooling, self).__init__()

        in_channels = 1
        out_channels = 16
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        in_channels = 16
        out_channels = 16
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(3, 3), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        in_channels = 16
        out_channels = 32
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(3, 3), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Reduced linear layer: 32 * 18 * 1 = 576 features 
        self.linear_layer = nn.Linear(576, 100, bias=True)
        # Dense layer #1

        each_emotion_class = 1

        self.fc_final_event = nn.Linear(100, event_class, bias=True)

        self.fc_final_pleasant = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_eventful = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_chaotic = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_vibrant = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_uneventful = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_calm = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_annoying = nn.Linear(100, each_emotion_class, bias=True)
        self.fc_final_monotonous = nn.Linear(100, each_emotion_class, bias=True)

    def forward(self, input):

        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        # print(x.size())
        # torch.Size([32, 1, 3001, 64])

        # CNN layer #1
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.size())
        # torch.Size([32, 16, 3001, 64])

        # CNN layer #2
        x = F.relu(self.bn2(self.conv2(x)))
        pool_size = (10, 5)
        x = F.max_pool2d(x, kernel_size=pool_size)
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())
        # torch.Size([32, 16, 300, 12])

        # CNN layer #3
        x = F.relu(self.bn3(self.conv3(x)))
        pool_size = (16, 12)
        x = F.max_pool2d(x, kernel_size=pool_size)
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())
        # torch.Size([32, 32, 18, 1])

        x = x.flatten(1)
        # print(x.size())
        # torch.Size([32, 576])

        x = F.relu(self.linear_layer(x))
        x = F.dropout(x, p=0.3, training=self.training)

        event = self.fc_final_event(x)

        pleasant = self.fc_final_pleasant(x)
        eventful = self.fc_final_eventful(x)
        chaotic = self.fc_final_chaotic(x)
        vibrant = self.fc_final_vibrant(x)
        uneventful = self.fc_final_uneventful(x)
        calm = self.fc_final_calm(x)
        annoying = self.fc_final_annoying(x)
        monotonous = self.fc_final_monotonous(x)

        return event, pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous
    

class AD_CNN_dense_layer_hop_combined(nn.Module):
    def __init__(self, event_class=len(config.event_labels)):

        super(AD_CNN_dense_layer_hop_combined, self).__init__()

        in_channels = 1
        out_channels = 16
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        in_channels = 16
        out_channels = 16
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(3, 3), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        in_channels = 16
        out_channels = 32
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(7, 7), stride=(1, 1),
                               padding=(3, 3), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.linear_layer = nn.Linear(2400, 50, bias=True)
        # Dense layer #1

        each_emotion_class = 1

        self.fc_final_event = nn.Linear(50, event_class, bias=True)

        self.fc_final_pleasant = nn.Linear(50, each_emotion_class, bias=True)
        self.fc_final_eventful = nn.Linear(50, each_emotion_class, bias=True)
        self.fc_final_chaotic = nn.Linear(50, each_emotion_class, bias=True)
        self.fc_final_vibrant = nn.Linear(50, each_emotion_class, bias=True)
        self.fc_final_uneventful = nn.Linear(50, each_emotion_class, bias=True)
        self.fc_final_calm = nn.Linear(50, each_emotion_class, bias=True)
        self.fc_final_annoying = nn.Linear(50, each_emotion_class, bias=True)
        self.fc_final_monotonous = nn.Linear(50, each_emotion_class, bias=True)

    def forward(self, input):

        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        # print(x.size())
        # torch.Size([64, 1, 3001, 64])

        # CNN layer #1
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.size())
        # torch.Size([64, 16, 3001, 64])

        # CNN layer #2
        x = F.relu(self.bn2(self.conv2(x)))
        pool_size = (5, 5)
        x = F.max_pool2d(x, kernel_size=pool_size)
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())
        # torch.Size([64, 16, 600, 12])

        # CNN layer #3
        x = F.relu(self.bn3(self.conv3(x)))
        pool_size = (4, 10)
        x = F.max_pool2d(x, kernel_size=pool_size)
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.size())
        # torch.Size([64, 32, 150, 1])

        x = x.flatten(1)
        # print(x.size())
        # torch.Size([64, 4800])

        x = F.relu(self.linear_layer(x))
        x = F.dropout(x, p=0.3, training=self.training)

        event = self.fc_final_event(x)

        pleasant = self.fc_final_pleasant(x)
        eventful = self.fc_final_eventful(x)
        chaotic = self.fc_final_chaotic(x)
        vibrant = self.fc_final_vibrant(x)
        uneventful = self.fc_final_uneventful(x)
        calm = self.fc_final_calm(x)
        annoying = self.fc_final_annoying(x)
        monotonous = self.fc_final_monotonous(x)

        return event, pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous
