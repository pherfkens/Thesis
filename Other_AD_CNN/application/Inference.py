import sys, os, argparse

gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
# from framework.data_generator import *

from framework.data_generator_ad_cnn_org_case_study import *

from framework.processing import *
from framework.models_pytorch import *
from framework.pytorch_utils import count_parameters


def cal_auc(targets_event, outputs_event):
    aucs = []
    for i in range(targets_event.shape[0]):
        test_y_auc, pred_auc = targets_event[i, :], outputs_event[i, :]
        if np.sum(test_y_auc):
            test_auc = metrics.roc_auc_score(test_y_auc, pred_auc)
            aucs.append(test_auc)
    final_auc_event_branch = sum(aucs) / len(aucs)
    return final_auc_event_branch


def main(argv):
    using_model = AD_CNN

    model = using_model()

    print(model)

    syspath = os.path.join(os.getcwd(), 'system', 'model')
    file = 'final_model.pth'

    event_model_path = os.path.join(syspath, file)

    model_event = torch.load(event_model_path, map_location='cpu')

    if 'state_dict' in model_event.keys():
        model.load_state_dict(model_event['state_dict'])
    else:
        model.load_state_dict(model_event)

    if config.cuda:
        model.cuda()

    Dataset_path = os.path.join(os.getcwd(), 'Dataset')
    generator = DataGenerator_Mel_loudness_no_graph(Dataset_path)

    # Generate function
    generate_func = generator.generate_testing(data_type='testing')
    dict = forward(model=model, generate_func=generate_func, cuda=config.cuda)

    # AUC
    # event_auc = cal_auc(dict['label_event'], dict['output_event'])

    # softmax classification acc
    # scene_acc = cal_softmax_classification_accuracy(dict['label_scene'], dict['output_scene'], average='macro')

    pleasant_mse = metrics.mean_squared_error(dict['label_pleasant'], dict['output_pleasant'])
    eventful_mse = metrics.mean_squared_error(dict['label_eventful'], dict['output_eventful'])
    chaotic_mse = metrics.mean_squared_error(dict['label_chaotic'], dict['output_chaotic'])
    vibrant_mse = metrics.mean_squared_error(dict['label_vibrant'], dict['output_vibrant'])
    uneventful_mse = metrics.mean_squared_error(dict['label_uneventful'], dict['output_uneventful'])
    calm_mse = metrics.mean_squared_error(dict['label_calm'], dict['output_calm'])
    annoying_mse = metrics.mean_squared_error(dict['label_annoying'], dict['output_annoying'])
    monotonous_mse = metrics.mean_squared_error(dict['label_monotonous'], dict['output_monotonous'])

    PAQ8 = [pleasant_mse, eventful_mse, chaotic_mse, vibrant_mse, uneventful_mse, calm_mse, annoying_mse, monotonous_mse]

    PAQ8_mean = np.mean(PAQ8)

    # print('ASC\tAcc: ', "%.2f"%(scene_acc*100), '%')
    # print(f'AEC\tAUC: {"%.2f"%(event_auc)}')
    # print(f'PAQ_8D_AQ\tMEAN MSE: {"%.3f"%(PAQ8_mean)}')

    params_num = count_parameters(model)
    print('Parameters num: {} M'.format(params_num / 1000 ** 2))

    # print('ASC\tAcc: ', "%.2f" % (scene_acc * 100), '%')
    # print('AEC\tAUC: ', "%.2f" % (event_auc))
    print(f'PAQ_8D_AQ\tMSE MEAN: {"%.3f" % (PAQ8_mean)}')

    print('pleasant_mse: %.3f' % float(pleasant_mse), 'eventful_mse: %.3f' % float(eventful_mse),
          'chaotic_mse: %.3f' % float(chaotic_mse), 'vibrant_mse: %.3f' % float(vibrant_mse))
    print('uneventful_mse: %.3f' % float(uneventful_mse), 'calm_mse: %.3f' % float(calm_mse),
          'annoying_mse: %.3f' % float(annoying_mse), 'monotonous_mse: %.3f' % float(monotonous_mse))

    from ad_cnn_timing import measure_single_sample_inference_ad_cnn, print_timing_stats, warm_up_model_ad_cnn

    # # After loading your AD_CNN model
    warm_up_model_ad_cnn(model, generator, cuda=config.cuda)
    timing_stats = measure_single_sample_inference_ad_cnn(model, generator, cuda=config.cuda, num_samples=100)
    print_timing_stats(timing_stats)

#  Parameters num: 0.521472 M
# ASC	Acc:  89.30 %
# AEC	AUC:  0.84
# PAQ_8D_AQ	MSE MEAN: 1.137
# pleasant_mse: 0.995 eventful_mse: 1.174 chaotic_mse: 1.155 vibrant_mse: 1.135
# uneventful_mse: 1.184 calm_mse: 1.048 annoying_mse: 1.200 monotonous_mse: 1.205


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)














