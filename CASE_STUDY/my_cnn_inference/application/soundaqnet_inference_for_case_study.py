import sys, os, argparse

gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

# from framework.data_generator import *
from framework.data_generator_my_cnn_case_study import *   # AANGEPAST | Terug veranderen bij gebruik van normale dataset

from framework.processing import *
from framework.models_pytorch import *
from framework.pytorch_utils import count_parameters
from inference_timing import *

def cal_auc(targets_event, outputs_event):
    aucs = []
    for i in range(targets_event.shape[0]):
        test_y_auc, pred_auc = targets_event[i, :], outputs_event[i, :]
        if np.sum(test_y_auc):
            test_auc = metrics.roc_auc_score(test_y_auc, pred_auc)
            aucs.append(test_auc)
    if len(aucs) == 0:
        print("No valid AUCs found, returning 0.0")
        return 0.0
    final_auc_event_branch = sum(aucs) / len(aucs)
    return final_auc_event_branch


# def main(argv):
#     using_model = AD_CNN_dense_layer_hop_combined                                          ##   [AD_CNN_decreased_conv_layers, AD_CNN_linear_layer, AD_CNN_hop_length, AD_CNN_harder_max_pooling, AD_CNN_dense_layer_hop_combined] 

#     model = using_model()

#     print(model)

#     syspath = os.path.join(os.getcwd(), 'system', 'model')
#     file = 'final_model_' + using_model.__name__ + '_monitor_pleasant.pth'   # f'final_model_AD_CNN_decreased_conv_layersmonitor_pleasant.pth' 

#     event_model_path = os.path.join(syspath, file)

#     model_event = torch.load(event_model_path, map_location='cpu')

#     if 'state_dict' in model_event.keys():
#         model.load_state_dict(model_event['state_dict'], strict=False)
#     else:
#         model.load_state_dict(model_event)

#     # Example modification of your config
#     config.cuda = torch.cuda.is_available()

#     if config.cuda:
#         model = model.cuda()
#     else:
#         print("Running in CPU mode since CUDA is not available.")

#     Dataset_path = os.path.join(os.getcwd(), 'Dataset')
#     print(f"Dataset path: {Dataset_path}")
#     generator = DataGenerator_Mel_loudness_no_graph(Dataset_path)

#     # # Generate function
#     # generate_func = generator.generate_testing(data_type='testing')
#     # dict = forward(model=model, generate_func=generate_func, cuda=config.cuda)

#     # # AUC
#     # event_auc = cal_auc(dict['label_event'], dict['output_event'])

#     # # softmax classification acc
#     # scene_acc = cal_softmax_classification_accuracy(dict['label_scene'], dict['output_scene'], average='macro')

#     # pleasant_mse = metrics.mean_squared_error(dict['label_pleasant'], dict['output_pleasant'])
#     # eventful_mse = metrics.mean_squared_error(dict['label_eventful'], dict['output_eventful'])
#     # chaotic_mse = metrics.mean_squared_error(dict['label_chaotic'], dict['output_chaotic'])
#     # vibrant_mse = metrics.mean_squared_error(dict['label_vibrant'], dict['output_vibrant'])
#     # uneventful_mse = metrics.mean_squared_error(dict['label_uneventful'], dict['output_uneventful'])
#     # calm_mse = metrics.mean_squared_error(dict['label_calm'], dict['output_calm'])
#     # annoying_mse = metrics.mean_squared_error(dict['label_annoying'], dict['output_annoying'])
#     # monotonous_mse = metrics.mean_squared_error(dict['label_monotonous'], dict['output_monotonous'])

#     # PAQ8 = [pleasant_mse, eventful_mse, chaotic_mse, vibrant_mse, uneventful_mse, calm_mse, annoying_mse, monotonous_mse]

#     # PAQ8_mean = np.mean(PAQ8)

#     # # print('ASC\tAcc: ', "%.2f"%(scene_acc*100), '%')
#     # print(f'AEC\tAUC: {"%.2f"%(event_auc)}')
#     # # print(f'PAQ_8D_AQ\tMEAN MSE: {"%.3f"%(PAQ8_mean)}')

#     params_num = count_parameters(model)
#     print('Parameters num: {} M'.format(params_num / 1000 ** 2))

#     # print('ASC\tAcc: ', "%.2f" % (scene_acc * 100), '%')
#     # print('AEC\tAUC: ', "%.2f" % (event_auc))
#     # print(f'PAQ_8D_AQ\tMSE MEAN: {"%.3f" % (PAQ8_mean)}')

#     # print('pleasant_mse: %.3f' % float(pleasant_mse), 'eventful_mse: %.3f' % float(eventful_mse),
#     #       'chaotic_mse: %.3f' % float(chaotic_mse), 'vibrant_mse: %.3f' % float(vibrant_mse))
#     # print('uneventful_mse: %.3f' % float(uneventful_mse), 'calm_mse: %.3f' % float(calm_mse),
#     #       'annoying_mse: %.3f' % float(annoying_mse), 'monotonous_mse: %.3f' % float(monotonous_mse))

#     # Inference.py — inside main()

#     # Forward pass
#     generate_func = generator.generate_testing(data_type='testing')
#     dict = forward(model=model, generate_func=generate_func, cuda=config.cuda)

#     # AUC for event detection
#     event_auc = cal_auc(dict['label_event'], dict['output_event']) 

#     # MSE for PAQ attributes
#     pleasant_mse = metrics.mean_squared_error(dict['label_pleasant'], dict['output_pleasant'])
#     eventful_mse = metrics.mean_squared_error(dict['label_eventful'], dict['output_eventful'])
#     chaotic_mse = metrics.mean_squared_error(dict['label_chaotic'], dict['output_chaotic'])
#     vibrant_mse = metrics.mean_squared_error(dict['label_vibrant'], dict['output_vibrant'])
#     uneventful_mse = metrics.mean_squared_error(dict['label_uneventful'], dict['output_uneventful'])
#     calm_mse = metrics.mean_squared_error(dict['label_calm'], dict['output_calm'])
#     annoying_mse = metrics.mean_squared_error(dict['label_annoying'], dict['output_annoying'])
#     monotonous_mse = metrics.mean_squared_error(dict['label_monotonous'], dict['output_monotonous'])

#     PAQ8 = [pleasant_mse, eventful_mse, chaotic_mse, vibrant_mse,
#             uneventful_mse, calm_mse, annoying_mse, monotonous_mse]

#     PAQ8_mean = np.mean(PAQ8)

#     # Reporting
#     print('AEC\tAUC: ', "%.2f" % (event_auc))
#     print(f'PAQ_8D_AQ\tMSE MEAN: {"%.3f" % (PAQ8_mean)}')

#     print('pleasant_mse: %.3f' % pleasant_mse, 'eventful_mse: %.3f' % eventful_mse,
#         'chaotic_mse: %.3f' % chaotic_mse, 'vibrant_mse: %.3f' % vibrant_mse)
#     print('uneventful_mse: %.3f' % uneventful_mse, 'calm_mse: %.3f' % calm_mse,
#         'annoying_mse: %.3f' % annoying_mse, 'monotonous_mse: %.3f' % monotonous_mse)

#     # from framework.models_pytorch import AD_CNN_decreased_conv_layers
#     # from cnn_timing import test_single_model_timing, warm_up_model_cnn, measure_single_sample_inference_cnn_models, print_timing_stats

#     # Initialize your data generator
#     Dataset_path = os.path.join(os.getcwd(), 'Dataset')
#     # generator = YourDataGenerator(Dataset_path)  # Replace with actual generator

#     # Test one model
#     # model = AD_CNN_hop_length()
#     if config.cuda:
#         model.cuda()

#     # warm_up_model_cnn(model, generator, cuda=config.cuda)
#     # stats = measure_single_sample_inference_cnn_models(
#     #     model, generator, cuda=config.cuda, num_samples=100, 
#     #     model_name="AD_CNN_dense_layer_hop_combined"
#     # )
#     # print_timing_stats(stats)

# #  Parameters num: 0.521472 M
# # ASC	Acc:  89.30 %
# # AEC	AUC:  0.84
# # PAQ_8D_AQ	MSE MEAN: 1.137
# # pleasant_mse: 0.995 eventful_mse: 1.174 chaotic_mse: 1.155 vibrant_mse: 1.135
# # uneventful_mse: 1.184 calm_mse: 1.048 annoying_mse: 1.200 monotonous_mse: 1.205

def main(argv):
    using_model = AD_CNN_hop_length

    model = using_model()
    print(model)

    syspath = os.path.join(os.getcwd(), 'system', 'model')
    file = 'final_model_' + using_model.__name__ + '_monitor_pleasant.pth'
    event_model_path = os.path.join(syspath, file)

    model_event = torch.load(event_model_path, map_location='cpu')
    if 'state_dict' in model_event.keys():
        model.load_state_dict(model_event['state_dict'], strict=False)
    else:
        model.load_state_dict(model_event)

    config.cuda = torch.cuda.is_available()
    if config.cuda:
        model = model.cuda()
    else:
        print("Running in CPU mode since CUDA is not available.")

    Dataset_path = os.path.join(os.getcwd(), 'Dataset')
    generator = DataGenerator_Mel_loudness_no_graph(Dataset_path)
    print(config.return_names)
    generate_func = generator.generate_testing(data_type='testing')
    dict = forward(model=model, return_names=True, generate_func=generate_func, cuda=config.cuda)
    print('xxxx')
    print(config.return_names)
    # ======================== OUTPUT TO FILES ================================
    # result_event_dir = os.path.join(os.getcwd(), 'SoundAQ_event_probability_case_study')
    # result_PAQ_dir = os.path.join(os.getcwd(), 'SoundAQ_PAQ_case_study')

    result_event_dir = os.path.join(os.getcwd(), 'EVENTS_ad_cnn_hop_length_case_study')
    result_PAQ_dir = os.path.join(os.getcwd(), 'PAQ_ad_cnn_hop_length_case_study')

    create_folder(result_event_dir)
    create_folder(result_PAQ_dir)

    for each_index, name in enumerate(dict['audio_names']):
        print(f"Processing {name} at index {each_index}")
        print(each_index, name)

        # Save event probabilities
        event_file = os.path.join(result_event_dir, name.replace('.npy', '_event.txt'))
        np.savetxt(event_file, dict['output_event'][each_index])

        # Save scene and PAQ attributes
        paq_file = os.path.join(result_PAQ_dir, name.replace('.npy', '_scene_PAQ.txt'))
        with open(paq_file, 'w') as f:
            # Scene prediction (optional: handle if not present in model)
            # if 'output_scene' in dict:
            #     max_id = np.argmax(dict['output_scene'][each_index])
            #     f.write(config.scene_labels[max_id] + '\n')
            # else:
            #     f.write("SceneLabelUnavailable\n")

            # Write ISOP/ISOE if available
            # iso_pl = dict.get('output_ISOPls', [None])[each_index]
            # iso_ev = dict.get('output_ISOEvs', [None])[each_index]
            # if iso_pl is not None and iso_ev is not None:
            #     f.write(f"{iso_pl[0]}\t{iso_ev[0]}\n")

            # Write PAQ 8D attributes
            paq_values = [
                dict['output_pleasant'][each_index][0],
                dict['output_eventful'][each_index][0],
                dict['output_chaotic'][each_index][0],
                dict['output_vibrant'][each_index][0],
                dict['output_uneventful'][each_index][0],
                dict['output_calm'][each_index][0],
                dict['output_annoying'][each_index][0],
                dict['output_monotonous'][each_index][0],
            ]
            f.write('\t'.join([str(v) for v in paq_values]) + '\n')
    # ========================================================================

    # Summary print (optional)
    event_auc = cal_auc(dict['label_event'], dict['output_event']) 
    print('AEC\tAUC: ', "%.2f" % event_auc)

    PAQ8 = [
        metrics.mean_squared_error(dict['label_pleasant'], dict['output_pleasant']),
        metrics.mean_squared_error(dict['label_eventful'], dict['output_eventful']),
        metrics.mean_squared_error(dict['label_chaotic'], dict['output_chaotic']),
        metrics.mean_squared_error(dict['label_vibrant'], dict['output_vibrant']),
        metrics.mean_squared_error(dict['label_uneventful'], dict['output_uneventful']),
        metrics.mean_squared_error(dict['label_calm'], dict['output_calm']),
        metrics.mean_squared_error(dict['label_annoying'], dict['output_annoying']),
        metrics.mean_squared_error(dict['label_monotonous'], dict['output_monotonous']),
    ]

    print(f'PAQ_8D_AQ\tMSE MEAN: {np.mean(PAQ8):.3f}')



if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















