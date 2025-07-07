import time, os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import csv

from framework.utilities import create_folder
from framework.models_pytorch import move_data_to_gpu
import framework.config as config
from sklearn import metrics
from framework.earlystop import EarlyStopping



def forward(model, generate_func, cuda, return_names = False, using_mel=True, using_loudness=False):
    output_event = []
    output_pleasant, output_eventful, output_chaotic, output_vibrant = [], [], [], []
    output_uneventful, output_calm, output_annoying, output_monotonous = [], [], [], []

    label_event = []
    label_pleasant, label_eventful, label_chaotic, label_vibrant = [], [], [], []
    label_uneventful, label_calm, label_annoying, label_monotonous = [], [], [], []

    audio_names = []
    config.return_names = return_names
    # Evaluate on mini-batch
    for num, data in enumerate(generate_func):
        # if return_names:
        #     if using_mel:
        #         (batch_x, batch_event,
        #         batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
        #         batch_uneventful, batch_calm, batch_annoying, batch_monotonous) = data
        #     if using_loudness:
        #         (batch_x_loudness, batch_event,
        #          batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
        #          batch_uneventful, batch_calm, batch_annoying, batch_monotonous, names) = data
        #     if using_mel and using_loudness:
        #         (batch_x, batch_x_loudness, batch_event,
        #          batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
        #          batch_uneventful, batch_calm, batch_annoying, batch_monotonous, names) = data
        # # print(batch_x.shape, batch_scene.shape, batch_event.shape, batch_ISOPls.shape, batch_ISOEvs.shape,
        # #  batch_pleasant.shape, batch_eventful.shape, batch_chaotic.shape, batch_vibrant.shape,
        # #  batch_uneventful.shape, batch_calm.shape, batch_annoying.shape, batch_monotonous.shape)
        #     audio_names.append(names)
        # else:
        #     if using_mel:
        #         (batch_x, batch_event,
        #          batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
        #          batch_uneventful, batch_calm, batch_annoying, batch_monotonous) = data
        #     if using_loudness:
        #         ( batch_x_loudness, batch_event,
        #          batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
        #          batch_uneventful, batch_calm, batch_annoying, batch_monotonous) = data
        #     if using_mel and using_loudness:
        #         (batch_x, batch_x_loudness, batch_event,
        #          batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
        #          batch_uneventful, batch_calm, batch_annoying, batch_monotonous) = data
        if return_names:
            if using_mel and using_loudness:
                (batch_x, batch_x_loudness, batch_event,
                batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
                batch_uneventful, batch_calm, batch_annoying, batch_monotonous, names) = data
            elif using_mel:
                (batch_x, batch_event,
                batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
                batch_uneventful, batch_calm, batch_annoying, batch_monotonous, names) = data
                print('Using mel features only')
                # print(data)
                # names = True  # Or handle differently
            elif using_loudness:
                (batch_x_loudness, batch_event,
                batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
                batch_uneventful, batch_calm, batch_annoying, batch_monotonous, names) = data
            # if names is not None:
            audio_names.append(names)

        else:
            if using_mel and using_loudness:
                (batch_x, batch_x_loudness, batch_event,
                batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
                batch_uneventful, batch_calm, batch_annoying, batch_monotonous) = data
            elif using_mel:
                (batch_x, batch_event,
                batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
                batch_uneventful, batch_calm, batch_annoying, batch_monotonous) = data
            elif using_loudness:
                (batch_x_loudness, batch_event,
                batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
                batch_uneventful, batch_calm, batch_annoying, batch_monotonous) = data

        if using_mel:
            batch_x = move_data_to_gpu(batch_x, cuda)
        if using_loudness:
            batch_x_loudness = move_data_to_gpu(batch_x_loudness, cuda)


        model.eval()
        with torch.no_grad():
            # if using_mel:
            #     event, pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous = model(batch_x)

            # if using_loudness:
            #     event, pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous = model(batch_x_loudness
            #                                                                                          )

            # if using_mel and using_loudness:
            #     event, pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous = model(batch_x,
            #                                                                                          batch_x_loudness
            #                                                                                          )
            if using_mel and using_loudness:
                model_outputs = model(batch_x, batch_x_loudness)
            elif using_mel:
                model_outputs = model(batch_x)
            elif using_loudness:
                model_outputs = model(batch_x_loudness)

            event, pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous = model_outputs

            event = F.sigmoid(event)
            
            output_event.append(event.data.cpu().numpy())

            output_pleasant.append(pleasant.data.cpu().numpy())
            output_eventful.append(eventful.data.cpu().numpy())
            output_chaotic.append(chaotic.data.cpu().numpy())
            output_vibrant.append(vibrant.data.cpu().numpy())
            output_uneventful.append(uneventful.data.cpu().numpy())
            output_calm.append(calm.data.cpu().numpy())
            output_annoying.append(annoying.data.cpu().numpy())
            output_monotonous.append(monotonous.data.cpu().numpy())

            # ------------------------- labels -------------------------------------------------------------------------
            label_event.append(batch_event)

            label_pleasant.append(batch_pleasant)
            label_eventful.append(batch_eventful)
            label_chaotic.append(batch_chaotic)
            label_vibrant.append(batch_vibrant)
            label_uneventful.append(batch_uneventful)
            label_calm.append(batch_calm)
            label_annoying.append(batch_annoying)
            label_monotonous.append(batch_monotonous)

    dict = {}

    if return_names:
        dict['audio_names'] = np.concatenate(audio_names, axis=0)
        print('Audio names:', dict['audio_names'])

    dict['output_event'] = np.concatenate(output_event, axis=0)

    dict['output_pleasant'] = np.concatenate(output_pleasant, axis=0)
    dict['output_eventful'] = np.concatenate(output_eventful, axis=0)
    dict['output_chaotic'] = np.concatenate(output_chaotic, axis=0)
    dict['output_vibrant'] = np.concatenate(output_vibrant, axis=0)
    dict['output_uneventful'] = np.concatenate(output_uneventful, axis=0)
    dict['output_calm'] = np.concatenate(output_calm, axis=0)
    dict['output_annoying'] = np.concatenate(output_annoying, axis=0)
    dict['output_monotonous'] = np.concatenate(output_monotonous, axis=0)

    # print(dict)
    # ----------------------------- labels -------------------------------------------------------------------------
    dict['label_event'] = np.concatenate(label_event, axis=0)

    dict['label_pleasant'] = np.concatenate(label_pleasant, axis=0)
    dict['label_eventful'] = np.concatenate(label_eventful, axis=0)
    dict['label_chaotic'] = np.concatenate(label_chaotic, axis=0)
    dict['label_vibrant'] = np.concatenate(label_vibrant, axis=0)
    dict['label_uneventful'] = np.concatenate(label_uneventful, axis=0)
    dict['label_calm'] = np.concatenate(label_calm, axis=0)
    dict['label_annoying'] = np.concatenate(label_annoying, axis=0)
    dict['label_monotonous'] = np.concatenate(label_monotonous, axis=0)

    return dict


def cal_auc(targets_event, outputs_event):
    # print(targets_event)
    # print(outputs_event)
    #
    # print(targets_event.shape)
    # print(outputs_event.shape)
    aucs = []
    for i in range(targets_event.shape[0]):
        test_y_auc, pred_auc = targets_event[i, :], outputs_event[i, :]
        if np.sum(test_y_auc):
            test_auc = metrics.roc_auc_score(test_y_auc, pred_auc)
            aucs.append(test_auc)
    final_auc_event_branch = sum(aucs) / len(aucs)
    return final_auc_event_branch


def cal_softmax_classification_accuracy(target, predict, average=None, eps=1e-8):
    """Calculate accuracy.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)

    Outputs:
      accuracy: float
    """
    classes_num = predict.shape[-1]

    predict = np.argmax(predict, axis=-1)  # (audios_num,)
    samples_num = len(target)

    correctness = np.zeros(classes_num)
    total = np.zeros(classes_num)

    for n in range(samples_num):

        total[target[n]] += 1

        if target[n] == predict[n]:
            correctness[target[n]] += 1

    accuracy = correctness / (total + eps)

    if average == 'each_class':
        return accuracy

    elif average == 'macro':
        return np.mean(accuracy)

    else:
        raise Exception('Incorrect average!')



def evaluate(model, generate_func, cuda, using_mel=True, using_loudness=True):
    # Forward
    dict = forward(model=model, generate_func=generate_func, cuda=cuda, using_mel=using_mel, using_loudness=using_loudness)

    # mse loss

    # rate_rmse_loss = metrics.mean_squared_error(targets, predictions, squared=False)
    # squared: If True returns MSE value, if False returns RMSE value.

    # AUC
    event_auc = cal_auc(dict['label_event'], dict['output_event'])

    # softmax classification acc


    pleasant_mse = metrics.mean_squared_error(dict['label_pleasant'], dict['output_pleasant'])
    eventful_mse = metrics.mean_squared_error(dict['label_eventful'], dict['output_eventful'])
    chaotic_mse = metrics.mean_squared_error(dict['label_chaotic'], dict['output_chaotic'])
    vibrant_mse = metrics.mean_squared_error(dict['label_vibrant'], dict['output_vibrant'])
    uneventful_mse = metrics.mean_squared_error(dict['label_uneventful'], dict['output_uneventful'])
    calm_mse = metrics.mean_squared_error(dict['label_calm'], dict['output_calm'])
    annoying_mse = metrics.mean_squared_error(dict['label_annoying'], dict['output_annoying'])
    monotonous_mse = metrics.mean_squared_error(dict['label_monotonous'], dict['output_monotonous'])

    return event_auc, \
           pleasant_mse, eventful_mse, chaotic_mse, vibrant_mse, uneventful_mse, calm_mse, annoying_mse, monotonous_mse



def Training_early_stopping(generator, model, monitor, models_dir, batch_size, cuda=config.cuda, epochs=config.epochs, patience=10,
                            lr_init=config.lr_init, using_mel=True, using_loudness=False, model_name=None):
    create_folder(models_dir)

    # Use provided model_name or get model class name if not provided
    if model_name is None:
        model_name = model.__class__.__name__

    # Initialize CSV log files for training and validation metrics
    train_log_file = os.path.join(models_dir, f'{model_name}_train_metrics.csv')
    val_log_file = os.path.join(models_dir, f'{model_name}_val_metrics.csv')
    time_log_file = os.path.join(models_dir, f'{model_name}_time_metrics.csv')

    # Write headers
    with open(train_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'loss_event',
                         'loss_pleasant', 'loss_eventful', 'loss_chaotic', 'loss_vibrant',
                         'loss_uneventful', 'loss_calm', 'loss_annoying', 'loss_monotonous'])

    with open(val_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'val_event_auc', 'val_pleasant_mse', 'val_eventful_mse',
                         'val_chaotic_mse', 'val_vibrant_mse', 'val_uneventful_mse',
                         'val_calm_mse', 'val_annoying_mse', 'val_monotonous_mse'])
        
    with open(time_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_time_s', 'iteration_time_ms', 'validate_time_s', 'inference_time_ms'])


    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.999), eps=1e-08)

    # ------------------------------------------------------------------------------------------------------------------

    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCELoss()

    sample_num = len(generator.train_scene_labels)
    one_epoch = int(sample_num / batch_size)
    print('one_epoch: ', one_epoch, 'iteration is 1 epoch')
    print('Batch size: ', batch_size)
    check_iter = one_epoch
    print('validating every: ', check_iter, ' iteration')

    # initialize the early_stopping object
    model_path = os.path.join(models_dir, f'early_stopping_{model_name}_{monitor}{config.endswith}')
    early_stopping_mse_loss = EarlyStopping(model_path, decrease=True, patience=patience, verbose=True) #Changed decrease to False since AUC needs to increase actually

    training_start_time = time.time()
    # for iteration, all_data in enumerate(generator.generate_train()):
    #     if using_mel:
    #         (batch_x, batch_sound_masker,
    #          batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
    #          batch_uneventful, batch_calm, batch_annoying, batch_monotonous) = all_data
    #     if using_loudness:
    #         (batch_x_loudness, batch_sound_masker,
    #          batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
    #          batch_uneventful, batch_calm, batch_annoying, batch_monotonous) = all_data
    #     if using_mel and using_loudness:
    #         (batch_x, batch_x_loudness, batch_sound_masker,
    #          batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
    #          batch_uneventful, batch_calm, batch_annoying, batch_monotonous) = all_data
            

    for iteration, all_data in enumerate(generator.generate_train()):
        if using_mel and using_loudness:
            (batch_x, batch_x_loudness, batch_sound_masker,
            batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
            batch_uneventful, batch_calm, batch_annoying, batch_monotonous) = all_data
        elif using_mel:
            (batch_x, batch_sound_masker,
            batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
            batch_uneventful, batch_calm, batch_annoying, batch_monotonous) = all_data
        elif using_loudness:
            (batch_x_loudness, batch_sound_masker,
            batch_pleasant, batch_eventful, batch_chaotic, batch_vibrant,
            batch_uneventful, batch_calm, batch_annoying, batch_monotonous) = all_data

             

        if using_mel:
            batch_x = move_data_to_gpu(batch_x, cuda)
        if using_loudness:
            batch_x_loudness = move_data_to_gpu(batch_x_loudness, cuda)

        batch_sound_masker = move_data_to_gpu(batch_sound_masker, cuda)

        # MSE

        # MSE
        batch_pleasant = move_data_to_gpu(batch_pleasant, cuda, using_float=True)
        batch_eventful = move_data_to_gpu(batch_eventful, cuda, using_float=True)
        batch_chaotic = move_data_to_gpu(batch_chaotic, cuda, using_float=True)
        batch_vibrant = move_data_to_gpu(batch_vibrant, cuda, using_float=True)
        batch_uneventful = move_data_to_gpu(batch_uneventful, cuda, using_float=True)
        batch_calm = move_data_to_gpu(batch_calm, cuda, using_float=True)
        batch_annoying = move_data_to_gpu(batch_annoying, cuda, using_float=True)
        batch_monotonous = move_data_to_gpu(batch_monotonous, cuda, using_float=True)

        train_bgn_time = time.time()
        model.train()
        optimizer.zero_grad()

        if using_mel and using_loudness:
            event, \
            pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous = model(batch_x, batch_x_loudness)

        elif using_mel:
            event, \
            pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous = model(batch_x)

        elif using_loudness:
            event, \
            pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous = model(batch_x_loudness)

        loss_event = bce_loss(F.sigmoid(event), batch_sound_masker)

        loss_pleasant = mse_loss(pleasant, batch_pleasant)
        loss_eventful = mse_loss(eventful, batch_eventful)
        loss_chaotic = mse_loss(chaotic, batch_chaotic)
        loss_vibrant = mse_loss(vibrant, batch_vibrant)
        loss_uneventful = mse_loss(uneventful, batch_uneventful)
        loss_calm = mse_loss(calm, batch_calm)
        loss_annoying = mse_loss(annoying, batch_annoying)
        loss_monotonous = mse_loss(monotonous, batch_monotonous)

        loss_common = loss_event + \
                      loss_pleasant + loss_eventful + loss_chaotic + loss_vibrant + \
                      loss_uneventful + loss_calm + loss_annoying + loss_monotonous

        loss_common.backward()
        optimizer.step()

        Epoch = iteration / one_epoch

        with open(train_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([Epoch, float(loss_common), float(loss_event),
                            float(loss_pleasant), float(loss_eventful), float(loss_chaotic), 
                            float(loss_vibrant), float(loss_uneventful), float(loss_calm), 
                            float(loss_annoying), float(loss_monotonous)])

        print('epoch: ', '%.3f' % (Epoch), 'loss: %.3f' % float(loss_common),
              'event: %.3f' % float(loss_event),

              'plea: %.3f' % float(loss_pleasant), 'eventf: %.3f' % float(loss_eventful),
              'chao: %.3f' % float(loss_chaotic), 'vib: %.3f' % float(loss_vibrant),
              'uneve: %.3f' % float(loss_uneventful), 'calm: %.3f' % float(loss_calm),
              'ann: %.3f' % float(loss_annoying), 'mono: %.3f' % float(loss_monotonous))
              

        if iteration % check_iter == 0 and iteration > 1:
            train_fin_time = time.time()
            # Generate function
            generate_func = generator.generate_validate(data_type='validate')
            val_event_auc, \
            val_pleasant_mse, val_eventful_mse, val_chaotic_mse, val_vibrant_mse, \
            val_uneventful_mse, val_calm_mse, val_annoying_mse, val_monotonous_mse = evaluate(model=model,
                                                              generate_func=generate_func,
                                                              cuda=cuda, using_mel=using_mel, using_loudness=using_loudness)

            # Write validation metrics to CSV
            with open(val_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([Epoch, float(val_event_auc), float(val_pleasant_mse), 
                                float(val_eventful_mse), float(val_chaotic_mse), 
                                float(val_vibrant_mse), float(val_uneventful_mse),
                                float(val_calm_mse), float(val_annoying_mse), 
                                float(val_monotonous_mse)])

            print('E: ', '%.3f' % (Epoch), # 'P8_mean: %.3f' % float(PAQ8_mean_mse),
                  'val_event: %.3f' % float(val_event_auc),
                  'val_plea: %.3f' % float(val_pleasant_mse), 'val_even: %.3f' % float(val_eventful_mse),
                  'val_chao: %.3f' % float(val_chaotic_mse), 'val_vibr: %.3f' % float(val_vibrant_mse),
                  'val_uneve: %.3f' % float(val_uneventful_mse), 'val_calm: %.3f' % float(val_calm_mse),
                  'val_anno: %.3f' % float(val_annoying_mse), 'val_mono: %.3f' % float(val_monotonous_mse))

            train_time = train_fin_time - train_bgn_time
                

            validation_end_time = time.time()
            validate_time = validation_end_time - train_fin_time
            print('epoch: {}, train time: {:.3f} s, iteration time: {:.3f} ms, validate time: {:.3f} s, '
                  'inference time : {:.3f} ms'.format('%.2f' % (Epoch), train_time,
                                                      (train_time / sample_num) * 1000, validate_time,
                                                      1000 * validate_time / sample_num))
            # Write time metrics to CSV
            with open(time_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                iteration_time_ms = (train_time / sample_num) * 1000
                inference_time_ms = 1000 * validate_time / sample_num
                writer.writerow([Epoch, train_time, iteration_time_ms, validate_time, inference_time_ms])    
            #------------------------ validation done ------------------------------------------------------------------

            # -------- early stop---------------------------------------------------------------------------------------
            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            # if monitor == 'ISOPls':
            #     early_stopping_mse_loss(val_ISOPls_mse, model)
            # if monitor == 'ISOEvs':
            #     early_stopping_mse_loss(val_ISOEvs_mse, model)
            if monitor == 'pleasant':
                early_stopping_mse_loss(val_pleasant_mse, model)

            if early_stopping_mse_loss.early_stop:
                finish_time = time.time() - training_start_time
                print('Model training finish time: {:.3f} s,'.format(finish_time))
                print("Early stopping")

                save_out_path = os.path.join(models_dir, f'final_model_{model_name}{config.endswith}')

                torch.save(model.state_dict(), save_out_path)  # <-- save it as final      ADDED BY ME
                print('✅ Final model (best validation score) saved to:', save_out_path)

                print('Model training finish time: {:.3f} s,'.format(finish_time))
                # print('Model training finish time: {:.3f} s,'.format(finish_time))
                # print('Model training finish time: {:.3f} s,'.format(finish_time))

                print('Training is done!!!')

                break

        # Stop learning
        if iteration > (epochs * one_epoch):
                finish_time = time.time() - training_start_time
                print('Model training finish time: {:.3f} s,'.format(finish_time))
                print("All epochs are done.")

                save_out_path = os.path.join(models_dir, f'final_model_{model_name}{config.endswith}')

                torch.save(model.state_dict(), save_out_path)  # <-- save it as final     ADDED BY ME
                print('✅ Final model (best validation score) saved to:', save_out_path)

                print('Model training finish time: {:.3f} s,'.format(finish_time))
                print('Model training finish time: {:.3f} s,'.format(finish_time))
                print('Model training finish time: {:.3f} s,'.format(finish_time))

                print('Training is done!!!')

                print('Training is done!!!')


                break







