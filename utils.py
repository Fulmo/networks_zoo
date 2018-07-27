# -*- coding: utf-8 -*
import matplotlib.pyplot as plt
import numpy as np
import easygui
import pickle as pkl
from keras.models import load_model
#plt.xkcd()

def draw_reconstruction_to_png(ecg_true, ecg_predicted, png_filename):
    """

    :param ecg_true: истинная экг
    :param ecg_predicted: предсказанное
    :param png_filename: имя для файла с картинкой
    :return:
    """
    ecg_true = reshape_ecg_tensor(ecg_true)
    ecg_predicted = reshape_ecg_tensor(ecg_predicted)

    assert ecg_true.shape == ecg_predicted.shape

    len_of_time = len(ecg_true[0])
    t = [i for i in range(len_of_time)]


    rows = len(ecg_true)  # кол-во каналов
    cols = 2              # true и predicted - 2 столбика
    f, axarr = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)
    for i in range(rows):
        true_i_chanel = ecg_true[i]
        predicted_i_chanel = ecg_predicted[i]
        axarr[i, 0].plot(t, true_i_chanel)
        axarr[i, 1].plot(t, predicted_i_chanel)


    plt.savefig(png_filename+".png")

def reshape_ecg_tensor(ecg):
    # превратим (252, 1, 12) в (12, 252)
    print ("форма тезора с экг =  " + str(ecg.shape))
    ecg = ecg[:,0,:]
    ecg = np.transpose(ecg)
    print ("форма тезора с экг (после напильника) =" + str(ecg))
    return ecg

def save_history(history, canterpillar_name):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(canterpillar_name+"_loss.png")

    if 'acc' in history.history.keys():
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model loss')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(canterpillar_name + "_acc.png")

def show_reconstruction_by_ae(ecg_sample, name):
    filepath = easygui.fileopenbox("выберите файл с обученной моделью .h5")
    trained_model = load_model(filepath)
    trained_model.summary()
    ecg_sample = np.array([ecg_sample])
    prediction = trained_model.predict(ecg_sample)

    draw_reconstruction_to_png(ecg_sample[0],prediction[0], name)

def get_addon_mask(annotations_in):
    addon = np.zeros((1, annotations_in.shape[1], 1))
    
    for i in range(0, annotations_in.shape[1]):
        sum = annotations_in[:, i, 0] + annotations_in[:, i, 1] + annotations_in[:, i, 2]
        if (sum == 0):
                addon[0:1, i, 0:1] = 1
        else:
            continue
    return addon

def open_pickle(path):
    infile = open(path, 'rb')
    load_pkl = pkl.load(infile)
    infile.close()
    return load_pkl

