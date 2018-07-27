# -*- coding: utf-8 -*
import os

import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.model_selection import train_test_split

from annotation.ann_generator import (
    get_mulimask_generator, get_mulimask_generator_addon
)
from annotation.dice_koef import (
    get_custom_objects
)
from annotation.models.model_yana import unet_trihead
from utils import open_pickle
from utils import save_history

dataset_path = "./DSET_argentina.pkl"
segment_len=512


def split_dict_annotations(dict_dataset):
    # бьем датасет-мапу на трейновую и тестовую
    x = dict_dataset['x']
    ann = dict_dataset['ann']
    X_train, X_test, ann_train, ann_test = train_test_split(x, ann, test_size=0.33, random_state=42)
    return {'x':X_train, 'ann':ann_train},{'x':X_test, 'ann':ann_test}

def get_generators(train_batch, test_batch):
    """
    слепим два генератора- тестовый и трейновый
    :param train_batch: размер батча для трейнового генератора
    :param test_batch: размер батча для тестового генератора
    :return:
    """
    dataset_in = open_pickle(dataset_path)
    train_dset, test_dset = split_dict_annotations(dataset_in)
    my_generator_train = get_mulimask_generator(segment_len, batch_size=train_batch, dataset_in=train_dset)
    my_generator_test = get_mulimask_generator(segment_len, batch_size=test_batch, dataset_in=test_dset)
    return my_generator_train, my_generator_test

def get_generators_permute(train_batch, test_batch):
    """чтобы возвращаелось none,4,512, а не none, 513, 3 как обычно
    """
    dataset_in = open_pickle(dataset_path)
    train_dset, test_dset = split_dict_annotations(dataset_in)
    my_generator_train = get_mulimask_generator_addon(segment_len, batch_size=train_batch, dataset_in=train_dset)
    my_generator_test = get_mulimask_generator_addon(segment_len, batch_size=test_batch, dataset_in=test_dset)
    return my_generator_train, my_generator_test

def get_model():
    #return unet(seg_len=segment_len)
    #return unet_yana(seg_len=segment_len)
    return unet_trihead(seg_len=segment_len)
    #return  unet_trihead_permute(seg_len=segment_len)


def train(name, need_permute=False):
    model = get_model()
    if need_permute:
        generator_train, generator_test = get_generators_permute(train_batch=15, test_batch=50)
    else:
        generator_train, generator_test = get_generators(train_batch=15, test_batch=50)
    history = model.fit_generator(generator=generator_train,
                                  steps_per_epoch=40,
                                  epochs=45,
                                  validation_data=generator_test,
                                  validation_steps=1)

    save_history(history, name)
    model.save(name + '.h5')

def eval_models_in_folder(num_pictures):
    """
    размечаем обученной моделью случайные сегменты экг-шек, рисуем картинки
    :param num_pictures: кол-во сегментов ЭКГ (взятых случайным образом), на которых хотим увидеть работу модели
    :return:
    """
    _, generator_test = get_generators(train_batch=0, test_batch=num_pictures)
    batch = next(generator_test)

    folder = "./"
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(".h5"):
            model = load_model(os.path.join(folder,filename),custom_objects=get_custom_objects())
            test_model_multimask(model, batch, name="VIS_"+filename[0:-len(".h5")])


def test_model_multimask(model, batch, name):
    """

    :param model: бученная модель
    :param batch: батч из тестовго генератора, (x, ann)
    :param name: имя для серии картинок
    :return:
    """
    x = batch[0]
    ann = batch[1]

    print("модель предсказывает на х с формой " + str(x.shape))
    predictions = model.predict_on_batch(x)
    #predictions = np.swapaxes(predictions, 2, 1) -- если permute
    print("предсказания (ann) имеют форму " + str(predictions.shape))
    print ("кол-во предсказаний = " + str(len(predictions)))
    print("кол-во экг-шек = " + str(len(x)))

    t = range(0, len(x[0]))
    for i in range(len(x)):
        print("bo!")
        print (len(x[i]))
        print (len(ann[i]), "len ann i")
        print(ann[i].shape, "sh ann i")
        plot_name = "VIS" + name + "_" + str(i) + ".png"
        #f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=False, sharex=False)

        gridsize = (2, 3)
        fig = plt.figure(figsize=(8, 5))
        ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=1)
        ax2 = plt.subplot2grid(gridsize, (1, 0), rowspan=1)
        ax3 = plt.subplot2grid(gridsize, (1, 1), rowspan=1)
        ax4 = plt.subplot2grid(gridsize, (1, 2), rowspan=1)


        ax1.plot(x[i], 'k-', label="ЭКГ", alpha=0.6)
        y_base = 12
        ax1.axhline(y=y_base, linestyle='--', color='m')
        ax1.fill_between(t, 0, 10, alpha=0.6, where=ann[i,:,0] > 0.6, label="правильн.отв.0", facecolor='red')
        ax1.fill_between(t, 0, 10, alpha=0.6, where=ann[i, :, 1] > 0.6, label="правильн.отв.1", facecolor='green')
        ax1.fill_between(t, 0, 10, alpha=0.6, where=ann[i, :, 2] > 0.6, label="правильн.отв.2", facecolor='blue')

        d = 4
        ax1.fill_between(t, y_base, y_base+10, alpha=0.5, where=predictions[i,:,0] > 0.5, facecolor='red')
        ax1.fill_between(t, y_base+d, y_base + 10+d, alpha=0.5, where=predictions[i,:,1] > 0.5, facecolor='green')
        ax1.fill_between(t, y_base+2*d, y_base + 10+2*d, alpha=0.5, where=predictions[i,:,2] > 0.5, facecolor='blue')

        ax2.set_ylim([0, 1.1])
        ax2.plot(predictions[i, :, 0],'k-',  alpha=0.6)
        ax2.fill_between(t, 0, 1, alpha=0.6, where=ann[i, :, 0] > 0.6, label="правильн.отв.0", facecolor='red')

        ax3.set_ylim([0, 1.1])
        ax3.fill_between(t, 0, 1, alpha=0.6, where=ann[i, :, 1] > 0.6, label="правильн.отв.1", facecolor='green')
        ax3.plot(predictions[i, :, 1],'k-', alpha=0.6)

        ax4.set_ylim([0, 1.1])
        ax4.fill_between(t, 0, 1, alpha=0.6, where=ann[i, :, 2] > 0.6, label="правильн.отв.2", facecolor='blue')
        ax4.plot(predictions[i, :, 2], 'k-', alpha=0.6)
        plt.legend(loc=2)
        plt.savefig(plot_name)
        plt.clf()


    print("картинки сохранены!")

if __name__ == "__main__":
    name = "oxxy_quad_annotator_long"

    train(name)
    eval_models_in_folder(45)



