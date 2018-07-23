# -*- coding: utf-8 -*
import matplotlib.pyplot as plt
import numpy as np
import easygui
import pickle as pkl
from keras.models import load_model
import os
from caterpillar_feeder import (
    annotated_generator, annotated_generator_simple
)
from utils import open_pickle
from sklearn.model_selection import train_test_split
from annotation.model import unet
from annotation.one_lead_one_mask import unet_simple
from utils import save_history
from annotation.trained_model_testing import test_model
from annotation.dice_koef import (
    dice_coef, dice_coef_loss
)

IS_SIMPLE= True
dataset_path = "./DSET_argentina.pkl"
segment_len=512

def split_dict_annotations(dict_dataset):
    # бьем датасет-мапу на трейновую и тестовую
    x = dict_dataset['x']
    ann = dict_dataset['ann']
    X_train, X_test, ann_train, ann_test = train_test_split(x, ann, test_size=0.33, random_state=42)
    return {'x':X_train, 'ann':ann_train},{'x':X_test, 'ann':ann_test}

def get_generators(train_batch, test_batch, is_simple=False):
    """
    слепим два генератора- тестовый и трейновый
    :param train_batch: размер батча для трейнового генератора
    :param test_batch: размер батча для тестового генератора
    :return:
    """
    dataset_in = open_pickle('./DSET_argentina.pkl')
    train_dset, test_dset = split_dict_annotations(dataset_in)
    my_generator_train = annotated_generator(segment_len, batch_size=train_batch, dataset_in=train_dset)
    my_generator_test = annotated_generator(segment_len, batch_size=test_batch, dataset_in=test_dset)

    if is_simple is False:
        return my_generator_train, my_generator_test
    return annotated_generator_simple(my_generator_train), annotated_generator_simple(my_generator_test)

def get_model():
    #return unet(seg_len=segment_len)
    return unet_simple(seg_len=segment_len)

def train(name):
    model = get_model()
    generator_train, generator_test = get_generators(train_batch=15, test_batch=50,is_simple=IS_SIMPLE)
    history = model.fit_generator(generator=generator_train,
                                  steps_per_epoch=40,
                                  epochs=10,
                                  validation_data=generator_test,
                                  validation_steps=1)

    save_history(history, name)
    model.save(name + '.h5')


def eval_models_in_folder(num_pictures):
    _, generator_test = get_generators(train_batch=0, test_batch=num_pictures, is_simple=IS_SIMPLE)
    batch = next(generator_test)

    folder = "./"
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(".h5"):
            model = load_model(os.path.join(folder,filename),custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef':dice_coef})
            test_model(model, batch, name="VIS_"+filename[0:-len(".h5")], is_simple=IS_SIMPLE)


if __name__ == "__main__":
    name = "mila_annotator"

    #train(name)
    eval_models_in_folder(3)









