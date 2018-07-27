import numpy as np
import os

import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.losses import  categorical_crossentropy

from annotation.dice_koef import (
    dice_coef, dice_coef_loss, dice_coef_multilabel_quad
)
#модель гнойшый - трехгралавя юнет с категориальной кросэнтропией и софтмаксом
# на конце, работает, но закоментирована - т.к. не приносит резкого улучшения по сравнению с дайсом
def unet_trihead_permute(seg_len):
    input_size = (seg_len, 1)
    inputs = Input(input_size)
    conv1 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling1D(pool_size= 2)(conv1)
    conv2 = Conv1D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv1D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling1D(pool_size= 2)(conv2)
    conv3 = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling1D(pool_size= 2)(conv3)
    conv4 = Conv1D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv1D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling1D(pool_size=2)(drop4)

    conv5 = Conv1D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv1D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv1D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling1D(size= 2)(drop5))
    merge6 = concatenate([drop4, up6], axis=2)
    conv6 = Conv1D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv1D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv1D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling1D(size=2)(conv6))
    merge7 = concatenate([conv3, up7], axis=2)
    conv7 = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv1D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling1D(size= 2)(conv7))
    merge8 = concatenate([conv2, up8], axis=2)
    conv8 = Conv1D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv1D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling1D(size=2)(conv8))
    merge9 = concatenate([conv1, up9], axis=2)

    # теперь на выход три головы
    # первая
    conv9_1 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9_1 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_1)
    conv9_1 = Conv1D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_1)
    conv10_1 = Conv1D(1, 1, activation='sigmoid', name='head_1')(conv9_1)

    # вторая
    conv9_2 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9_2 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_2)
    conv9_2 = Conv1D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_2)
    conv10_2 = Conv1D(1, 1, activation='sigmoid', name='head_2')(conv9_2)

    # третья
    conv9_3 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9_3 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_3)
    conv9_3 = Conv1D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_3)
    conv10_3 = Conv1D(1, 1, activation='sigmoid', name='head_3')(conv9_3)

    # 4 бекграунд, чтобы категорная кросэентропия с софтмаксом могли всегда выбрать победителя!
    conv9_4 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9_4 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_4)
    conv9_4 = Conv1D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_4)
    conv10_4 = Conv1D(1, 1, activation='sigmoid', name='head_4')(conv9_4)

    out = concatenate([conv10_1, conv10_2, conv10_3, conv10_4], axis=2)
    out = Permute((2, 1))(out)
    out = Activation('sigmoid')(out) # ищет победителя среди одного пикселя на 4 масках
    model = Model(inputs=inputs, outputs=out, name="unet_trihead")

    model.compile(optimizer=Adam(lr=1e-4), loss=categorical_crossentropy, metrics=[dice_coef])

    model.summary()
    return model

if __name__ == "__main__":
    unet_trihead_permute(seg_len=512)
