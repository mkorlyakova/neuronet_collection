import  tensorflow as tf
from tensorflow import keras
from tensorflow import GradientTape

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras.utils import losses_utils

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf

import matplotlib.pyplot as plt



import numpy as np

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import auto_enc
from generator_auto import CustomDataGenFace_fft_image, CustomDataGenFace_label_image, CustomDataGenFace_dm_label_image, \
    CustomDataGenFace_image, CustomDataGenFace_dm_label_image_new, CustomDataGenFace_test
from readcsvdata import read_data
from init_confir import config_model, tostring_model_name

import matplotlib.pyplot as plt

# from Bakay
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# from TF
# tf.config.experimental_run_functions_eagerly(True)
tf.config.run_functions_eagerly(True)
gpu_ = 1
vae_ = 1
# fft in samples
fft_ = True
# image in samples
image_ = False
# set parametr from model_config.cfg
# текущие каталоги куда запишем промежуточные результаты
file_desktopt, file_mobilet, file_desktop, file_mobile, file_name, cwd, path_label, path_image, \
numb_step, n_train, n_test, epoch, \
filters_disc, down_bloc_disc, up_bloc_gen, down_bloc_gen, rez_bloc_gen, \
gpu_, SIZE, load_models, batch_size_model, lr_model, \
model_name, model_path, file_name, desctop_train, mobile_train, desctop_test, mobile_test = config_model(gpu_)
gpu_ = 1
print(SIZE)
m1 = n_test  # batch_size тест

m2 = numb_step  # train+test subset number
m3 = n_train  # train subset number

print(type(lr_model), batch_size_model)
# name list from .csv
list_mob = []
list_mobt = []
list_desk = []
list_deskt = []
n_test1 = 0
n_test2 = 2
n_train1, n_train2 = 0, 2

if gpu_ == 0:
    # print(os.path.join(file_name,file_mobile) )
    batch_size_model = 1
    list_mob = os.listdir(os.path.join(file_name, file_mobile))
    list_desk = os.listdir(os.path.join(file_name, file_desktop))
    list_mobt = os.listdir(os.path.join(file_name, file_mobilet))
    list_deskt = os.listdir(os.path.join(file_name, file_desktopt))
    list_spoof_mob = os.listdir(os.path.join(file_name, 'spoofing'), )
    list_spoof_mobt = os.listdir(os.path.join(file_name, 'spoofing_test'), )
    list_desk = [os.path.join(file_name, file_desktop, file_i) for file_i in list_desk]
    list_deskt = [os.path.join(file_name, file_desktopt, file_i) for file_i in list_deskt]
    list_mob = [os.path.join(file_name, file_mobile, file_i) for file_i in list_mob]
    list_mobt = [os.path.join(file_name, file_mobilet, file_i) for file_i in list_mobt]

    tm = np.zeros((len(list_mob)))
    tmt = np.zeros((len(list_mobt)))
    td = np.zeros((len(list_desk)))
    tdt = np.zeros((len(list_deskt)))
    tms = np.ones(len(list_spoof_mob))
    tmst = np.ones(len(list_spoof_mob))
    list_mob = list_mob + [os.path.join(file_name, 'spoofing', file_i) for file_i in list_spoof_mob]
    list_mobt = list_mobt + [os.path.join(file_name, 'spoofing_test', file_i) for file_i in list_spoof_mobt]

    tm = np.hstack([tm, tms])
    tmt = np.hstack([tmt, tmst])
    # tm = np.random.randint(0,1,(len(list_mob)))
    # tmt = np.random.randint(0, 1, (len(list_mobt)))
    # td = np.random.randint(0,1,(len(list_desk)))
    # tdt = np.random.randint(0, 1, (len(list_deskt)))
    file_name, file_mobile, file_desktop, file_mobilet, file_desktopt = '', '', '', '', ''

if gpu_:
    list_mob, list_mobt, list_desk, list_deskt, n_test1, n_test2, n_train1, n_train2, tm, tmt, td, tdt = read_data(
        path_name='')

print(len(list_mob), len(tm), )
#

N_split = 3

lab = 1

if fft_ & image_:
    n_chanel = 9
else:
    n_chanel = 3

kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

# Gamma initializer for instance normalization.

gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

SIZE = [224, 224]
# SIZE = [56,56]

n_train = 1
print('data_gen:', len(list_mobt), len(list_deskt), n_train)

# custom data generator (from Sequens)


# train generator
if vae_:
    data_gen = CustomDataGenFace_test(tm=tm, td=td, file_mobile=list_mob, file_desk=list_desk,
                                                    path_m=os.path.join(file_name, file_mobile),
                                                    path_d=os.path.join(file_name, file_desktop), W=SIZE[0],
                                                    n_train=n_train, input_size=(SIZE[0], SIZE[0], n_chanel),
                                                    batch_size=20, fft_= 0 , image_=image_)
    print(data_gen.mobile_len, data_gen.batch_size)

for i in range(1):
    jj,ll = data_gen[i]

#
# list_g = ['/OPT/LABELER/STATIC/FRAMES/HALYK/9AAC81D0-9B4F-46Ad-b725-0e56cd90d3a3/4cb5a4ab-b0df-4386-9826-7374abaf91e9.mp4/4cb5a4ab-b0df-4386-9826-7374abaf91e9.mp4.0001.jpg',
#           '/opt/labeler/static/frames/halyk/9aac81d0-9b4f-46ad-b725-0e56cd90d3a3/4cb5a4ab-b0df-4386-9826-7374abaf91e9.mp4/4cb5a4ab-b0df-4386-9826-7374abaf91e9.mp4.0001.jpg',
#           '/opt/labeler/static/frames/halyk/9aac81d0-9b4f-46ad-b725-0e56cd90d3a3/4cb5a4ab-b0df-4386-9826-7374abaf91e9.mp4/4cb5a4ab-b0df-4386-9826-7374abaf91e9.mp4.0001.jpg']
# import pandas as pd
#
# df = pd.read_csv('mobile_train.csv')
# for ll in list_g:
#     print(df.loc[df.iloc[:,0]==ll,:])
# df = pd.read_csv('desktop_train.csv')
# for ll in list_g:
#     print(df.loc[df.iloc[:,0]==ll,:])
# df = pd.read_csv('mobile_test.csv')
# for ll in list_g:
#     print(df.loc[df.iloc[:,0]==ll,:])
# df = pd.read_csv('desktop_test.csv')
# for ll in list_g:
#    print(df.loc[df.iloc[:,0]==ll,:])