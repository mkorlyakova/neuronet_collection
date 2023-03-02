# генераторы
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout,Flatten, Reshape
from tensorflow.keras import backend as K
from tensorflow import GradientTape
#
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers  import concatenate, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python import tf2
from tensorflow.python.keras.utils import losses_utils

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import matplotlib.pyplot as plt
import numpy as np
import os

from preproc import preprocess_val

# fft batch
def fft_y(image):
    #D = tf.math.
    if image.shape[0] is not None:
        
        Flist =[]
        for i in range(image.shape[0]):
            #print(i,type(image),image.shape,image.dtype)
            F = fft_img(image[i,:,:,:3])
            F[:,:,3:6] = F[:,:,3] * D[:,:,:]
            
            if F is not None:

                Flist.append(tf.reshape(F,[1,224,224,9]))

        fft_image_y = tf.concat(Flist,axis=0)
        return tf.cast(fft_image_y,dtype =tf.float32)
    else:
        return tf.zeros_like(image)

#fft from image
def fft_img(image):
    # image - вход 3Д 
    n = image.shape[0]
    i = tf.cast([list(range(n))], dtype=tf.float32) - n // 2
    h = tf.math.exp(-(i/n * 3)**2)
    H = tf.transpose(h) * h
    H = tf.reshape(H,[n,n,1])
    H = tf.concat([H,H,H], axis=-1)
    print(H.shape)

    if image.shape[0] is not None:
        ffs1 = tf.math.log(tf.signal.fft3d(tf.cast(image, dtype = tf.complex64)))
        ffti = tf.math.abs(ffs1)#.numpy()

        ffti= tf.math.multiply(ffti, H)
        fftr = tf.math.angle(ffs1)#.numpy()
        mm = tf.cast(tf.reduce_max(tf.math.abs(ffti)),dtype=tf.float32)#.numpy()
        ff = 3.14
        ffti =tf.cast(ffti/mm,dtype=tf.float32)
        fftr = tf.cast(fftr/ff,dtype=tf.float32)
        #print(fftr)
        image =tf.concat([image,ffti,fftr],axis=-1)
        #print(image.shape,type(image))
        #image = tf.Variable(tf.transpose(image,perm = [1, 2, 0]))

    #print('fin fft:',image.shape)

    return image

import pandas as pd

# Базовый генератор
class CustomDataGen_Base(Sequence):

    def __init__(self, tm=[], td=[], file_mobile='', file_desk='', path_m='', path_d='',
                 batch_size=2,
                 input_size=(112, 112, 3),
                 s=100,
                 W=112,
                 lab=0,
                 shuffle=True, fft_=False, image_=True, n_train=0, random_ind  = True, hsv = False):
        '''
        self, 
        tm=[], td=[],  - список меток для  файлов
        file_mobile='', - список имен для класса 1
        file_desk='', - список имен для класса 0
        path_m='', путь до класса 1
        path_d='',  путь до класса 0
        batch_size=2,
        input_size=(112, 112, 3),
        s=100,
        W=112, - размер картинки
        lab=0,
        shuffle=True, 
        fft_=False, 
        image_=True, 
        n_train=0, 
        random_ind  = True, 
        sv = False
        '''
        self.batch_size = batch_size  #
        self.input_size = input_size  #
        self.shuffle = shuffle
        self.mobile_ = len(file_mobile)
        self.desktop_ = len(file_desk)

        if n_train == 0:
            self.mobile_len = len(file_mobile) - batch_size  # 1
        else:
            self.mobile_len = n_train * batch_size
        self.s = s
        self.W = W
        self.lab = lab  # РїРµСЂРµР№С‚Рё РІ Lab

        self.desk_len = len(file_desk)  #
        self.file_mobile = file_mobile,
        self.path_m = path_m,
        self.file_desk = file_desk,
        self.path_d = path_d,
        self.fft_ = fft_,
        self.image_ = image_
        self.tagm = tm,
        self.tagd = td,
        self.random_ind = random_ind
        self.hsv = hsv
        #self.dfm = pd.read_csv('mobile_train.csv')
        #self.dfd = pd.read_csv('desktop_train.csv')

    def on_epoch_end(self):
        pass
    def image_preproc(self,path_):
        # preprocess image
        pass
    def returned_batch(self, batch_image_input, batch_image_output,desk_mob, label):
        pass


    def __getitem__(self, index):
        #print(index)
        #

        batch_image_input = []
        batch_image_output = []
        label = []
        desk_mob = []
        # get random samples
        if self.random_ind:
            ind1 = tf.random.uniform(shape=[self.batch_size], minval=0, maxval=self.mobile_ - self.batch_size,
                                     dtype=tf.int32).numpy()
            ind2 = tf.random.uniform(shape=[self.batch_size], minval=0, maxval=self.desktop_ - self.batch_size,
                                     dtype=tf.int32).numpy()
        else:
            ind1 = [i for i in range(index * self.batch_size, self.batch_size * (index + 1))]
            ind2 = [i for i in range(index * self.batch_size, self.batch_size * (index + 1))]


        for i in range(self.batch_size):
            try:
                # почему то были [self.tagm[0][ind1[i]], self.tagd[ind2[i]]]
                list_buff = [self.tagm[0][ind1[i]], self.tagd[0][ind2[i]]]

                file_m = os.path.join(self.path_m[0], self.file_mobile[0][ind1[i]])
                file_d = os.path.join(self.path_d[0], self.file_desk[0][ind2[i]])

                for k, path_ in enumerate([file_m, file_d]):
                    try:
                        image_desk, image_mob =self.image_preproc(path_)

                        if self.fft_[0]:
                            # print('fft')

                            image_desk = fft_img(image_desk)
                            image_mob = fft_img(image_mob)

                            if self.image_:
                                # fft and image
                                batch_image_input.append(image_desk[:, :, :].numpy())
                                batch_image_output.append(image_mob[:, :, :].numpy())
                            else:
                                # only fft
                                batch_image_input.append(image_desk[:, :, 3:6].numpy())
                                batch_image_output.append(image_mob[:, :, 3:6].numpy())
                            label.append(list_buff[k] * 1.0)
                            desk_mob.append(k * 1.0)
                        else:
                            # only image
                            batch_image_input.append(image_desk[:, :, :].numpy())
                            batch_image_output.append(image_mob[:, :, :].numpy())
                            label.append(list_buff[k] * 1.0)
                            desk_mob.append(k * 1.0)
                    except:
                        print('input error')
                        with open('log_path.txt','a') as f:
                            f.write(path_+'\n')
                            f.close()
                        # return np.array(batch_image_input), np.array(batch_image_output), np.array(label * 1.0)
            except:
                print(
                    'error5')  # return np.array(batch_image_input), np.array(batch_image_output), np.array(label * 1.0)
                with open('log_path.txt','a') as f:
                    f.write(file_m,'\n',file_d,'\n')
                    f.close()


        return self.returned_batch(np.array(batch_image_input), np.array(batch_image_output),np.array(desk_mob), np.array(label))

    def __len__(self):

        return self.mobile_len // self.batch_size



# рабочий вариант генератора
class CustomDataGenFace_test(CustomDataGen_Base):



    def image_preproc(self, path_):
        #print(path_)
        return tf.zeros(shape =  [224,224,3]), tf.zeros(shape =[224,224,3])

    def returned_batch(self, batch_image_input, batch_image_output, desk_mob, label):
        return batch_image_input, batch_image_output


# строит картинки в разных масштабах относительно одного примера (only from desk or only from mobile)
# (batch_image_input - 224х224х3 - сжато из 448х448х3, np.array(batch_image_output - вырезан из сырого кадра 224х224х3)
class CustomDataGenFace_image(CustomDataGen_Base):

    # препроцессино
    def image_preproc(self, path_):
        image_bytes = tf.io.read_file(path_)
        rgb = tf.image.decode_image(image_bytes, channels=3)
        # print(rgb.shape)
        #
        rgb = tf.cast(rgb / 255, dtype=tf.float32)
        rgb.set_shape((None, None, 3))
        initial_width = tf.shape(rgb)[-2]
        initial_height = tf.shape(rgb)[-3]
        w2 = (self.input_size[1] // 2)
        h2 = (self.input_size[0] // 2)
        w = (self.input_size[1])
        h = (self.input_size[0])

        if initial_height < initial_width:
            dy = initial_height // 2
            dx = initial_width // 2
            d = dy
        else:
            dy = initial_height // 2
            dx = initial_width // 2
            d = dx
        d = d - 10
        if (2 * d <= (initial_width-20)) & (2 * d <= (initial_height-20)):
            x =dx + np.random.randint(-5,5)
            y = dy + np.random.randint(-5,5)

        else:
            x = dx
            y = dy
            d = d // 2

        image_mob = rgb[y - d:y + d, x - d:x + d, :]
        image_mob = tf.image.resize(image_mob, [self.input_size[0], self.input_size[0]])
        image_desk = rgb[y - d:y + d, x - d:x + d, :]
        image_desk = tf.image.resize(image_desk, [self.input_size[0], self.input_size[0]])

        return image_desk,image_mob

    def returned_batch(self, batch_image_input, batch_image_output,desk_mob, label):
        return batch_image_input, batch_image_output

# генератор с препроцессингом FFT
class CustomDataGenFace_fft_image(CustomDataGen_Base):

    def image_preproc(self,path_):
        image_bytes = tf.io.read_file(path_)
        rgb = tf.image.decode_image(image_bytes, channels=3)
        rgb = tf.cast(rgb / 255, dtype=tf.float32)
        rgb.set_shape((None, None, 3))
        initial_width = tf.shape(rgb)[-2]
        initial_height = tf.shape(rgb)[-3]
        w2 = (self.input_size[1] // 2)
        h2 = (self.input_size[0] // 2)
        w = (self.input_size[1])
        h = (self.input_size[0])

        if w > h:
            h = w
        else:
            w = h
        if (2 * w < initial_width) & (2 * h < initial_height):
            x = np.random.randint(w + 1, initial_width - w - 1)
            y = np.random.randint(h + 1, initial_height - h - 1)
        else:
            x = w
            y = h
        # center crop image
        image_mob = rgb[y - h2:y + h2, x - w2:x + w2, :]
        image_desk = rgb[y - h2:y + h2, x - w2:x + w2, :]
        return image_desk,image_mob
    def returned_batch(self, batch_image_input, batch_image_output,desk_mob, label):
        return batch_image_input, batch_image_output

# generate image and label (atak/ok)
# balansed from desktop/mobile

class CustomDataGenFace_label_image(CustomDataGen_Base):

    def image_preproc(self,path_):
        image_bytes = tf.io.read_file(path_)
        rgb = tf.image.decode_image(image_bytes, channels=3)
        rgb = tf.cast(rgb / 255, dtype=tf.float32)
        rgb.set_shape((None, None, 3))
        initial_width = tf.shape(rgb)[-2]
        initial_height = tf.shape(rgb)[-3]
        w2 = (self.input_size[1] // 2)
        h2 = (self.input_size[0] // 2)
        w = (self.input_size[1])
        h = (self.input_size[0])
        if initial_height < initial_width:
            dy = initial_height // 2
            dx = initial_width // 2
            d = dy
        else:
            dy = initial_height // 2
            dx = initial_width // 2
            d = dx
        d = d - 10
        if (2 * d <= (initial_width-20)) & (2 * d <= (initial_height-20)):
            if np.random.randint(0,2)==0:
                x=d + 10 + np.random.randint(-5,5)
                y = dy + np.random.randint(-5,5)

            else:
                x =initial_width-10 - d
                y = dy

        else:
            x = dx
            y = dy
            d = d // 2

        # central crop(448x448) & resize(224x224)
        image_mob = rgb[y - d:y +d, x - d:x + d, :]*2-1
        image_mob = tf.image.resize(image_mob, [self.input_size[0], self.input_size[0]])
        image_desk = rgb[y - d:y + d, x - d:x + d, :]*2-1
        image_desk = tf.image.resize(image_desk, [self.input_size[0], self.input_size[0]])
        return image_desk,image_mob
    def returned_batch(self, batch_image_input, batch_image_output,desk_mob, label):
        return (batch_image_input, batch_image_output),label

#generator : image, desk/mobile(1/0), label (atak/ok - 1/0)
class CustomDataGenFace_dm_label_image(CustomDataGen_Base):

    def image_preproc(self, path_):
        # preprocess image
        pass
        image_bytes = tf.io.read_file(path_)
        rgb = tf.image.decode_image(image_bytes, channels=3)
        rgb = tf.cast(rgb / 255, dtype=tf.float32)
        rgb.set_shape((None, None, 3))
        initial_width = tf.shape(rgb)[-2]
        initial_height = tf.shape(rgb)[-3]
        w2 = (self.input_size[1] // 2)
        h2 = (self.input_size[0] // 2)
        w = (self.input_size[1])
        h = (self.input_size[0])

        if w > h:
            h = w
        else:
            w = h
        if (2 * w < initial_width) & (2 * h < initial_height):
            x = np.random.randint(w + 1, initial_width - w - 1)
            y = np.random.randint(h + 1, initial_height - h - 1)
        else:
            x = w
            y = h
        image_mob = rgb[y - h:y + h, x - w:x + w, :]
        image_mob = tf.image.resize(image_mob, [self.input_size[0], self.input_size[0]])
        image_desk = rgb[y - h:y + h, x - w:x + w, :]
        image_desk = tf.image.resize(image_desk, [self.input_size[0], self.input_size[0]])
        return image_desk, image_mob

    def returned_batch(self, batch_image_input, batch_image_output,desk_mob, label):
        return (batch_image_input, desk_mob), label


# generator image+preproc "Alina"
class CustomDataGenFace(Sequence):
    def __init__(self,file_mobile,file_desk, path_m,path_d ,
               batch_size = 32,
               input_size=(224, 224, 3),
               s = 100,
               W = 224,
               lab = 0,
               shuffle=True, fft_=False, rand_ind = False, n_train = 0):

        self.batch_size = batch_size # 
        self.input_size = input_size # 
        self.shuffle = shuffle
        if n_train == 0:
            self.mobile_len = len(file_mobile)  # 1
        else:
            self.mobile_len = n_train * batch_size
        self.s = s
        self.W = W
        self.lab = lab # РїРµСЂРµР№С‚Рё РІ Lab

        self.desk_len = len(file_desk) #
        self.file_mobile =file_mobile
        self.path_m =path_m 
        self.file_desk =file_desk
        self.path_d =path_d 
        self.fft = fft_
        self.rand_ind = rand_ind


    def on_epoch_end(self):
        pass
    
    def __getitem__(self, index):
        #print(index)
        # 
        batch_image_mob =[]
        batch_image_desk =[]

        if self.rand_ind:
            ind1 = tf.random.uniform(shape=[self.batch_size],minval=0,maxval=self.mobile_len-self.batch_size*2 ,dtype=tf.int32).numpy()
            ind2 = tf.random.uniform(shape=[self.batch_size],minval=0,maxval=self.desk_len-self.batch_size*2 ,dtype=tf.int32).numpy()
        else:
            ind1 =list(range(index,index+self.batch_size))
            ind2 =list(range(index,index+self.batch_size))

        for i in range(self.batch_size):
            print(ind1[i])

            try:
                image_mob = preprocess_val(os.path.join(self.path_m,self.file_mobile[ind1[i]]))
                #print('1 preproc:')
                image_desk = preprocess_val(os.path.join(self.path_d,self.file_desk[ind2[i]]))
                #print('2 preproc:')
                image_desk = image_desk
                image_mob = image_mob
                if self.fft:
                    image_desk = fft_img(image_desk)
                    image_mob = fft_img(image_mob)
                batch_image_mob.append(image_mob[:,:,:].numpy())
                batch_image_desk.append(image_desk[:,:,:].numpy())
            except:
                print('No File')
        return np.array(batch_image_mob), np.array(batch_image_desk)  
    def __len__(self):
        return self.mobile_len // self.batch_size

# генератор с возвратом картинки и метки на выходк
class CustomDataGenFace_dm_label_image_new(CustomDataGen_Base):

    def image_preproc(self,path_):
        # preprocess image
        image_bytes = tf.io.read_file(path_)
        rgb = tf.image.decode_image(image_bytes, channels=3)
        rgb = tf.cast(rgb / 255, dtype=tf.float32)
        rgb.set_shape((None, None, 3))
        initial_width = tf.shape(rgb)[-2]
        initial_height = tf.shape(rgb)[-3]
        w2 = (self.input_size[1] // 2)
        h2 = (self.input_size[0] // 2)
        w = (self.input_size[1])
        h = (self.input_size[0])

        if initial_height < initial_width:
            dy = initial_height // 2
            dx = initial_width // 2
            d = dy
        else:
            dy = initial_height // 2
            dx = initial_width // 2
            d = dx
        d = d - 10
        if (2 * d <= (initial_width-20)) & (2 * d <= (initial_height-20)):
            if np.random.randint(0,2)==0:
                x=d+10 + np.random.randint(-5,5)
                y = dy + np.random.randint(-5,5)
            else:
                x =initial_width-10 - d
                y = dy

        else:
            x = dx
            y = dy
            d = d // 2

        # central crop(448x448) & resize(224x224)
        image_mob = rgb[y - d:y +d, x - d:x + d, :]*2-1
        image_mob = tf.image.resize(image_mob, [self.input_size[0], self.input_size[0]])
        image_desk = rgb[y - d:y + d, x - d:x + d, :]*2-1
        image_desk = tf.image.resize(image_desk, [self.input_size[0], self.input_size[0]])

        return image_desk, image_mob

    def returned_batch(self, batch_image_input, batch_image_output,desk_mob, label):
        return (batch_image_input, desk_mob), label

