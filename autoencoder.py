
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers


import sys

sys.path.insert(0,"../autoenc_main")

from net_module import upsample,downsample
from readcsvdata import read_data
from init_confir import config_model, tostring_model_name

import auto_enc

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)



def classif(latent = 2048, BN = 0, alpha = 1.0, imagenet = 1, input_shape_im = [224,224,3] ,
            class_mod='../CycleGAN/full.0200-0.0933-0.9710.hdf5', custom_object = {"Swish": auto_enc.Swish}):
        '''
        latent = 2048, -  размер предпоследнего плотного слоя
        BN = 0, - включить ли нормализацию после слоя разности (0 не включать, 1 включить)
        alpha = 1.0, - коэффициент размера мобайлнет (если 0, то грузить сеть по class_mod ))
        imagenet = 1,- грузить имаджнет
        input_shape_im = [224,224,3] , - размер картинки на входе
        class_mod='../CycleGAN/full.0200-0.0933-0.9710.hdf5' - сеть для загрузки
        custom_object = {"Swish": auto_enc.Swish} - custom net
        model_class_aux -  output model
        '''

        # входы классификатора (от автоэнкодера и исходная картинка)
        x_auto = layers.Input(shape=input_shape_im) # autoencoder
        input_class_enc = layers.Input(shape=input_shape_im) # input image

        x= layers.Subtract()([x_auto, input_class_enc])  # anomaly

        # batchnormaly
        if BN:
            x = layers.BatchNormalization()(x)

        # classificator body
        if alpha == 0:
            # cusnom model
            base_classif = tf.keras.models.load_model(class_mod,custom_objects=custom_object)
            x_out_class = base_classif(x)

        else:
            # MobileNet
            if imagenet:
                base_classif = tf.keras.applications.MobileNetV2(input_shape=input_shape_im,include_top=False, weights='imagenet',alpha=alpha)
            else:

                base_classif = tf.keras.applications.MobileNetV2(input_shape=input_shape_im,include_top=False, weights=None,alpha=alpha)


            #base_classif.summary()
            x = base_classif(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.BatchNormalization()(x)
            # hiden output layers
            x = layers.Dense(latent,activation = 'relu')(x)
            x = layers.BatchNormalization()(x)
            # output layers
            x_out_class = layers.Dense(1,activation = 'sigmoid')(x)

        model_class_aux = tf.keras.Model([x_auto,input_class_enc],x_out_class)
        #model_class.summary()
        return model_class_aux




def classif_1(latent = 2048, BN = 0, alpha = 1.0, imagenet = 1, input_shape_im = [224,224,3] ,
            class_mod='../CycleGAN/full.0200-0.0933-0.9710.hdf5', custom_object = {"Swish": auto_enc.Swish}):
        '''
        latent = 2048, -  размер предпоследнего плотного слоя
        BN = 0, - включить ли нормализацию после слоя разности (0 не включать, 1 включить)
        alpha = 1.0, - коэффициент размера мобайлнет (если 0, то грузить сеть по class_mod ))
        imagenet = 1,- грузить имаджнет
        input_shape_im = [224,224,3] , - размер картинки на входе
        class_mod='../CycleGAN/full.0200-0.0933-0.9710.hdf5' - сеть для загрузки
        custom_object = {"Swish": auto_enc.Swish} - custom net
        model_class_aux -  output model
        '''

        # входы классификатора (от автоэнкодера и исходная картинка)
        #x_auto = layers.Input(shape=input_shape_im) # autoencoder
        input_class_enc = layers.Input(shape=input_shape_im) # input image
        x = input_class_enc
        #x= layers.Subtract()([x_auto, input_class_enc])  # anomaly

        # batchnormaly
        if BN:
            x = layers.BatchNormalization()(x)

        # classificator body
        if alpha == 0:
            # cusnom model
            base_classif = tf.keras.models.load_model(class_mod,custom_objects=custom_object)
            x_out_class = base_classif(x)

        else:
            # MobileNet
            if imagenet & (alpha>0) & (alpha<1.5):
                base_classif = tf.keras.applications.MobileNetV2(input_shape=input_shape_im,include_top=False, weights='imagenet',alpha=alpha)
            else:
                if alpha>0:
                    base_classif = tf.keras.applications.MobileNetV2(input_shape=input_shape_im,include_top=False, weights=None,alpha=alpha)
                else:
                    x_in = tf.keras.layers.Input(input_shape_im)
                    x_out,_ = encoder(x_in,nk=4, input_shape_im = input_shape_im, kernel_size=[3,3], kernel_num=[32,64,256,1024,2048])
                    base_classif = tf.keras.Model(x_in,x_out)

            #base_classif.summary()
            x = base_classif(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.BatchNormalization()(x)
            # hiden output layers
            x = layers.Dense(latent,activation = 'relu')(x)
            x = layers.BatchNormalization()(x)
            # output layers
            x_out_class = layers.Dense(1,activation = 'sigmoid')(x)

        model_class_aux = tf.keras.Model([input_class_enc],x_out_class)
        #model_class.summary()
        return model_class_aux





def classif_2(latent = 2048, BN = 0, alpha = 1.0, imagenet = 1, input_shape_im = [224,224,3] ):
        '''
        latent = 2048, -  размер предпоследнего плотного слоя
        BN = 0, - включить ли нормализацию после слоя разности (0 не включать, 1 включить)
        alpha = 1.0, - коэффициент размера мобайлнет (если 0, то грузить сеть по class_mod ))
        imagenet = 1,- грузить имаджнет
        input_shape_im = [224,224,3] , - размер картинки на входе
        class_mod='../CycleGAN/full.0200-0.0933-0.9710.hdf5' - сеть для загрузки
        custom_object = {"Swish": auto_enc.Swish} - custom net
        model_class_aux -  output model
        '''
        # входы классификатора (от автоэнкодера и исходная картинка)

        input_class_enc = layers.Input(shape=input_shape_im) # input image
        x = input_class_enc

        # batchnormaly
        if BN:
            x = layers.BatchNormalization()(x)

        # classificator body
        if imagenet & (alpha>0) & (alpha<1.5):
            base_classif = tf.keras.applications.MobileNetV2(input_shape=input_shape_im,include_top=False, weights='imagenet',alpha=alpha)
        else:
            base_classif = tf.keras.applications.MobileNetV2(input_shape=input_shape_im,include_top=False, weights=None,alpha=alpha)

        #base_classif.summary()
        x = base_classif(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        # hiden output layers
        x = layers.Dense(latent,activation = 'relu')(x)
        x = layers.BatchNormalization()(x)
        # output layers
        x_out_class = layers.Dense(1,activation = 'sigmoid')(x)

        model_class_aux = tf.keras.Model(input_class_enc,x_out_class)

        #model_class.summary()
        return model_class_aux



# base encoder
def encoder(x_in, nk=4, input_shape_im=[112, 112, 3],latent_dim=256,
            kernel_num = [32, 64, 128, 256, 512, 1024, 2048],
            kernel_size = [4, 3]):
    '''
    inputs:
        x_in, - dinput tensor
        nk=4, - number of level
        latent = 265, -  размер latent layers
        kernel_num = [32, 64, 128, 256, 512, 1024, 2048], - kernel size from level
        kernel_size = [4, 3] - kernel size of all layers
        input_shape_im = [224,224,3] , - размер картинки на входе
    outputs
         x_latent_enc-  output tensor
         List_scip - list sckips
    '''

    x = layers.Conv2D(kernel_num[0], (kernel_size[0], kernel_size[0]), strides=(1, 1),
                      activation='relu', padding="same", name='Conv2d_input1')(x_in)
    x = layers.Conv2D(kernel_num[0], (kernel_size[0], kernel_size[0]), strides=(1, 1),
                      activation='relu', padding="same", name='Conv2d_input2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    List_scip = [x]
    jk = 0
    for jk in range(nk - 1):
        x = downsample(x,
                       filters=kernel_num[jk],
                       activation=layers.LeakyReLU(0.2),
                       kernel_size=(kernel_size[0], kernel_size[0]),
                       strides=(2, 2))
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(kernel_num[jk], (kernel_size[0], kernel_size[0]),
                          strides=(1, 1),
                          activation='relu',
                          padding="same", kernel_regularizer='l2', name='down_Conv2d_down' + str(jk)[0])(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.DepthwiseConv2D((2, 2),
                                   strides=(1, 1),
                                   activation='relu',
                                   padding="same", kernel_regularizer='l2', name='down_DEPTHW2d' + str(jk)[0])(x)
        List_scip += [x]

    if nk>0:
        x = downsample(x,
                   filters=kernel_num[jk],
                   activation=layers.LeakyReLU(0.2),
                   kernel_size=(kernel_size[0], kernel_size[0]),
                   strides=(2, 2))
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    x_latent_enc = x

    List_scip += [x]
    return x_latent_enc, List_scip

#base decoder
def decoder(x_in, nk=4, input_shape_im=[224, 224, 3],latent_dim=256,
            kernel_num = [32, 64, 128, 256, 512, 1024, 2048],
            kernel_size = [4, 3]):
    '''
    inputs:
        x_in, - input tensor
        nk=4, - number of level
        latent = 265, -  размер latent layers
        kernel_num = [32, 64, 128, 256, 512, 1024, 2048], - kernel size from level
        kernel_size = [4, 3] - kernel size of all layers
        input_shape_im = [224,224,3] , - output размер картинки
    outputs
         x_latent_enc-  output tensor
         List_scip - list sckips
    '''



    List_contr = [x_in]
    x = upsample(x_in,
                 filters=kernel_num[nk],
                 activation=layers.LeakyReLU(0.2),
                 kernel_size=(kernel_size[1], kernel_size[1]),
                 strides=(2, 2))
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(kernel_num[nk], (2, 2),  # n4*(2**(nk-jk)), (2, 2),
                      strides=(1, 1),
                      activation='relu',
                      padding="same", kernel_regularizer='l2', name='transpose_Conv2d_hiden' + str(nk)[0])(x)
    List_contr += [x]
    for jk in range(nk):
        x = upsample(x,
                     filters=kernel_num[nk - jk],
                     activation=layers.LeakyReLU(0.2),
                     kernel_size=(kernel_size[1], kernel_size[1]),
                     strides=(2, 2))
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(kernel_num[nk - jk], (2, 2),  # n4*(2**(nk-jk)), (2, 2),
                          strides=(1, 1),
                          activation='relu',
                          padding="same", kernel_regularizer='l2', name='transpose_Conv2d_up' + str(jk)[0])(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.DepthwiseConv2D((3, 3),
                                   strides=(1, 1),
                                   activation='relu',
                                   padding="same", kernel_regularizer='l2',
                                   name='transpose_DEPTHW2d_up' + str(jk)[0])(x)
        List_contr += [x]
    x_out = layers.Conv2D(input_shape_im[2], (1, 1),
                          strides=(1, 1),
                          activation='tanh',
                          padding="same", kernel_regularizer='l1', name='auto_out')(x)
    return x_out, List_contr


# decoder + scip
def decoder_unet(x_in, List_scip, nk=4, input_shape_im=[224, 224, 3],latent_dim=256,
            kernel_num = [32, 64, 128, 256, 512, 1024, 2048],
            kernel_size = [4, 3], activation='tanh'):
    '''
    inputs:
        List_scip - list sckips
        x_in, - input tensor
        nk=4, - number of level
        latent = 265, -  размер latent layers
        kernel_num = [32, 64, 128, 256, 512, 1024, 2048], - kernel size from level
        kernel_size = [4, 3] - kernel size of all layers
        input_shape_im = [224,224,3] , - output размер картинки
        activation='tanh' - output activation
    outputs
         x_out-  output tensor
    '''


    x = layers.Concatenate()([x, List_scip[-1]])

    x = layers.Conv2D(kernel_num[jk + 1], (kernel_size[0], kernel_size[0]), strides=(1, 1),
                      activation='relu', padding="same", name='latent3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Latent

    if nk>0:
        x = upsample(x,
                 filters=kernel_num[jk],
                 activation=layers.LeakyReLU(0.2),
                 kernel_size=(kernel_size[1], kernel_size[1]),
                 strides=(2, 2))
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(kernel_num[jk], (2, 2),  # n4*(2**(nk-jk)), (2, 2),
                      strides=(1, 1),
                      activation='relu',
                      padding="same", kernel_regularizer='l2', name='transpose_Conv2d_hiden' + str(jk)[0])(x)

    x = layers.Concatenate()([x, List_scip[-2]])
    x = layers.Conv2D(kernel_num[jk], (kernel_size[0], kernel_size[0]), strides=(1, 1),
                      activation='relu', padding="same", name='l_conv1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    for jk in range(nk):
        x = upsample(x,
                     filters=kernel_num[nk - jk],
                     activation=layers.LeakyReLU(0.2),
                     kernel_size=(kernel_size[1], kernel_size[1]),
                     strides=(2, 2))
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(kernel_num[nk - jk], (2, 2),  # n4*(2**(nk-jk)), (2, 2),
                          strides=(1, 1),
                          activation='relu',
                          padding="same", kernel_regularizer='l2', name='transpose_Conv2d_up' + str(jk)[0])(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.DepthwiseConv2D((3, 3),
                                   strides=(1, 1),
                                   activation='relu',
                                   padding="same", kernel_regularizer='l2',
                                   name='transpose_DEPTHW2d_up' + str(jk)[0])(x)
    if (3 + jk) < len(List_scip):
        x = layers.Concatenate()([x, List_scip[-3 - jk]])
    x = layers.Conv2D(kernel_num[jk], (kernel_size[0], kernel_size[0]), strides=(1, 1),
                          activation='relu', padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x_out = layers.Conv2D(input_shape_im[2], (1, 1),
                          strides=(1, 1),
                          activation = activation,
                          padding="same", kernel_regularizer='l1', name='auto_out')(x)
    # print(x_out)
    return x_out

# superresolution unet
def unet2(nk=4, input_shape_im=[112, 112, 3],
          latent_dim=256,
          BN=0 ,alpha=1.0, imagenet = True,
          class_mod = '../CycleGAN/full.0200-0.0933-0.9710.hdf5', custom_object = {"Swish": auto_enc.Swish}):
    '''
    latent = 256, -  размер предпоследнего плотного слоя
    BN = 0, - включить ли нормализацию после слоя разности (0 не включать, 1 включить)
    alpha = 1.0, - коэффициент размера мобайлнет (если 0, то грузить сеть по class_mod ))
    imagenet = 1,- грузить имаджнет
    input_shape_im = [224,224,3] , - размер картинки на входе
    class_mod='../CycleGAN/full.0200-0.0933-0.9710.hdf5' - сеть для загрузки
    custom_object = {"Swish": auto_enc.Swish} - custom net
    model_class_aux -  output model
    '''

    kernel_num = [32, 64, 128, 256, 512, 1024, 2048]
    kernel_size = [4, 3]
    latent_dim = latent_dim
    input_shape_im = input_shape_im
    #
    x = layers.Input(shape=input_shape_im)
    input_enc = x
    x, List_scip = encoder(x,nk = nk, input_shape_im=input_shape_im,latent_dim=latent_dim, kernel_size=kernel_size,kernel_num=kernel_num)

    # Latent

    x = layers.Conv2D(kernel_num[jk + 1], (kernel_size[0], kernel_size[0]), strides=(1, 1),
                      activation='relu', padding="same", name='latent1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(kernel_num[jk + 1], (kernel_size[0], kernel_size[0]), strides=(1, 1),
                      activation='relu', padding="same", name='latent2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Latent end

    x_out,_ = decoder_unet(x,List_scip,nk=nk, latent_dim=latent_dim, kernel_num=kernel_num, kernel_size=kernel_size)

    model_auto = tf.keras.Model(input_enc, x_out)

    model_class = classif(latent=2048,BN=BN, alpha=alpha,imagenet=imagenet,class_mod=class_mod,custom_object=custom_object)


    loss_fn = keras.losses.MeanSquaredError()
    loss_fn_class = keras.losses.BinaryCrossentropy(from_logits=False)
    model_auto.compile(loss=loss_fn, metrics=[loss_fn])
    model_class.compile(loss=loss_fn_class, metrics=['accuracy'])

    x_in = model_auto(input_enc)
    x_out_ = model_class([x_in, input_enc])

    model = tf.keras.Model([input_enc], x_out_)

    return model_auto, model_class, model


# convolution autoencoder
def autoenc_c(nk  = 4,
            input_shape_im=[224, 224, 3],
            latent_dim=256,
            class_mod='../CycleGAN/full.0200-0.0933-0.9710.hdf5', BN = 1,alpha=1.0, imagenet = 1, custom_object = {"Swish": auto_enc.Swish}):
        """
         Convolution autoencoder

         nk  = 4, - size autoencoder factor
         input_shape_im=[224, 224, 3], - image shape
         latent_dim=256, - latent space dimention
         BN=0, - batch Norm (1) , not Batch Norm (0)
         alpha = 1.0, - MobileV2 -size
         imagenet = 1 ,- set imagenet weght

         """
        kernel_num = [32, 64, 128, 256, 512, 1024, 2048]
        kernel_size  = [4,3]
        latent_dim = latent_dim
        input_shape_im = input_shape_im
        #
        x = layers.Input(shape=input_shape_im)
        input_enc = x
        x,_ = encoder(x, nk=nk+1, input_shape_im=input_shape_im, latent_dim=latent_dim, kernel_size=kernel_size,
                               kernel_num=kernel_num)

        # Latent

        x = layers.Conv2D(kernel_num[nk ], (kernel_size[0], kernel_size[0]), strides=(1, 1),
                          activation='relu', padding="same", name='latent1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(kernel_num[nk ], (kernel_size[0], kernel_size[0]), strides=(1, 1),
                          activation='relu', padding="same", name='latent2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        # Latent end

        x_out,_ = decoder(x, nk=nk, latent_dim=latent_dim, kernel_num=kernel_num, kernel_size=kernel_size)

        model_auto = tf.keras.Model(input_enc, x_out)
        print(model_auto.summary())
        model_class = classif(latent=2048, BN=BN, alpha=alpha, imagenet=imagenet, class_mod=class_mod,
                              custom_object=custom_object)

        #model_class.summary()

        loss_fn = keras.losses.MeanSquaredError()
        loss_fn_class = keras.losses.BinaryCrossentropy(from_logits = False)

        model_auto.compile(loss = loss_fn, metrics=[loss_fn])
        model_class.compile(loss=loss_fn_class,metrics =['accuracy'])


        x_in = model_auto(input_enc)
        x_out_ = model_class(x_in)
        model = tf.keras.Model(input_enc,x_out_)
        #

        return model_auto,model_class,model


# model Autoencode+classifier - classical
def autoenc_(nk  = 4,
            input_shape_im=[224, 224, 3],
            latent_dim=256,
            class_mod='../CycleGAN/full.0200-0.0933-0.9710.hdf5',
            BN = 1, alpha=1.0, imagenet = 1, custom_object = {"Swish": auto_enc.Swish}):
        """
         Classic autoencoder (Dense latent)

         nk  = 4, - size autoencoder factor
         input_shape_im=[224, 224, 3], - image shape
         latent_dim=256, - latent space dimention
         BN=0, - batch Norm (1) , not Batch Norm (0)
         alpha = 1.0, - MobileV2 -size
         imagenet = 1 ,- set imagenet weght

         """

        kernel_num = [32, 64, 128, 256, 512, 1024, 2048]
        kernel_size  = [4,3]

        #
        x = layers.Input(shape=input_shape_im)
        input_enc = x
        x,_ = encoder(x,nk=4, input_shape_im=input_shape_im, latent_dim=latent_dim,kernel_size=kernel_size,kernel_num=kernel_num )

        #Latent
        x = layers.Flatten()(x)
        x = layers.Dense(latent_dim)(x)

        s_x = [0,input_shape_im[1]//(2**(nk+1)),input_shape_im[1]//(2**(nk+1)),3]
        print(s_x)
        x = layers.Dense(kernel_num[nk]* s_x[1] *s_x[2] )(x)

        x = layers.Reshape((s_x[1], s_x[2], kernel_num[nk]))(x)


        #Latent end

        x_out,_  = decoder(x,nk=4,input_shape_im=input_shape_im,kernel_num=kernel_num,kernel_size=kernel_size)
        model_auto = tf.keras.Model(input_enc,x_out)
        #model_auto.summary()

        #model_class = classif(latent=2048,BN = BN, input_shape_im=input_shape_im,alpha=alpha,class_mod=class_mod,imagenet=imagenet, custom_object=custom_object)
        #model_class.summary()

        loss_fn = keras.losses.MeanSquaredError()
        loss_fn_class = keras.losses.BinaryCrossentropy(from_logits = False)

        model_auto.compile(loss = loss_fn, metrics=[loss_fn])
        
        # TO DO - get classificator from function
        #model_class.compile(loss=loss_fn_class,metrics =['accuracy'])


        #x_in = model_auto(input_enc)

        #x_out_ = model_class([x_in,input_enc])
        #model = tf.keras.Model([input_enc],x_out_)

        return model_auto,model_auto,model_auto#model_class,model



# model Autoencode+classifier
## contrastiv out

def autoenc_contr(nk  = 4,
            input_shape_im=[224, 224, 3],
            latent_dim=256,
            class_mod='../CycleGAN/full.0200-0.0933-0.9710.hdf5', BN=0, alpha = 1.0, imagenet = 1 , custom_object = {"Swish": auto_enc.Swish}):
        """
         Contrastiv autoencoder

         nk  = 4, - size autoencoder factor
         input_shape_im=[224, 224, 3], - image shape
         latent_dim=256, - latent space dimention
         BN=0, - batch Norm (1) , not Batch Norm (0)
         alpha = 1.0, - MobileV2 -size
         imagenet = 1 ,- set imagenet weght

         """
        kernel_num = [32, 64, 128, 256, 512, 1024, 2048]
        kernel_size  = [4,3]
        latent_dim = latent_dim
        input_shape_im = input_shape_im
        #encoder
        x = layers.Input(shape=input_shape_im)
        input_enc = x

        x,_ = encoder(input_enc, nk=nk, kernel_num=kernel_num, input_shape_im=input_shape_im, kernel_size=kernel_size)
        #Latent
        x = layers.Flatten()(x)
        x = layers.Dense(latent_dim)(x)

        s_x = [0,input_shape_im[1]//(2**(nk+1)),input_shape_im[1]//(2**(nk+1)),3]

        x = layers.Dense(kernel_num[jk]* s_x[1] *s_x[2] )(x)

        x = layers.Reshape((s_x[1], s_x[2], kernel_num[jk]))(x)
        # Latent end

        #decoder
        x, List_contr1 = decoder(x,nk=nk, input_shape_im=input_shape_im, latent_dim=latent_dim, kernel_size=kernel_size,kernel_num=kernel_num)
        x_out = x

        #contrastiv out
        List_contr = []

        for x1 in List_contr1:
            x2 = tf.keras.layers.Lambda(lambda x1: tf.math.l2_normalize(x1, axis=1))(x1)
            List_contr += [tf.keras.layers.GlobalAveragePooling2D()(x2) ]

        model_class = classif(latent=2048,BN = BN, alpha=alpha, imagenet=imagenet,input_shape_im=input_shape_im,class_mod=class_mod,custom_object=custom_object)
        model_auto = tf.keras.Model(input_enc,x_out)
        #model_class.summary( )

        model_auto_class = tf.keras.Model(input_enc,[x_out]+List_contr)

        #model_class.summary()
        loss_fn = keras.losses.MeanSquaredError()
        loss_fn_class = keras.losses.BinaryCrossentropy(from_logits = False)

        model_auto.compile(loss = loss_fn, metrics=[loss_fn])
        model_class.compile(loss=loss_fn_class,metrics =['accuracy'])


        x_in = model_auto(input_enc)
        x_out_ = model_class([x_in,input_enc])
        model = tf.keras.Model(input_enc,x_out_)
        #model.summary()

        return model_auto, model_class, model_auto_class, model



# unet autoencoder
def unet(nk=4,
          input_shape_im=[224, 224, 3],
          latent_dim=256,
          class_mod='../CycleGAN/full.0200-0.0933-0.9710.hdf5', BN=0 ,alpha=1.0, custom_object = {"Swish": auto_enc.Swish}):
    """
     UNet autoencoder

     nk  = 4, - size autoencoder factor
     input_shape_im=[224, 224, 3], - image shape
     latent_dim=256, - latent space dimention
     BN=0, - batch Norm (1) , not Batch Norm (0)
     alpha = 1.0, - MobileV2 -size
     imagenet = 1 ,- set imagenet weght

     """
    kernel_num = [32, 64, 128, 256, 512, 1024, 2048]
    kernel_size = [4, 3]
    #
    x = layers.Input(shape=input_shape_im)
    input_enc = x

    x, List_scip = encoder(input_enc, nk=nk, kernel_num=kernel_num, input_shape_im=input_shape_im, kernel_size=kernel_size)

    x_latent_enc = x
    s_x = tf.shape(x_latent_enc)

    # Latent

    x = layers.Conv2D(kernel_num[jk + 1], (kernel_size[0], kernel_size[0]), strides=(1, 1),
                      activation='relu', padding="same", name='latent1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(kernel_num[jk + 1], (kernel_size[0], kernel_size[0]), strides=(1, 1),
                      activation='relu', padding="same", name='latent2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    #latent end

    x_out, _ = decoder_unet(x,List_scip,nk=nk, input_shape_im=input_shape_im, latent_dim=latent_dim,kernel_num=kernel_num,kernel_size=kernel_size)

    model_auto = tf.keras.Model(input_enc, x_out)

    model_auto.summary()


    model_class = classif(latent=2048,BN=BN, alpha=alpha, input_shape_im=input_shape_im, class_mod=class_mod, custom_object=custom_object)
    # model_class.summary()
    loss_fn = keras.losses.MeanSquaredError()
    loss_fn_class = keras.losses.BinaryCrossentropy(from_logits=False)

    model_auto.compile(loss=loss_fn, metrics=[loss_fn])
    model_class.compile(loss=loss_fn_class, metrics=['accuracy'])

    x_in = model_auto(input_enc)


    x_out_ = model_class([x_in, input_enc])
    model = tf.keras.Model([input_enc], x_out_)
    # print('MODEL:')
    # model.summary()

    return model_auto, model_class, model

## Autoencoder convolusionary + classif
def autoenc_conv(nk  = 4,
            input_shape_im=[224, 224, 3],
            latent_dim=256,
            BN = 1,
            imagenet = 1,
            alpha = 1.0,
            class_mod='../CycleGAN/full.0200-0.0933-0.9710.hdf5', custom_object = {"Swish": auto_enc.Swish}):
        """
         Convolution autoencoder

         nk  = 4, - size autoencoder factor
         input_shape_im=[224, 224, 3], - image shape
         latent_dim=256, - latent space dimention
         BN=0, - batch Norm (1) , not Batch Norm (0)
         alpha = 1.0, - MobileV2 -size
         imagenet = 1 ,- set imagenet weght

         """
        kernel_num = [32,64,128,256,512,1024,2048]
        kernel_size = [4,3]
        latent_dim = latent_dim
        input_shape_im = input_shape_im
        #
        x = layers.Input(shape=input_shape_im)
        input_enc = x
        x, _ = encoder(x, kernel_num=kernel_num,kernel_size=kernel_size,input_shape_im=input_shape_im,latent_dim=latent_dim)
        x_latent_enc = x
        s_x = tf.shape(x_latent_enc)
        #Latent

        x = keras.layers.Conv2D(latent_dim,3,padding="same",activation=layers.LeakyReLU(0.2))(x)

        x = keras.layers.Conv2D(latent_dim,1,padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = keras.layers.Conv2D(latent_dim,3,activation=layers.LeakyReLU(0.2),padding="same")(x)

        #Latent end
        x = decoder(x,nk=nk, input_shape_im=input_shape_im,latent_dim=latent_dim,kernel_size=kernel_size,kernel_num=kernel_num)

        model_auto = tf.keras.Model(input_enc,x_out)

        #model_auto.summary()

        model_class = classif(BN = 1, alpha=alpha,imagenet = imagenet,input_shape_im =input_shape_im)

        loss_fn = keras.losses.MeanSquaredError()
        loss_fn_class = keras.losses.BinaryCrossentropy(from_logits = False)

        model_auto.compile(loss = loss_fn, metrics=[loss_fn])
        model_class.compile(loss=loss_fn_class,metrics =['accuracy'])

        x_in = model_auto(input_enc)
        x_out_ = model_class([x_in, input_enc])
        model_ = tf.keras.Model([input_enc], x_out_)

        return model_auto,model_class, model_



# model sparce Autoencode+classifier
# https://github.com/sudharsan13296/Hands-On-Deep-Learning-Algorithms-with-Python/blob/master/10.%20Reconsturcting%20Inputs%20using%20Autoencoders/10.09%20Building%20the%20Sparse%20Autoencoder.ipynb
def autoenc_sparse(nk  = 4,
            input_shape_im=[224, 224, 3],
            latent_dim=256,
            BN = 1,
            imagenet = 1,
            alpha = 1.0,
            class_mod='../CycleGAN/full.0200-0.0933-0.9710.hdf5', custom_object = {"Swish": auto_enc.Swish}):
        """
        Sparse autoencoder

        nk  = 4, - size autoencoder factor
        input_shape_im=[224, 224, 3], - image shape
        latent_dim=256, - latent space dimention
        BN=0, - batch Norm (1) , not Batch Norm (0)
        alpha = 1.0, - MobileV2 -size
        imagenet = 1 ,- set imagenet weght

        """
        kernel_num = [32,64,128,256,512,1024,2048]
        kernel_size = [4,3]
        #
        x = layers.Input(shape=input_shape_im)
        input_enc = x
        x,_ = encoder(x,nk=nk, input_shape_im=input_shape_im,latent_dim=latent_dim,kernel_num=kernel_num,kernel_size=kernel_size)
        s_x = tf.shape(x)
        # x = keras.layers.Conv2D(latent_dim,1)(x)
        #Latent
        x = layers.Flatten()(x)
        lambda_l1=0.00001
        x = layers.Dense(latent_dim, activity_regularizer = keras.regularizers.L1L2(lambda_l1))(x)

        x = layers.Dense(kernel_num[nk]* s_x[1] *s_x[2] )(x)

        x = layers.Reshape((s_x[1], s_x[2], kernel_num[nk]))(x)

        #Latent
        x,_ = decoder(x, nk=nk, kernel_size=kernel_size,kernel_num=kernel_num, latent_dim=latent_dim)


        model_auto = tf.keras.Model(input_enc,x_out)

        model_auto.summary()

        #model_auto.summary()

        model_class = classif(BN = 1, alpha=alpha,imagenet = imagenet,input_shape_im =input_shape_im, latent=2048,class_mod=class_mod,custom_object=custom_object)

        loss_fn = keras.losses.MeanSquaredError()
        loss_fn_class = keras.losses.BinaryCrossentropy(from_logits = False)

        model_auto.compile(loss = loss_fn, metrics=[loss_fn])
        model_class.compile(loss=loss_fn_class,metrics =['accuracy'])

        x_in = model_auto(input_enc)
        x_out_ = model_class([x_in, input_enc])
        model_ = tf.keras.Model([input_enc], x_out_)

        return model_auto,model_class, model_


# Autoencoder (input-output) + classificator
def auto_class(model_auto,model_class, im_size=[224,224,3]):
    x_input = tf.keras.layers.Input(shape=(im_size[0],im_size[1],im_size[2]))
    x_auto = model_auto(x_input)
    x = tf.keras.layers.Subtract()([x_input,x_auto])
    x = tf.keras.layers.Lambda(lambda x1: x1*5)(x)
    x_out = model_class(x)

    return tf.keras.Model(x_input,x_out)




# Autoencoder classic - convolution latent
def auto_GD(kernel_num_enc = [64,128,192,64,128],kernel_size_enc=[3,3,3,4,1],stride_enc = [2,2,2,1,1],act_enc =['relu']*5,
        kernel_num_dec = [64,64,64,32],kernel_size_dec=[4,3,3,3],stride_dec = [1,2,2,2], act_dec = [tf.keras.layers.LeakyReLU(alpha=0.1)]*3+['tanh']):

    x_in = layers.Input(shape=[224,224,3])

    x = layers.Conv2D(kernel_num_enc[0],kernel_size_enc[0],strides = (stride_enc[0],stride_enc[0]),activation = act_enc[0])(x_in)
    for i in range(len(kernel_num_enc[1:])):
        x = layers.Conv2D(kernel_num_enc[i],(kernel_size[i],kernel_size[i]),activation=act_enc[i])(x)
        x = layers.BatchNormalization()(x)

    x = layers.Conv2D(kernel_num_dec[0],kernel_size_dec[0],strides = (stride_dec[0],stride_dec[0]),activation = act_dec[0])(x)
    x = layers.BatchNormalization()(x)

    for i in range(len(kernel_num_dec[1:-1])):
        x = layers.Conv2D(kernel_num_enc[i],(kernel_size[i],kernel_size[i]),activation=act_dec[i])(x)
        x = layers.BatchNormalization()(x)
    x = layers.Conv2D(kernel_num_enc[-1],(kernel_size[-1],kernel_size[-1]),activation=act_dec[-1])(x)

    return kf.keras.Model(x_in,x)

# model Autoencode convolution or dense +classifier
## contrastiv out

def autoenc_contr1(nk  = 4,
            input_shape_im=[224, 224, 3],
            latent_dim=256,
            BN=0, alpha = 1.0, imagenet = 1 , custom_object = {}, global_pool = True, dense=1, activation = 'tanh'):
        """
        Contrastiv autoencoder

        nk  = 4, - size autoencoder factor
        input_shape_im=[224, 224, 3], - image shape
        latent_dim=256, - latent space dimention
        BN=0, - batch Norm (1) , not Batch Norm (0)
        alpha = 1.0, - MobileV2 -size
        imagenet = 1 ,- set imagenet weght
        custom_object = {},
        global_pool = True, - ,Global Pooling
        dense=1,
        activation = 'tanh'

        """
        kernel_num = [32, 64, 128, 256, 512, 1024, 2048]
        kernel_size  = [4,3]
        latent_dim = latent_dim
        input_shape_im = input_shape_im
        #encoder
        x = layers.Input(shape=input_shape_im)
        input_enc = x
        # encoder
        x,List_scip = encoder(input_enc, nk=nk, kernel_num=kernel_num, input_shape_im=input_shape_im, kernel_size=kernel_size)
        #Latent
        if dense:
          x = layers.Flatten()(x)
          x = layers.Dense(latent_dim, activation=activation, kernel_regularizer='l1')(x)
          x_latent = x
          s_x = [0,input_shape_im[1]//(2**(nk+1)),input_shape_im[1]//(2**(nk+1)),3]

          x = layers.Dense(kernel_num[nk]* s_x[1] *s_x[2] )(x)

          x = layers.Reshape((s_x[1], s_x[2], kernel_num[nk]))(x)

        else:
          print('1')
          x = layers.Conv2D(latent_dim,(kernel_size[0],kernel_size[0]),strides=(2, 2),padding ='same')(x)
          x = layers.Conv2D(latent_dim,(kernel_size[1],kernel_size[1]),activation = activation,padding ='same',kernel_regularizer='l1')(x)
          x_latent = tf.keras.layers.GlobalAveragePooling2D()(x)
          x = layers.Conv2D(latent_dim,(kernel_size[1],kernel_size[1]),padding ='same')(x)
        # Latent end

        #decoder
        x, List_contr1 = decoder(x,nk=nk, input_shape_im=input_shape_im, latent_dim=latent_dim, kernel_size=kernel_size,kernel_num=kernel_num)
        x_out = x

        #contrastiv out
        List_contr = [x_latent]

        for x1 in List_contr1+[x_out]:
            if global_pool:
              x2 = tf.keras.layers.GlobalAveragePooling2D()(x1)
            else:
              x2 = tf.keras.layers.Conv2D((1,1),activation='sigmoid')(x1)
              x2 = tf.keras.layers.MaxPooling((8,8))(x2)
              x2 = tf.keras.layers.Flatten()(x2)
            x2 = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x2)
            print(x2.shape)
            List_contr += [x2 ]
        print(List_contr)
        contrast = tf.keras.layers.Concatenate()(List_contr)
        print(contrast)
        # autoencoder
        model_auto = tf.keras.Model(input_enc,x_out)
        # model for contrastiv vector
        model_auto_class = tf.keras.Model(input_enc,contrast)
        model_auto_class.summary()
        return model_auto, model_auto_class
