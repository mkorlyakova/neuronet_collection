# - 
"""
Autoencoder collection

"""

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import generator_auto
from net_module import ReflectionPadding2D, upsample,downsample,residual_block

# # Weights initializer for the layers.
#
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

# # Gamma initializer for instance normalization.
#
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
#
#
#
buffer_size = 128
#
batch_size = 1
input_img_size =[224,224,3]

# Decoder net
def get_decoder(
    latent_dim = [16,16,64],
    filters=64, kernel_initializer=kernel_init, name=None,
    output_img_size = [input_img_size[0],input_img_size[1],9]):
    '''
    latent_dim = [16,16,64], - размеры входного объекта
    filters=64, базовый размер фильтра
    kernel_initializer=kernel_init, - инициализация ядер
    name=None, - имя сети
    output_img_size = [input_img_size[0], input_img_size[1],9] - выходной размер
    '''

    img_input = layers.Input(shape = latent_dim , name=name + "_img_input_l")

    x =img_input

    numb_up = np.math.floor(np.math.log2(output_img_size[0] // latent_dim[0]))

    num_filters = filters*numb_up
    
    for num_upsample_block in range(numb_up):
        num_filters  //= 2
        if num_upsample_block < 5:
            x = upsample( x, filters=num_filters, activation=layers.LeakyReLU(0.2), kernel_size=(4, 4), strides=(2, 2),)
        else:
            x = upsample(x,filters=num_filters,activation=layers.LeakyReLU(0.2), kernel_size=(4, 4), strides=(1, 1),)

    x = layers.Conv2D( output_img_size[2], (4, 4), strides=(1, 1), 
                      activation = 'tanh',padding="same",
                      kernel_initializer=kernel_initializer)(x)

    model = keras.Model(inputs=img_input, outputs=x, name=name)
    return model
# Encoder
def get_encoder(
    latent_dim = [16,16,64],
    filters=64, kernel_initializer=kernel_init,  name=None,
    input_img_size = [input_img_size[0],input_img_size[1],3]):
    '''
    latent_dim = [16,16,64], - размеры входного объекта
    filters=64, базовый размер фильтра
    kernel_initializer=kernel_init, - инициализация ядер
    name=None, - имя сети
    output_img_size = [input_img_size[0], input_img_size[1],9] - выходной размер
    '''

    img_input = keras.layers.Input(shape=input_img_size, name=name + "_img_input_x")
    x =img_input
    numb_down = np.math.floor(np.math.log2(input_img_size[0] // latent_dim[0]))
    #print(input_img_size[0],latent_dim[0],numb_down)
    num_filters = filters
    #x = tf.keras.layers.Lambda(fft_y)(img_input[:,:,:,:3])
    for num_downsample_block in range(numb_down):
        print(num_downsample_block)
        num_filters *= 2
        if x.shape[1] > 5:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
            )

    x = layers.Conv2D(latent_dim[2], (4, 4), strides=(1, 1), padding="same",kernel_initializer=kernel_initializer)(x) # activation = 'sigmoid'

    model = keras.Model(inputs=img_input, outputs=x, name=name)
    return model


def get_latent_conv(
        latent_dim1=[16, 16, 64],
        latent_dim2=[16, 16, 64],
        latent_dim=2,
        filters=64,

        gamma_initializer=gamma_init,
        name=None,
        lat_layer = 2,
):
    img_input = layers.Input(shape=latent_dim1, name=name + "_img_input")
    #img_input = layers.Input(shape = (None,None,None) , name=name + "_img_input_l")

    x_in = tf.keras.layers.Conv2D(latent_dim1[2],(4,4), activation ='relu', padding = 'same' )(img_input)  # Downsampling
    x = x_in
    x = tf.keras.layers.Conv2D(latent_dim1[2] // 2, (4, 4), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    for i in range(lat_layer):
        x1 = tf.keras.layers.Conv2D(latent_dim1[2] // 2, (4, 4), activation ='relu', padding = 'same' )(x)
        x1 = tf.keras.layers.Conv2D(latent_dim1[2] // 2, (4, 4), activation ='relu', padding = 'same' )(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.Conv2D(latent_dim1[2] // 2, (4, 4), activation ='relu' , padding = 'same' )(x)
        x1 = tf.keras.layers.Conv2D(latent_dim1[2] // 2, (4, 4), activation ='relu', padding = 'same' )(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.Add()([x1,x])
        x = x1
    x2 = tf.keras.layers.Conv2D(latent_dim2[2],(4,4), padding = 'same' ,activation = 'relu')(x1)
    x2 = keras.layers.BatchNormalization()(x2)  # Upsampling


    model = keras.models.Model(img_input, x2, name=name)
    return model

def get_latent(
    latent_dim1 = [16,16,64],
    latent_dim2 = [16,16,64],
    latent_dim = 2,
    filters=64,
    
    gamma_initializer=gamma_init,
    name=None,
    ):
    img_input = layers.Input(shape=latent_dim1, name=name + "_img_input")
    x = tf.keras.layers.Flatten()(img_input)# Downsampling
    x1 = tf.keras.layers.Dense(filters)(x)
    x2 = tf.keras.layers.Dense(latent_dim2[0]*latent_dim2[1]*latent_dim2[2])(x1)# Upsampling
    x2 = tf.keras.layers.Reshape( latent_dim2)(x2) 

    model = keras.models.Model(img_input, x2, name=name)
    return model



    # Пример
    # Get the generators
    #gen_G = get_resnet_generator(name="generator_G")
    #gen_F = get_resnet_generator(name="generator_F")

    # Get the discriminators
    #disc_X = get_discriminator(name="discriminator_X")
    #disc_Y = get_discriminator(name="discriminator_Y")

# Автоэнкодер классической схемы + классификатор на выходе
class Autoencoder_v(keras.Model):


    def contastive_loss(self,labels,embs):
        # print(embs,labels)
        MARGIN = 10.0
        bs = embs.shape[0]
        embs1 = embs[:bs // 2, :]

        embs2 = embs[bs // 2:, :]
        labels1 = labels[:bs // 2]
        labels2 = labels[bs // 2:]
        # print(labels1,labels2)
        d2 = tf.reduce_sum(tf.square(embs1 - embs2), axis=1)
        d = tf.sqrt(d2)
        z = tf.cast(labels1 == labels2, tf.float32)

        return tf.reduce_mean(z * d2 + (1 - z) * tf.maximum(0, MARGIN - d) ** 2)
    def __init__(
        self,
        decoder_X,
        latent_X,
        encoder_X,
        input_shape_im = [224,224,3],
        lambda_=1.0,
        lambda_identity=1.5,
        latent_dim = 256,
        mod_ = 'mod',
        outlevel='block_5_expand_relu'
    ):
        '''
        decoder_X, - декодор
        latent_X, - латентный слой
        encoder_X, - энкодер
        input_shape_im = [224,224,3], размер входа
        lambda_=1.0, - коэффициент для потерь
        lambda_identity=1.5, - коэффициент потерь идентичности
        latent_dim = 256, - размер латентного слоя
        mod_ = 'mod', - использовать предобученную сеть с именем из mod 
        outlevel='block_5_expand_relu' - выходной слой
        '''
        super(Autoencoder_v, self).__init__()
        # базовый автоэнкодер
        self.dec_X = decoder_X
        self.outlevel = outlevel
        if mod_ == 'mod':
            # используем предобученную сеть
            base_classif = keras.applications.mobilenet_v2.MobileNetV2(
                input_shape=(input_shape_im[0], input_shape_im[1], input_shape_im[2]),
                alpha=1.0, include_top=False, weights='imagenet')
            x_base_end = base_classif.get_layer(self.outlevel).output  # 3-5613-14 6 - 28
            self.enc_X = keras.Model(base_classif.input, x_base_end)
            #x_base = down_model(input_enc)
        else:
            # новый энкодер
            self.enc_X = encoder_X
        #модуль латентного слоя
        self.latent_X = latent_X
        self.latent_dim  = latent_dim
        #
        self.enc_X.summary()
        self.dec_X.summary()
        self.latent_X.summary()



        # сборка автоэнкодера
        self.model_auto = tf.keras.models.Sequential([self.enc_X,self.latent_X,self.dec_X])
        # классификатор
        self.classificator = tf.keras.Sequential([self.enc_X])
        ###########################
        self.classificator.add(keras.layers.Conv2D(256,(4,4),activation='relu',padding='same'))
        self.classificator.add(keras.layers.BatchNormalization())
        self.classificator.add(keras.layers.Dropout(0.2))

        self.classificator.add(keras.layers.Conv2D(256,(4,4),activation='relu',padding='same'))
        self.classificator.add(keras.layers.BatchNormalization())
        self.classificator.add(keras.layers.Dropout(0.2))
        self.classificator.add(keras.layers.Conv2D(256,(4,4),strides=(2,2),activation='relu',padding='same'))
        self.classificator.add(keras.layers.BatchNormalization())
        self.classificator.add(keras.layers.Dropout(0.2))

        self.classificator.add(keras.layers.Conv2D(512,(4,4),activation='relu',padding='same'))
        self.classificator.add(keras.layers.BatchNormalization())
        self.classificator.add(keras.layers.Dropout(0.2))

        self.classificator.add(keras.layers.Conv2D(512,(4,4),activation='relu',padding='same'))
        self.classificator.add(keras.layers.BatchNormalization())
        self.classificator.add(keras.layers.Dropout(0.2))
        self.classificator.add(keras.layers.Conv2D(512,(4,4),strides=(2,2),activation='relu',padding='same'))
        self.classificator.add(keras.layers.BatchNormalization())
        self.classificator.add(keras.layers.Dropout(0.2))
        ############################

        self.classificator.add(keras.layers.Flatten())
        self.classificator.add(keras.layers.Dense(self.latent_dim * 4, activation='relu'))
        self.classificator.add(keras.layers.BatchNormalization())
        self.classificator.add(keras.layers.Dropout(0.2))

        self.classificator.add(keras.layers.Dense(self.latent_dim * 2, activation ='relu'))

        self.classificator.add(keras.layers.BatchNormalization())
        self.classificator.add(keras.layers.Dropout(0.2))

        self.classificator.add(keras.layers.Dense(self.latent_dim , activation='relu'))
        self.classificator.add(keras.layers.BatchNormalization())
        self.classificator.add(keras.layers.Dropout(0.2))

        self.classificator.add(keras.layers.Dense(1, activation='sigmoid'))

        x_latent = self.classificator.layers[-10].output ## [-7].output

        x_latent_input = self.classificator.input
        # полная модель 
        self.latent_model = keras.Model(x_latent_input,x_latent)

        self.lambda_= lambda_
        self.lambda_identity = lambda_identity
        self.input_shape_im = input_shape_im

        ## super(Autoencoder_v, self).build(input_shape = input_shape_im)
        self.model_auto.summary()
        self.classificator.summary()

    def compile(
        self,
        optimizer,
        loss_fn
    ):
        
   
        self.optimizer = optimizer
        
        self.loss_fn = loss_fn
        
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()
        self.loss_class = keras.losses.BinaryCrossentropy()
        self.metric_class = keras.metrics.BinaryCrossentropy()
        super(Autoencoder_v, self).compile(optimizer = self.optimizer,loss = loss_fn, metrics =[loss_fn] )
        self.model_auto.compile(run_eagerly=True,optimizer = self.optimizer,loss = self.loss_fn,metrics =[self.loss_fn])
        self.classificator.compile(run_eagerly=True, optimizer=self.optimizer, loss=self.loss_class, metrics=[self.metric_class])
    def call(self, x):
        if isinstance(x, tuple):
            x = x[0]
            #print('call:',x.shape)
            x = self.model_auto(x)
            return x
    # тренировка
    def train_step(self, batch_data):
      # x desktop and y mobile
      if isinstance(batch_data,tuple):
        real_x_real_y, y_label = batch_data
        real_x,real_y = real_x_real_y

        #print(": : ",real_x.shape,real_y.shape)
        if real_x.shape[0]>0:
            #self.enc_X.trainable = False
            with tf.GradientTape(persistent=True) as tape:
                # desk to mobile
                ind_ = np.where(y_label.numpy() == 0)[0]
                #print(ind_)
                if len(ind_) > 0:  # 0
                    #print(x.shape)
                    real_xp = tf.cast([real_x[ik, :, :, :] for ik in ind_], dtype=tf.float32)
                    real_yp = tf.cast([real_x[ik,:,:,:] for ik in ind_], dtype=tf.float32)
                    #print(': - : ',real_xp.shape,real_yp.shape)
                else:
                    real_xp = real_x
                    real_yp = real_x





                fake_y = self.model_auto(real_xp) #  , training=True)
    
                if real_xp.shape[0] >0 :

                    if real_x.shape[-1]>3:
                        fake_fft_y = generator_auto.fft_y(fake_y[:,:,:,:3])
                    else:
                        fake_fft_y = fake_y

                else:
                    print('not good')

                    fake_fft_y = fake_y


                if real_xp.shape[0] >0 :
                    # decoder loss
                    loss_G = self.loss_fn(real_yp[:,:,:,:3], fake_y[:,:,:,:3]) * self.lambda_
                    if real_xp.shape[-1]>3:
                        id_loss_G = ( self.identity_loss_fn(real_yp[:,:,:,3:6], fake_fft_y[:,:,:,3:6])* self.lambda_* self.lambda_identity )

                    # Total loss
                    id_loss_G = 0.0
                    total_loss = loss_G# + id_loss_G
                else:
                    total_loss  = self.loss_fn(real_yp[:,:,:,:3], fake_y[:,:,:,:3]) * self.lambda_
                    loss_G , id_loss_G = 0.0,0.0


            # Get the gradients
            grads_G = tape.gradient(total_loss, self.model_auto.trainable_variables)

            # Update the weights rs
            self.optimizer.apply_gradients(zip(grads_G, self.model_auto.trainable_variables))

            self.enc_X.trainable = False
            with tf.GradientTape() as tape:

                y_pred = self.latent_model(real_x)
                y_loss = self.contastive_loss(y_label,y_pred)
                #print('contrast: ',y_loss)
                y_grad = tape.gradient(y_loss, self.latent_model.trainable_variables)
            self.optimizer.apply_gradients(zip(y_grad,self.latent_model.trainable_variables))

            with tf.GradientTape() as tape:

                y_pred = self.classificator(real_x)
                y_loss = self.loss_class(y_label,y_pred)
                #print(y_loss)
                y_grad = tape.gradient(y_loss, self.classificator.trainable_variables)
            self.optimizer.apply_gradients(zip(y_grad,self.classificator.trainable_variables))
            self.enc_X.trainable=True
            return {
                "loss": total_loss,
                "los_G": loss_G,
                "identity_loss": id_loss_G,
                "loss_clf":y_loss,
            }
        else:
            return {
                "loss": 1000,
                "los_G": 100,
                "identity_loss": 100,
                "loss_clf":1,
            }
      else:
          return {
              "loss": 1000,
              "los_G": 100,
              "identity_loss": 100,
              "loss_clf":1,
          }

    # тест
    def test_step(self, batch_data):
        # x desctop - y 
        if isinstance(batch_data,tuple):


            real_x_real_y, y_label = batch_data
            real_x, real_y = real_x_real_y
            #print('test', real_x.shape, real_y.shape,y_label.shape)

        # 
            fake_y = self.model_auto(real_x, training=True)

            if fake_y.shape[0] > 0 :
                #print('good')
                if real_x.shape[-1]>3:
                    fake_fft_y = generator_auto.fft_y(fake_y)
                else:
                    fake_fft_y = fake_y

            else:
                #print('not good')
                fake_fft_y = fake_y
            #  output
            # decoder loss
            if fake_y.shape[0] >0 :
                    # decoder loss
                    loss_G = self.loss_fn(real_y[:,:,:,:3], fake_y[:,:,:,:3]) * self.lambda_

                    # identity loss

                    if real_y.shape[-1]>3:
                        id_loss_G = (self.identity_loss_fn(real_y[:,:,:,3:6], fake_fft_y[:,:,:,3:6]) * self.lambda_* self.lambda_identity  )
                    else:
                        id_loss_G =  0.0


                    # Total loss
                    total_loss = loss_G # + id_loss_G

            else:
                    total_loss  = self.loss_fn(real_x[:,:,:,:3], fake_y[:,:,:,:3]) * self.lambda_
                    loss_G , id_loss_G = 0.0,0.0

            y_pred = self.classificator(real_x)
            y_loss = self.loss_class(y_label, y_pred)


            return {
                "loss": total_loss,
                "los_G": loss_G,
                "identity_loss": id_loss_G,
                "loss_clf":y_loss,
            }
        else:
            return {
                "loss": 1000,
                "los_G": 200,
                "identity_loss": 200,
                "loss_clf":1,
            }
        

# Автоэнкодер с разностной схемой (выход- вход) + классификатор
class Autoencoder_rez(keras.Model):
    def __init__(
            self,
            decoder_X,
            latent_X,
            encoder_X,
            input_shape_im=[224, 224, 3],
            lambda_=100.0,
            lambda_identity=1.5,
            latent_dim=256,
            mod_='mod',
            outlevel='block_5_expand_relu'
    ):
        '''
        decoder_X, - декодор
        latent_X, - латентный слой
        encoder_X, - энкодер
        input_shape_im = [224,224,3], размер входа
        lambda_=1.0, - коэффициент для потерь
        lambda_identity=1.5, - коэффициент потерь идентичности
        latent_dim = 256, - размер латентного слоя
        mod_ = 'mod', - использовать предобученную сеть с именем из mod 
        outlevel='block_5_expand_relu' - выходной слой
        '''
        super(Autoencoder_rez, self).__init__()
        self.dec_X = decoder_X
        self.outlevel = outlevel
        input_enc = keras.layers.Input(shape=(input_shape_im[0], input_shape_im[1], input_shape_im[2]),
                                       name="auto_input")

        # генерация энкодера
        if mod_ == 'mod':
            base_classif = keras.applications.mobilenet_v2.MobileNetV2(
                input_shape=(input_shape_im[0], input_shape_im[1], input_shape_im[2]),
                alpha=1.0, include_top=False, weights='imagenet')
            x_base_end = base_classif.get_layer(self.outlevel).output  # 3-56, 13-14, 6 - 28
            self.enc_X = keras.Model(base_classif.input, x_base_end)
            base_classif.trainable = False
            # x_base = down_model(input_enc)
        else:
            self.enc_X = encoder_X
        
        # латентный слой
        self.latent_X = latent_X
        self.latent_dim = latent_dim
        #
        self.enc_X.summary()
        self.dec_X.summary()
        self.latent_X.summary()

        #
        self.model_auto = tf.keras.models.Sequential([self.enc_X, self.latent_X, self.dec_X])

        x_auto_output = self.model_auto(input_enc)
        x_rez = keras.layers.Subtract( )([input_enc,x_auto_output])
        x_c2 = keras.layers.BatchNormalization()(x_rez)

        num_filters = 64
        # блоки 
        for num_downsample_block in range(3):

            num_filters *= 2
            x_c2 = downsample(
                x_c2,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
                )
            x_c2 = layers.Conv2D( input_shape_im[2], (2, 2),
                               strides=(1, 1),
                               activation = 'tanh',
                               padding="same",name = 'unet_out'+str(num_filters)[:2],kernel_regularizer='l2')(x_c2)


        x_c2 = keras.layers.Flatten()(x_c2)
        x_c2 = keras.layers.Dense(self.latent_dim * 4, activation='relu',kernel_regularizer='l2')(x_c2)
        x_c2 = keras.layers.BatchNormalization()(x_c2)
        x_c2 = keras.layers.Dropout(0.2)(x_c2)

        x_c2 = keras.layers.Dense(self.latent_dim, activation='relu',kernel_regularizer='l2')(x_c2)
        x_c2 = keras.layers.BatchNormalization()(x_c2)
        x_c2 = keras.layers.Dropout(0.2)(x_c2)

        x_c2 = keras.layers.Dense(1, activation='sigmoid')(x_c2)
        
        self.classificator_rez = keras.Model(input_enc,x_c2)
        # classificator
        self.classificator = tf.keras.Sequential([self.enc_X])
        self.classificator.add(keras.layers.Flatten())
        self.classificator.add(keras.layers.Dense(self.latent_dim * 4, activation='relu'))
        self.classificator.add(keras.layers.BatchNormalization())
        self.classificator.add(keras.layers.Dropout(0.2))

        self.classificator.add(keras.layers.Dense(self.latent_dim * 2, activation='relu'))
        self.classificator.add(keras.layers.BatchNormalization())
        self.classificator.add(keras.layers.Dropout(0.2))

        self.classificator.add(keras.layers.Dense(self.latent_dim, activation='relu'))
        self.classificator.add(keras.layers.BatchNormalization())
        self.classificator.add(keras.layers.Dropout(0.2))

        self.classificator.add(keras.layers.Dense(1, activation='sigmoid'))

        self.lambda_ = lambda_
        self.lambda_identity = lambda_identity
        self.input_shape_im = input_shape_im

        ## super(Autoencoder_v, self).build(input_shape = input_shape_im)
        self.model_auto.summary()
        self.classificator.summary()
        

    def compile(
            self,
            optimizer,
            loss_fn
    ):

        self.optimizer = optimizer

        self.loss_fn = loss_fn

        self.identity_loss_fn = keras.losses.MeanAbsoluteError()
        self.loss_class = keras.losses.BinaryCrossentropy()
        self.metric_class = keras.metrics.BinaryCrossentropy()
        super(Autoencoder_rez, self).compile(optimizer=self.optimizer, loss=loss_fn, metrics=[loss_fn])
        self.model_auto.compile(run_eagerly=True, optimizer=self.optimizer, loss=self.loss_fn, metrics=[self.loss_fn])
        self.classificator.compile(run_eagerly=True, optimizer=self.optimizer, loss=self.loss_class,
                                   metrics=[self.metric_class])
        self.classificator_rez.compile(run_eagerly=True, optimizer=self.optimizer, loss=self.loss_class,
                                   metrics=[self.metric_class])
    def call(self, x):
        if isinstance(x, tuple):
            x = x[0]
            print('call:', x.shape)
            x = self.model_auto(x)
        return x
    
        

    def train_step(self, batch_data):
        # x desktop and y mobile
        if isinstance(batch_data, tuple):
            real_x_real_y, y_label = batch_data
            real_x, real_y = real_x_real_y

            if real_x.shape[0] > 0:
                self.enc_X.trainable = False
                with tf.GradientTape(persistent=True) as tape:
                    
                    # desk to mobile
                    ind_ = np.where(y_label.numpy() == 0)[0]

                    if len(ind_) > 0:  # 0
                        

                        real_xp = tf.cast([real_x[ik, :, :, :] for ik in ind_], dtype=tf.float32)
                        real_yp = tf.cast([real_x[ik,:,:,:] for ik in ind_], dtype=tf.float32)

                    else:
                        real_xp = real_x
                        real_yp = real_x

                    fake_y = self.model_auto(real_xp)  #
                    
                    if real_xp.shape[0] > 0:

                        if real_xp.shape[-1] > 3:
                            fake_fft_y = generator_auto.fft_y(fake_y[:, :, :, :3])
                        else:
                            fake_fft_y = fake_y
                    else:
                        print('not good')

                        fake_fft_y = fake_y

                    if real_xp.shape[0] > 0:
                        
                        # decoder loss
                        loss_G = self.loss_fn(real_xp[:, :, :, :3], fake_y[:, :, :, :3])

                        if real_xp.shape[-1] > 3:
                            id_loss_G = (self.identity_loss_fn(real_xp[:, :, :, 3:6], fake_fft_y[:, :, :,3:6]))
                                         
                        # Total loss
                        id_loss_G = 0.0
                                         
                        total_loss = loss_G  # + id_loss_G
                                         
                    else:
                        total_loss = self.loss_fn(real_xp[:, :, :, :3], fake_y[:, :, :, :3])
                        loss_G, id_loss_G = 0.0, 0.0

                # Get the gradients
                                         
                grads_G = tape.gradient(total_loss, self.model_auto.trainable_variables)
                                         

                # Update the weights rs
                self.optimizer.apply_gradients(zip(grads_G, self.model_auto.trainable_variables) )

                self.enc_X.trainable = False
                self.dec_X.tarinable = False
                self.latent_X.trainable = False

                with tf.GradientTape() as tape:

                    y_pred = self.classificator(real_x)
                    y_loss = self.loss_class(y_label, y_pred)
                    #print(y_loss)
                    y_grad = tape.gradient(y_loss/10, self.classificator.trainable_variables)
                self.optimizer.apply_gradients(zip(y_grad, self.classificator.trainable_variables))

                with tf.GradientTape() as tape:

                    y_pred2 = self.classificator_rez(real_x)
                    y_loss2 = self.loss_class(y_label, y_pred2)
                    #print(y_loss2)
                    y_grad2 = tape.gradient(y_loss2/10, self.classificator_rez.trainable_variables)
                self.optimizer.apply_gradients(zip(y_grad2, self.classificator_rez.trainable_variables))

                self.enc_X.trainable = True
                self.dec_X.tarinable = True
                self.latent_X.trainable = True

                return {
                    "loss": total_loss,
                    "los_G": loss_G,
                    "identity_loss": id_loss_G,
                    "loss_clf": y_loss,
                    "loss_cls_rez":y_loss2,
                }
            else:
                return {
                    "loss": 1000,
                    "los_G": 100,
                    "identity_loss": 100,
                    "loss_clf": 1,
                    "loss_cls_rez": 1,
                }
        else:
            return {
                "loss": 1000,
                "los_G": 100,
                "identity_loss": 100,
                "loss_clf": 1,
                "loss_cls_rez": 1,
            }
        
                                         

    def test_step(self, batch_data):
        # x desctop - y
        if isinstance(batch_data, tuple):

            real_x_real_y, y_label = batch_data
            real_x, real_y = real_x_real_y

            #
            fake_y = self.model_auto(real_x, training=True)

            if fake_y.shape[0] > 0:

                if real_x.shape[-1] > 3:
                    fake_fft_y = generator_auto.fft_y(fake_y)
                else:
                    fake_fft_y = fake_y

            else:
                #print('not good')
                fake_fft_y = fake_y
            #  output
            # decoder loss
            if fake_y.shape[0] > 0:
                # decoder loss
                loss_G = self.loss_fn(real_y[:, :, :, :3], fake_y[:, :, :, :3]) * self.lambda_

                # identity loss

                if real_y.shape[-1] > 3:
                    id_loss_G = (self.identity_loss_fn(real_y[:, :, :, 3:6],
                                                       fake_fft_y[:, :, :, 3:6]) * self.lambda_ * self.lambda_identity)
                else:
                    id_loss_G = 0.0

                loss_G, id_loss_G = 0.0, 0.0

            y_pred = self.classificator(real_x)
            y_loss = self.loss_class(y_label, y_pred)
            y_pred2 = self.classificator_rez(real_x)
            y_loss2 = self.loss_class(y_label, y_pred2)
            return {
                "loss": total_loss,
                "los_G": loss_G,
                "identity_loss": id_loss_G,
                "loss_clf": y_loss,
                "loss_cls_rez": y_loss2,
            }
        else:
            return {
                "loss": 1000,
                "los_G": 200,
                "identity_loss": 200,
                "loss_clf": 1,
                "loss_cls_rez": 1,
            }
                                         

# UNet autoencoder
class Autoencoder_unet(keras.Model):
    def __init__(
            self,
            encoder_X = None,
            input_shape_im=[224, 224, 3],
            lambda_=100.0,
            lambda_identity=1.5,
            latent_dim=256,
            mod_='mod',
            deep =  5,
            outlevel='block_16_expand_relu'
    ):
        '''
        encoder_X, - энкодер
        input_shape_im = [224,224,3], размер входа
        lambda_=1.0, - коэффициент для потерь
        lambda_identity=1.5, - коэффициент потерь идентичности
        latent_dim = 256, - размер латентного слоя
        mod_ = 'mod', - использовать предобученную сеть с именем из mod 
        outlevel='block_5_expand_relu' - выходной слой
        '''
        super(Autoencoder_unet, self).__init__()
        self.outlevel = outlevel

        self.latent_dim = latent_dim
        if mod_ == 'mod':
            base_classif = keras.applications.mobilenet_v2.MobileNetV2(
                input_shape=(input_shape_im[0], input_shape_im[1], input_shape_im[2]),
                alpha=1.0, include_top=False, weights='imagenet')
            #base_classif.summary()
            x_base_end = base_classif.get_layer(self.outlevel).output  # 3-56, 13-14, 6 - 28
            self.enc_X = keras.Model(base_classif.input, x_base_end)
            base_classif.trainable = False
            List_back = ['block_16_expand_relu',  # 7
                         'block_13_expand_relu',  # 14
                         'block_6_expand_relu',  # 28
                         'block_3_expand_relu',  # 56
                         'block_1_expand_relu']  # 112
            
        else:
            self.backborne = mod_
            # backborne = 'full.best.0600-0.0831-0.9774.hdf5', outlevel = 'out_swish'
            if mod_ != '':
                base_classif = tf.keras.models.load_model(self.backborne, custom_objects={"Swish": Swish})
                base_classif.summary()

                x_base_end = base_classif.get_layer(self.outlevel).output
                self.enc_x = keras.Model(base_classif.input, x_base_end)
                #self.enc_X = encoder_X
                List_back = ['block_16_expand',  # 7
                             'block_13_expand',  # 14
                             'block_6_expand_swish',  # 28
                             'block_3_expand_swish',  # 56
                             'block_1_expand_swish']  # 112

            else:
                self.enc_X = encoder_X
                List_back = outlevel  # 112
        self.enc_X.summary()
        print(List_back[len(List_back)-deep:])
        input_enc = self.enc_X.layers[0].input
        x_enc_end = self.enc_X.layers[-1].output#(input_enc)

        
        x  = keras.layers.Conv2D(self.latent_dim * 4,(3,3),activation = 'relu',kernel_regularizer ='l2' , padding = 'same')(x_enc_end )
        x = keras.layers.Conv2D(self.latent_dim * 4, (3, 3), activation='relu',kernel_regularizer ='l2' , padding = 'same')(x)
        x = keras.layers.Conv2D(x_enc_end.shape[3], (1, 1), activation='relu',kernel_regularizer ='l2' , padding = 'same')(x)
        x = keras.layers.Add(name='Add_latent_unet')([x, x_enc_end])
        x = keras.layers.BatchNormalization()(x)
        x_latent_enc = x
        for jk,x_block in enumerate(List_back[len(List_back)-deep:]):
            x_skip = self.enc_X.get_layer(x_block).output
            print(x_skip)
            x = keras.layers.Concatenate(axis=-1)([x,x_skip ])
            x.shape
            x = upsample(x,
                         filters= self.latent_dim * 4 //(2**jk),
                         activation=layers.LeakyReLU(0.2),
                         kernel_size=(4, 4),
                         strides=(2, 2))
            x = layers.Conv2D(latent_dim * 4 //(2**jk), (2, 2),
                                  strides=(1, 1),
                                  activation='relu',
                                  padding="same", kernel_regularizer ='l2' ,name = 'transpose_Conv2d'+str(jk)[0])(x)
            x = keras.layers.BatchNormalization()(x)
        x_r = x
        x = layers.Conv2D( latent_dim * 4, (4, 4),
                               strides=(1, 1),
                               activation = 'tanh',
                               padding="same",kernel_regularizer ='l2' ,name = 'unet_out2')(x)
        x_out = layers.Conv2D( latent_dim * 4 //(2**jk), (4, 4 ),
                               strides=(1, 1),
                               activation = 'tanh',
                               padding="same",kernel_regularizer ='l2' ,name = 'unet_out1')(x)
        x_out = keras.layers.BatchNormalization()(x_out)

        x_out = keras.layers.Add(name='Add_input-output_unet')([x_r, x_out])

        x_out = keras.layers.BatchNormalization()(x_out)
        x_out = layers.Conv2D( input_img_size[2], (4, 4 ),
                               strides=(1, 1),
                               activation = 'tanh',
                               padding="same",kernel_regularizer ='l2' ,name = 'unet_out3')(x_out)
        print(x_out)
        #self.enc_X.summary()
        print(input_enc)
        #
        self.model_auto = tf.keras.Model( input_enc ,x_out )
        self.model_auto.summary()

        x_rez = keras.layers.Subtract( name = 'substr_input-output_unet')([input_enc,x_out])
        x_c2 = keras.layers.BatchNormalization()(x_rez)

        num_filters = 32
        for num_downsample_block in range(deep):

            num_filters *= 2
            x_c2 = downsample(
                x_c2,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
                )
            x_c2 = layers.Conv2D(latent_dim * 4 //(2**jk), (2, 2),
                                  strides=(1, 1),
                                  activation='relu',
                                  padding="same", kernel_regularizer ='l2' ,name = 'transpose_Conv2d'+str(num_filters )[:2])(x_c2)

        x_c2 = keras.layers.Flatten()(x_c2)
        x_c2 = keras.layers.Dense(self.latent_dim *4 , activation='relu',kernel_regularizer ='l2' ,name = 'rez_class_dense_1')(x_c2)
        x_c2 = keras.layers.BatchNormalization()(x_c2)
        x_c2 = keras.layers.Dropout(0.2)(x_c2)

        x_c2 = keras.layers.Dense(1, activation='sigmoid',kernel_regularizer ='l2' ,name = 'rez_class_out')(x_c2)
        self.classificator_rez = keras.Model(input_enc,x_c2)

        x_c = layers.Conv2D(latent_dim * 4 // (2 ** jk), (2, 2),
                             strides=(1, 1),
                             activation='relu',
                             padding="same", kernel_regularizer='l2', name='transpose_Conv2d' + str(num_filters)[:2])(x_latent_enc)

        x_c = keras.layers.Flatten()(x_c)
        x_c = keras.layers.Dense(self.latent_dim * 4, activation='relu', kernel_regularizer='l2',
                                  name='rez_class_dense_1')(x_c)
        x_c = keras.layers.BatchNormalization()(x_c)
        x_c = keras.layers.Dropout(0.2)(x_c)

        x_c = keras.layers.Dense(self.latent_dim, activation='relu', kernel_regularizer='l2',
                                  name='rez_class_dense_3')(x_c)
        x_c = keras.layers.BatchNormalization()(x_c)
        x_c = keras.layers.Dropout(0.2)(x_c)

        x_c = keras.layers.Dense(1, activation='sigmoid', kernel_regularizer='l2', name='rez_class_out')(x_c)
        self.classificator = keras.Model(input_enc, x_c)


        self.lambda_ = lambda_
        self.lambda_identity = lambda_identity
        self.input_shape_im = input_shape_im

        ## super(Autoencoder_v, self).build(input_shape = input_shape_im)
        self.model_auto.summary()
        self.classificator.summary()
        self.classificator_rez.summary()

    def compile(
            self,
            optimizer,
            loss_fn
    ):

        self.optimizer = optimizer

        self.loss_fn = loss_fn

        self.identity_loss_fn = keras.losses.MeanAbsoluteError()
        self.loss_class = keras.losses.BinaryCrossentropy()
        self.metric_class = keras.metrics.BinaryCrossentropy()
        super(Autoencoder_unet, self).compile(optimizer=self.optimizer, loss=loss_fn, metrics=[loss_fn])
        self.model_auto.compile(run_eagerly=True, optimizer=self.optimizer, loss=self.loss_fn, metrics=[self.loss_fn])
        self.classificator.compile(run_eagerly=True, optimizer=self.optimizer, loss=self.loss_class,
                                   metrics=[self.metric_class])
        self.classificator_rez.compile(run_eagerly=True, optimizer=self.optimizer, loss=self.loss_class,
                                   metrics=[self.metric_class])
    def call(self, x):
        if isinstance(x, tuple):
            x = x[0]
            #print('call:', x.shape)
            x = self.model_auto(x)
            return x

    def train_step(self, batch_data):
        # x desktop and y mobile
        if isinstance(batch_data, tuple):
            real_x_real_y, y_label = batch_data
            real_x, real_y = real_x_real_y

            if real_x.shape[0] > 0:
                self.enc_X.trainable = False
                with tf.GradientTape(persistent=True) as tape:
                    # desk to mobile
                    ind_ = np.where(y_label.numpy() == 0)[0]
                    if len(ind_) > 0:  # 0
                        #print(x.shape)
                        real_xp = tf.cast([real_x[ik, :, :, :] for ik in ind_], dtype=tf.float32)
                        real_yp = tf.cast([real_x[ik,:,:,:] for ik in ind_], dtype=tf.float32)
                    else:
                        real_xp = real_x
                        real_yp = real_x

                    fake_y = self.model_auto(real_xp)  # , training=True)

                    if real_xp.shape[0] > 0:
                        if real_xp.shape[-1] > 3:
                            fake_fft_y = generator_auto.fft_y(fake_y[:, :, :, :3])
                        else:
                            fake_fft_y = fake_y
                    else:
                        fake_fft_y = fake_y

                    if real_xp.shape[0] > 0:
                        # decoder loss
                        loss_G = self.loss_fn(real_xp[:, :, :, :3], fake_y[:, :, :, :3])

                        if real_xp.shape[-1] > 3:
                            id_loss_G = (self.identity_loss_fn(real_xp[:, :, :, 3:6], fake_fft_y[:, :, :,3:6]))
                        # Total loss
                        id_loss_G = 0.0
                        total_loss = loss_G  # + id_loss_G
                    else:
                        total_loss = self.loss_fn(real_xp[:, :, :, :3], fake_y[:, :, :, :3])
                        loss_G, id_loss_G = 0.0, 0.0

                # Get the gradients
                grads_G = tape.gradient(total_loss, self.model_auto.trainable_variables)

                # Update the weights rs
                self.optimizer.apply_gradients(zip(grads_G, self.model_auto.trainable_variables) )

                self.model_auto.trainable = False

                with tf.GradientTape() as tape:

                    y_pred = self.classificator(real_x)
                    y_loss = self.loss_class(y_label, y_pred)
                    #print(y_loss)
                    y_grad = tape.gradient(y_loss, self.classificator.trainable_variables)
                self.optimizer.apply_gradients(zip(y_grad, self.classificator.trainable_variables))

                with tf.GradientTape() as tape:

                    y_pred2 = self.classificator_rez(real_x)
                    y_loss2 = self.loss_class(y_label, y_pred2)
                    #print(y_loss2)
                    y_grad2 = tape.gradient(y_loss2, self.classificator_rez.trainable_variables)
                self.optimizer.apply_gradients(zip(y_grad2, self.classificator_rez.trainable_variables))

                self.model_auto.trainable = True


                return {
                    "loss": total_loss,
                    "loss_clf": y_loss,
                    "loss_cls_rez":y_loss2,
                }
            else:
                return {
                    "loss": 1000,
                    "loss_clf": 1,
                    "loss_cls_rez": 1,
                }
        else:
            return {
                "loss": 1000,
                "loss_clf": 1,
                "loss_cls_rez": 1,
            }

    def test_step(self, batch_data):
        # x desctop - y

        if isinstance(batch_data, tuple):

            real_x_real_y, y_label = batch_data
            real_x, real_y = real_x_real_y
            #print('test', real_x.shape, real_y.shape, y_label.shape)

            #
            fake_y = self.model_auto(real_x, training=True)

            if fake_y.shape[0] > 0:
                if real_x.shape[-1] > 3:
                    fake_fft_y = generator_auto.fft_y(fake_y)
                else:
                    fake_fft_y = fake_y
            else:
                #print('not good')
                fake_fft_y = fake_y
            #  output
            # decoder loss
            if fake_y.shape[0] > 0:
                # decoder loss
                loss_G = self.loss_fn(real_x[:, :, :, :3], fake_y[:, :, :, :3]) * self.lambda_
                # identity loss
                if real_y.shape[-1] > 3:
                    id_loss_G = (self.identity_loss_fn(real_x[:, :, :, 3:6],
                                                       fake_fft_y[:, :, :, 3:6]) * self.lambda_ * self.lambda_identity)
                else:
                    id_loss_G = 0.0

                # Total loss
                total_loss = loss_G  # + id_loss_G
            else:
                total_loss = self.loss_fn(real_x[:, :, :, :3], fake_y[:, :, :, :3]) * self.lambda_
                loss_G, id_loss_G = 0.0, 0.0

            y_pred = self.classificator(real_x)
            y_loss = self.loss_class(y_label, y_pred)
            y_pred2 = self.classificator_rez(real_x)
            y_loss2 = self.loss_class(y_label, y_pred2)
            return {
                "loss": total_loss,
                "loss_clf": y_loss,
                "loss_cls_rez": y_loss2,
            }
        else:
            return {
                "loss": 1000,
                "loss_clf": 1,
                "loss_cls_rez": 1,
            }
                                         

# Автоэнкодер для сегментации спуфинг-не спуфинг
class Autoencoder_unet_seg(keras.Model):
    def __init__(
            self,
            encoder_X = None,
            input_shape_im=[224, 224, 3],
            lambda_=100.0,
            lambda_identity=1.5,
            latent_dim=256,
            mod_='mod',
            deep =  3,
            outlevel='block_6_expand_relu'
    ):
        
        '''
        encoder_X, - энкодер
        input_shape_im = [224,224,3], размер входа
        lambda_=100.0, - коэффициент для потерь
        lambda_identity=1.5, - коэффициент потерь идентичности
        latent_dim = 256, - размер латентного слоя
        mod_ = 'mod', - использовать предобученную сеть с именем из mod 
        deep =  3
        outlevel='block_5_expand_relu' - выходной слой
        '''                                 
        super(Autoencoder_unet_seg, self).__init__()

        self.outlevel = outlevel

        if mod_ == 'mod':
            base_classif = keras.applications.mobilenet_v2.MobileNetV2(
                input_shape=(input_shape_im[0], input_shape_im[1], input_shape_im[2]),
                alpha=1.0, include_top=False, weights='imagenet')
            x_base_end = base_classif.get_layer(self.outlevel).output  # 3-56, 13-14, 6 - 28
            self.enc_X = keras.Model(base_classif.input, x_base_end)
            base_classif.trainable = False
            List_back = ['block_16_expand_relu',  # 7
                         'block_13_expand_relu',  # 14
                         'block_6_expand_relu',  # 28
                         'block_3_expand_relu',  # 56
                         'block_1_expand_relu']  # 112
        else:
            self.backborne = mod_
            # backborne = 'full.best.0600-0.0831-0.9774.hdf5', outlevel = 'out_swish'
            if mod_ != '':
                base_classif = tf.keras.models.load_model(self.backborne, custom_objects={"Swish": Swish})
                base_classif.summary()
                x_base_end = base_classif.get_layer(self.outlevel).output
                self.enc_x = keras.Model(base_classif.input, x_base_end)
                List_back = ['block_16_expand',  # 7
                             'block_13_expand',  # 14
                             'block_6_expand_swish',  # 28
                             'block_3_expand_swish',  # 56
                             'block_1_expand_swish']  # 112

            else:
                self.enc_X = encoder_X
                List_back = outlevel  # 112
        self.enc_X.summary()

        self.latent_dim = latent_dim

        input_enc = self.enc_X.layers[0].input
        x_enc_end = self.enc_X.layers[-1].output#(input_enc)

        x  = keras.layers.Conv2D(self.latent_dim * 4,(3,3),activation = 'relu',kernel_regularizer ='l2' , padding = 'same')(x_enc_end )
        x = keras.layers.Conv2D(self.latent_dim * 4, (3, 3), activation='relu',kernel_regularizer ='l2' , padding = 'same')(x)
        x = keras.layers.Conv2D(x_enc_end.shape[3], (1, 1), activation='relu',kernel_regularizer ='l2' , padding = 'same')(x)
        x = keras.layers.Add(name='Add_latent_unet')([x, x_enc_end])
        x = keras.layers.BatchNormalization()(x)
        x_latent_enc = x
        for jk,x_block in enumerate(List_back[len(List_back)-deep:]):
            x_skip = self.enc_X.get_layer(x_block).output
            print(x_skip)
            x = keras.layers.Concatenate(axis=-1)([x,x_skip ])
            x.shape
            x = upsample(x,
                         filters= self.latent_dim * 4 //(2**jk),
                         activation=layers.LeakyReLU(0.2),
                         kernel_size=(3, 3),
                         strides=(2, 2))
            x = layers.Conv2D(latent_dim * 4 //(2**jk), (3, 3),
                                  strides=(1, 1),
                                  activation='relu',
                                  padding="same", kernel_regularizer ='l2' ,name = 'transpose_Conv2d'+str(jk)[0])(x)
            x = keras.layers.BatchNormalization()(x)
        x_r = x
        x = layers.Conv2D( latent_dim * 4, (3, 3),
                               strides=(1, 1),
                               activation = 'relu',
                               padding="same",kernel_regularizer ='l2' ,name = 'unet_out2')(x)
        x_out = layers.Conv2D( latent_dim * 4 //(2**jk), (3, 3 ),
                               strides=(1, 1),
                               activation = 'relu',
                               padding="same",kernel_regularizer ='l2' ,name = 'unet_out1')(x)
        x_out = keras.layers.BatchNormalization()(x_out)

        x_out = keras.layers.Add(name='Add_input-output_unet')([x_r, x_out])

        x_out = keras.layers.BatchNormalization()(x_out)
        x_out = layers.Conv2D( input_img_size[2], (3, 3 ),
                               strides=(1, 1),
                               activation = 'tanh',
                               padding="same",kernel_regularizer ='l2' ,name = 'unet_out1_1')(x_out)
        print(x_out)
        #self.enc_X.summary()
        print(input_enc)
        #
        self.model_auto = tf.keras.Model( input_enc ,x_out )
        self.model_auto.summary()

        x_rez = keras.layers.Add( name = 'Add_input-output_unet1')([input_enc,x_out])
        x_c2 = keras.layers.BatchNormalization()(x_rez)

        num_filters = 32

        for num_downsample_block in range(deep):

            num_filters *= 2
            x_c2 = downsample(
                x_c2,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(3, 3),
                strides=(2, 2),
                )
            x_c2 = layers.Conv2D(latent_dim * 4 //(2**jk), (3, 3),
                                  strides=(1, 1),
                                  activation='relu',
                                  padding="same", kernel_regularizer ='l2' ,name = 'transpose_Conv2d'+str(num_filters )[:2])(x_c2)

        x_c2 = keras.layers.Flatten()(x_c2)
        x_c2 = keras.layers.Dense(self.latent_dim *4 , activation='relu',kernel_regularizer ='l2' ,name = 'rez_class_dense_1')(x_c2)
        x_c2 = keras.layers.BatchNormalization()(x_c2)
        x_c2 = keras.layers.Dropout(0.2)(x_c2)

        x_c2 = keras.layers.Dense(1, activation='sigmoid',kernel_regularizer ='l2' ,name = 'rez_class_out')(x_c2)
        self.classificator_rez = keras.Model(input_enc,x_c2)

        x_c = layers.Conv2D(latent_dim * 4 // (2 ** jk), (2, 2),
                             strides=(1, 1),
                             activation='relu',
                             padding="same", kernel_regularizer='l2', name='transpose_Conv2d' + str(num_filters)[:2])(x_latent_enc)

        x_c = keras.layers.Flatten()(x_c)
        x_c = keras.layers.Dense(self.latent_dim * 4, activation='relu', kernel_regularizer='l2',
                                  name='rez_class_dense_1')(x_c)
        x_c = keras.layers.BatchNormalization()(x_c)
        x_c = keras.layers.Dropout(0.2)(x_c)

        x_c = keras.layers.Dense(self.latent_dim, activation='relu', kernel_regularizer='l2',
                                  name='rez_class_dense_3')(x_c)
        x_c = keras.layers.BatchNormalization()(x_c)
        x_c = keras.layers.Dropout(0.2)(x_c)

        x_c = keras.layers.Dense(1, activation='sigmoid', kernel_regularizer='l2', name='rez_class_out')(x_c)
        self.classificator = keras.Model(input_enc, x_c)

        self.lambda_ = lambda_
        self.lambda_identity = lambda_identity
        self.input_shape_im = input_shape_im

        ## super(Autoencoder_v, self).build(input_shape = input_shape_im)
        self.model_auto.summary()
        self.classificator.summary()
        self.classificator_rez.summary()
                                         

    def compile(
            self,
            optimizer,
            loss_fn
    ):

        self.optimizer = optimizer

        self.loss_fn = loss_fn

        self.identity_loss_fn = keras.losses.MeanAbsoluteError()
        self.loss_class = keras.losses.BinaryCrossentropy(from_logits=False)
        self.metric_class = keras.metrics.BinaryCrossentropy(from_logits=False)
        super(Autoencoder_unet_seg, self).compile(optimizer=self.optimizer, loss=loss_fn, metrics=[loss_fn])
        self.model_auto.compile(run_eagerly=True, optimizer=self.optimizer, loss=self.loss_fn, metrics=[self.loss_fn])
        self.classificator_rez.compile(run_eagerly=True, optimizer=self.optimizer, loss=self.loss_class,
                                   metrics=[self.metric_class])
    def call(self, x):
        if isinstance(x, tuple):
            x = x[0]
            #print('call:', x.shape)
            x = self.model_auto(x)
            return x

    def train_step(self, batch_data):
        # x desktop and y mobile
        if isinstance(batch_data, tuple):
            real_x_real_y, y_label = batch_data
            real_x, real_y = real_x_real_y
            y = y_label * 2 - 1
            y_s = tf.cast([tf.ones_like(real_x[ik, :, :, :])*y[ik] for ik in range(real_x.shape[0])], dtype=tf.float32)
            # print(": : ",real_x.shape,real_y.shape)
            if real_x.shape[0] > 0:
                
                with tf.GradientTape(persistent=True) as tape:

                    fake_y = self.model_auto(real_x)  # , training=True)
                    if real_x.shape[0] > 0:
                        # decoder loss
                        loss_G = 0
                        loss_G = tf.reduce_sum(tf.math.abs(fake_y[:, :, :, :3]-y_s)**2)
                        #print('loss G:',loss_G)
                        #print('real_y',real_y.shape)
                        # identity loss
                        total_loss1 = loss_G  # + id_loss_G
                    else:
                        total_loss1 = self.loss_fn(tf.math.abs(fake_y[:, :, :, :3]) ,real_y[:,:,:,:3])
                        loss_G, id_loss_G = 0.0, 0.0

                # Get the gradients
                #print('loss:',total_loss)
                grads_G1 = tape.gradient(total_loss1, self.model_auto.trainable_variables)
                self.optimizer.apply_gradients(zip(grads_G1, self.model_auto.trainable_variables) )


                self.model_auto.trainable = False
                with tf.GradientTape() as tape:
                    
                    y_pred2 = self.classificator_rez(real_x)
                    y_loss2 = self.loss_class(y_label, y_pred2)
                    #print(y_pred2.numpy().sum())
                    y_grad2 = tape.gradient(y_loss2, self.classificator_rez.trainable_variables)
                self.optimizer.apply_gradients(zip(y_grad2, self.classificator_rez.trainable_variables))

                self.model_auto.trainable = True

                return {
                    "loss": total_loss1,
                    "los_G": loss_G,
                    "loss_cls_rez":y_loss2,
                }
            else:
                return {
                    "loss": 1000,
                    "los_G": 100,
                    "loss_cls_rez": 1,
                }
        else:
            return {
                "loss": 1000,
                "los_G": 100,
                "loss_cls_rez": 1,
            }

    def test_step(self, batch_data):
        # x desctop - y

        if isinstance(batch_data, tuple):

            real_x_real_y, y_label = batch_data
            real_x, real_y = real_x_real_y

            y = y_label * 2 - 1
            y_s = tf.cast([tf.ones_like(real_x[ik, :, :, :])*y[ik] for ik in range(real_x.shape[0])], dtype=tf.float32)

            #
            fake_y = self.model_auto(real_x, training=True)

            # decoder loss
            if fake_y.shape[0] > 0:
                # decoder loss
                loss_G = tf.reduce_sum(tf.math.abs(fake_y[:, :, :, :3]-y_s)**2)

                total_loss = loss_G

            else:
                total_loss = self.loss_fn(real_x[:, :, :, :3], fake_y[:, :, :, :3]) 
                loss_G, id_loss_G = 0.0, 0.0

            y_pred2 = self.classificator_rez(real_x)
            y_loss2 = self.loss_class(y_label, y_pred2)
            return {
                "loss": total_loss,
                "los_G": loss_G,
                "loss_cls_rez": y_loss2,
            }
        else:
            return {
                "loss": 1000,
                "los_G": 200,
                "loss_cls_rez": 1,
            }


# Вариационный автоэнкодер для разнострой схема (вход-выход)+классификато
class ClassRezVAE(tf.keras.Model):
    def model_parametr(self, model_name):
        with open(model_name + '.txt', 'w') as f:
            f.write(str(self.latent_dim) + '\n')
            f.write(self.backborne + '\n')
            f.write(self.outlevel + '\n')
            f.write(str(self.input_shape_im[0]) + '\n')
            f.write(str(self.input_shape_im[1]) + '\n')
            f.write(str(self.input_shape_im[2]) + '\n')
            f.close()

    def vae_reconstruction_loss(self, y_true, y_predict):
        reconstruction_loss_factor = 1.0
        reconstruction_loss = tf.math.reduce_mean(tf.math.square(y_true - y_predict), axis=[1, 2, 3])
        return tf.cast(reconstruction_loss_factor * reconstruction_loss / 224 / 224 / 3, dtype=tf.float32)

    def vae_kl_loss(self, encoder_mu, encoder_log_variance):
        dd = (encoder_mu) ** 2 - tf.exp(encoder_log_variance)
        # print('dd:',dd.shape)
        dd = 1.0 + encoder_log_variance - (encoder_mu) ** 2 - tf.exp(encoder_log_variance)
        # print('dd:',dd.shape,np.sum(tf.math.is_nan(dd)),np.sum(dd.numpy()),np.max(dd.numpy()),np.min(dd.numpy()))
        dd = np.sum(dd.numpy(), axis=1)
        # print('dd:',dd.shape)
        kl_loss = -0.5 * np.sum(
            1.0 + encoder_log_variance.numpy() - (encoder_mu.numpy()) ** 2 - np.exp(encoder_log_variance.numpy()),
            axis=1)
        # print(kl_loss)
        return tf.cast(kl_loss, dtype=tf.float32)

    def vae_kl_loss_metric(self, y_true, y_predict, encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * tf.math.reduce_sum(
            1.0 + encoder_log_variance - tf.math.square(encoder_mu) - tf.math.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_loss(self, y_true, y_predict):
        reconstruction_loss = self.vae_reconstruction_loss(y_true, y_predict)
        kl_loss = self.vae_kl_loss(y_true, y_predict)

        loss = reconstruction_loss + kl_loss
        return tf.cast(loss, dtype=tf.float32)

    def sampling(self, mu_log_variance):
        mu, log_variance = mu_log_variance
        epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
        random_sample = mu + tf.keras.backend.exp(log_variance / 2) * epsilon
        return random_sample

    def __init__(self, latent_dim, encoder_X, decoder_X, input_shape_im=[224, 224, 3], n_latent=8, n_dec=32,
                 model_name='', outlevel='block_6_expand_relu', backborne='mob'):
        
        '''
        encoder_X, - энкодер
        input_shape_im = [224,224,3], размер входа
        lambda_=100.0, - коэффициент для потерь
        lambda_identity=1.5, - коэффициент потерь идентичности
        latent_dim = 256, - размер латентного слоя
        mod_ = 'mod', - использовать предобученную сеть с именем из mod 
        deep =  3
        outlevel='block_5_expand_relu' - выходной слой претренированной сети
        '''           
        
        print('init')
        
        super(ClassRezVAE, self).__init__()
        self.optimizer = keras.optimizers.Adam(1e-4)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
        self.input_shape_im = input_shape_im
        
        if model_name != '':
            with open(model_name + '.txt', 'r') as f:
                self.latent_dim = int(f.readline())
                self.backborne = f.readline().replace('\n', '')
                self.outlevel = f.readline().replace('\n', '')
                i1 = int(f.readline())
                i2 = int(f.readline())
                i3 = int(f.readline())
                self.input_shape_im = [i1, i2, i3]
                f.close()
        else:
            self.latent_dim = latent_dim
            self.backborne = backborne
            self.outlevel = outlevel
            self.input_shape_im = input_shape_im

        input_enc = keras.layers.Input(shape=(input_shape_im[0], input_shape_im[1], input_shape_im[2]),
                                       name="VAE_input")

        if self.backborne == 'mob':
            base_classif = keras.applications.mobilenet_v2.MobileNetV2(
                input_shape=(input_shape_im[0], input_shape_im[1], input_shape_im[2]),
                alpha=1.0, include_top=False, weights='imagenet')
            x_base_end = base_classif.get_layer(self.outlevel).output  # 3-5613-14 6 - 28
        else:
            base_classif = tf.keras.models.load_model(self.backborne, custom_objects={"Swish": Swish})
            x_base_end = base_classif.get_layer(self.outlevel).output  # 1920

        down_model = keras.Model(base_classif.input, x_base_end)
        x_base = down_model(input_enc)

        x_f = keras.layers.Flatten()(x_base)
        print(x_f.shape, x_base.shape)

        x_hiden_m = keras.layers.Dense(latent_dim)(x_f)
        x_hiden_s = keras.layers.Dense(latent_dim)(x_f)
        x_class = keras.layers.Dense(latent_dim, activation=tf.nn.leaky_relu)(x_f)

        encoder_output = keras.layers.Lambda(self.sampling, name="encoder_output")([x_hiden_m, x_hiden_s])
        print(encoder_output.shape)

        decoder_input = keras.layers.Input(shape=(latent_dim,))
        dec_2 = keras.layers.Dense(units=n_latent * n_latent * n_dec, activation=tf.nn.relu)(decoder_input)

        dec_2 = keras.layers.Reshape(target_shape=(n_latent, n_latent, n_dec))(dec_2)
        dec_1 = decoder_X(dec_2)

        self.dec_X = keras.Model(decoder_input, dec_1)

        decoder_output = self.dec_X(encoder_output)
        decoder_class = self.dec_X(x_class)

        print('dec_1:',decoder_output.shape,'input_enc:',input_enc.shape)

        x_rez_in = keras.layers.Subtract()( [input_enc,decoder_class])# разность вход-выход
        print('x_rez_dec',x_rez_in.shape)

        x_rez = encoder_X(x_rez_in)
        print(x_rez.shape)
        x_rez = keras.layers.Flatten()(x_rez)
        x_rez = keras.layers.Dense(latent_dim, activation=tf.nn.leaky_relu)(x_rez)
        x_rez = keras.layers.Dropout(0.2)(x_rez)
        x_rez = keras.layers.Dense(latent_dim, activation=tf.nn.leaky_relu)(x_rez)



        x_class_1 = keras.layers.Dense(latent_dim, activation=tf.nn.leaky_relu)(x_class)

        print('rez_1:', x_rez.shape, 'class:', x_class_1.shape)
        x_class_rez = keras.layers.concatenate([x_class_1,x_rez])
        x_class_rez = keras.layers.BatchNormalization()(x_class_rez)
        x_class_rez = keras.layers.Dense(latent_dim, activation=tf.nn.leaky_relu)(x_class_rez)
        x_class_rez = keras.layers.BatchNormalization()(x_class_rez)
        print(x_class_rez.shape)
        output_class = keras.layers.Dense(1, activation='sigmoid')(x_class_rez)
        print(dec_1.shape)


        self.enc_X = keras.Model(input_enc, [x_hiden_m, x_hiden_s, encoder_output])
        self.classificator = keras.Model(input_enc, output_class)
        self.model_ = keras.Model(input_enc,decoder_class)

        print(decoder_output.shape)
        self.classificator.summary()

    def call(self, data):
        # super(ClassRezCVAE, self).call()
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data
        return self.model_(x)

    def compile(self, optimizer=tf.keras.optimizers.Adam()):
        super(ClassRezVAE, self).compile()

        self.model_.optimizer = optimizer
        self.enc_X.optimizer = optimizer
        self.dec_X.optimizer = optimizer
        self.classificator.optimizer = optimizer
        self.classificator.compile(optimizer = optimizer,loss = keras.losses.BinaryCrossentropy(from_logits = False),metrics =['accuracy'])

    def train_step(self, data):
        try:
            loss_vae = 1000
            loss_classif = 1
            x, y, label = data
            # y,label = target
            if label.shape[0] is not None:
                # print('label:',label)
                print(data[0].shape)
                try:
                    self.classificator.trainable = False
                    self.enc_X.trainable = True
                    self.dec_X.trainable = True
                    with tf.GradientTape() as tape:
                        # print('label: ',label)
                        ind_ = np.where(label.numpy() == 0)[0]
                        print(ind_)
                        if len(ind_) > 0:  # 0
                            print(x.shape)
                            xp = tf.cast([x[ik, :, :, :] for ik in ind_], dtype=tf.float32)
                            x_m, x_s, x_x = self.enc_X(xp)
                            #print(x_x.shape)
                            y_pred = self.dec_X(x_x)
                            #print('y-pred :', y_pred.shape)
                            loss_reconstr = self.vae_reconstruction_loss(xp, y_pred)
                        else:
                            x_m, x_s, x_x = self.enc_X(x)
                            y_pred = self.dec_X(x_x)
                            # print('y_pred:',y_pred.shape)
                            loss_reconstr = self.vae_reconstruction_loss(x, y_pred)
                            # print(x.shape,loss_reconstr.shape)
                        # print(x_m.shape,x_s.shape)
                        loss_kl = self.vae_kl_loss(encoder_mu=x_m, encoder_log_variance=x_s)
                        # print(loss_kl.shape)
                        loss_vae = tf.reduce_mean(loss_kl + loss_reconstr)
                        # print(loss_vae.shape)
                        grad = tape.gradient(loss_vae, self.model_.trainable_variables)
                    self.optimizer.apply_gradients(zip(grad, self.model_.trainable_variables))

                except:
                    print('error')

                try:
                    self.classificator.trainable = True
                    self.enc_X.trainable = False
                    self.dec_X.trainable = False
                    #self.model_.trainable = False
                    with tf.GradientTape() as tape:
                        label_pred = self.classificator(x)
                        loss_classif = self.loss_fn(label, label_pred)
                        grad = tape.gradient(loss_classif, self.classificator.trainable_variables)
                    self.optimizer.apply_gradients(zip(grad, self.classificator.trainable_variables))
                    self.model_.trainable = True
                    self.enc_X.trainable = True
                    self.dec_X.trainable = True
                except:
                    print('error1')
                    loss_vae = 1000
                    loss_classif = 1
            else:
                loss_vae = 1000
                loss_classif = 1
        except:
            print('error2')
        return {"loss": loss_vae, "loss_clf": loss_classif}

    def test_step(self, data):
        # super(ClassCVAE, self).test_step()
        loss_vae = 1000
        loss_classif = 1
        try:

            x, y, label = data
            
            if 1:  # x.shape[0] is not None:
                try:
                    x_m, x_s, x_x = self.enc_X(x)
                    y_pred = self.dec_X(x_x)
                    loss_reconstr = self.vae_reconstruction_loss(x, y_pred)
                    # print('loss_reconstr',loss_reconstr.shape)
                    loss_kl = self.vae_kl_loss(encoder_mu=x_m, encoder_log_variance=x_s)
                    # print('loss_kl:',loss_kl.shape)
                    loss_vae = tf.cast(np.sum(loss_kl.numpy() + loss_reconstr.numpy()), dtype=tf.float32)
                    # print('loss_vae:',loss_vae.shape)
                    label_pred = self.classificator(x)
                    # print('test cl:',y_pred.shape)

                    loss_classif = self.loss_fn(label, label_pred)
                except:
                    print('error')
                    loss_vae = 1000
                    loss_classif = 1
            else:
                loss_vae = 1000
                loss_classif = 1
        except:
            print('error3')
        return {"loss": loss_vae, "loss_clf": loss_classif}

# Условный вариационный автоэнкодер + классификатор 
class ClassCVAE(tf.keras.Model):
    def model_parametr(self,model_name):
        with open(model_name+'.txt','w') as f:
            f.write(str(self.latent_dim)+'\n')
            f.write(self.backborne+'\n')
            f.write(self.outlevel+'\n')
            f.write(str(self.input_shape_im[0])+'\n')
            f.write(str(self.input_shape_im[1])+'\n')
            f.write(str(self.input_shape_im[2])+'\n')
            f.close()

    def vae_reconstruction_loss(self,y_true, y_predict):
            reconstruction_loss_factor = 1.0
            reconstruction_loss = tf.math.reduce_mean(tf.math.square(y_true - y_predict),axis=[1, 2, 3])
            return tf.cast(reconstruction_loss_factor * reconstruction_loss/224/224/3,dtype = tf.float32)

    def vae_kl_loss(self,encoder_mu, encoder_log_variance):
            dd = (encoder_mu)**2 - tf.exp(encoder_log_variance)
            #print('dd:',dd.shape)
            dd = 1.0 + encoder_log_variance - (encoder_mu)**2 - tf.exp(encoder_log_variance)
            #print('dd:',dd.shape,np.sum(tf.math.is_nan(dd)),np.sum(dd.numpy()),np.max(dd.numpy()),np.min(dd.numpy()))
            dd = np.sum(dd.numpy(),axis = 1)
            #print('dd:',dd.shape)
            kl_loss = -0.5 * np.sum(1.0 + encoder_log_variance.numpy() - (encoder_mu.numpy())**2 - np.exp(encoder_log_variance.numpy()), axis = 1)
            #print(kl_loss)
            return tf.cast(kl_loss,dtype =tf.float32)

    def vae_kl_loss_metric(self,y_true, y_predict,encoder_mu, encoder_log_variance):
            kl_loss = -0.5 * tf.math.reduce_sum(1.0 + encoder_log_variance - tf.math.square(encoder_mu) - tf.math.exp(encoder_log_variance), axis=1)
            return kl_loss

    def vae_loss(self,y_true, y_predict):
            reconstruction_loss = self.vae_reconstruction_loss(y_true, y_predict)
            kl_loss = self.vae_kl_loss(y_true, y_predict)

            loss = reconstruction_loss + kl_loss
            return tf.cast(loss,dtype=tf.float32)



    def sampling(self,mu_log_variance):
        mu, log_variance = mu_log_variance
        epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
        random_sample = mu + tf.keras.backend.exp(log_variance / 2) * epsilon
        return random_sample

    def __init__(self, latent_dim, encoder_X, decoder_X, input_shape_im=[224,224, 9], n_latent=8, n_dec=32, model_name ='', outlevel ='block_6_expand_relu',backborne = 'mob'):
        
        '''
        encoder_X,decoder_X - энкодер, декодер
        input_shape_im = [224,224,3], размер входа
        lambda_=100.0, - коэффициент для потерь
        lambda_identity=1.5, - коэффициент потерь идентичности
        latent_dim = 256, - размер латентного слоя
        n_latent=8, n_dec=32, - параметр латентного слоя, параметр декодера
        mod_ = 'mod', - использовать предобученную сеть с именем из mod 
        deep =  3
        outlevel='block_5_expand_relu' - выходной слой претренированной сети
        '''           
        
        print('init')
        super(ClassCVAE, self).__init__()

        self.optimizer = keras.optimizers.Adam(1e-4)
        self.loss_fn = keras.losses.BinaryCrossentropy( from_logits=False)
        self.input_shape_im = input_shape_im

        if model_name != '':
            with open(model_name+'.txt','r') as f:
                self.latent_dim = int(f.readline())
                self.backborne = f.readline().replace('\n','')
                self.outlevel = f.readline().replace('\n','')
                i1 = int(f.readline())
                i2 = int(f.readline())
                i3 = int(f.readline())
                self.input_shape_im = [i1,i2,i3]
                f.close()
        else:
            self.latent_dim = latent_dim
            self.backborne = backborne
            self.outlevel = outlevel
            self.input_shape_im = input_shape_im
            self.mse = keras.losses.MeanSquaredError()

        input_enc = keras.layers.Input(shape=(input_shape_im[0], input_shape_im[1], input_shape_im[2]), name="VAE_input")
        #x_enc = encoder_X(input_enc)

        if self.backborne == 'mob':
            base_classif = keras.applications.mobilenet_v2.MobileNetV2(
                input_shape=(input_shape_im[0], input_shape_im[1],input_shape_im[2]),
                alpha=1.0, include_top=False, weights='imagenet')
            x_base_end = base_classif.get_layer(self.outlevel ).output # 3-5613-14 6 - 28
        else:
            base_classif = tf.keras.models.load_model(self.backborne, custom_objects={"Swish": Swish})
            x_base_end = base_classif.get_layer(self.outlevel).output  # 1920

        down_model = keras.Model(base_classif.input,x_base_end)
        x_base = down_model(input_enc)

        x_f = keras.layers.Flatten()(x_base)
        print(x_f.shape,x_base.shape)

        x_hiden_m = keras.layers.Dense(self.latent_dim)(x_f)
        x_hiden_s = keras.layers.Dense(self.latent_dim)(x_f)


        x_class = keras.layers.Dense(latent_dim, activation=tf.nn.leaky_relu)(x_f)
        x_class = keras.layers.Dropout(0.2)(x_class)
        x_class = keras.layers.BatchNormalization()(x_class)

        x_class = keras.layers.concatenate([x_hiden_m,x_hiden_s,x_class])

        x_class = keras.layers.BatchNormalization()(x_class)
        x_class = keras.layers.Dense(latent_dim // 2 , activation = tf.nn.leaky_relu)(x_class)
        x_class = keras.layers.Dropout(0.2)(x_class)
        x_class = keras.layers.BatchNormalization()(x_class)

        output_class = keras.layers.Dense(1, activation='sigmoid')(x_class)
        
        encoder_output = keras.layers.Lambda(self.sampling, name="encoder_output")([x_hiden_m, x_hiden_s])
        print(encoder_output.shape)



        decoder_input = keras.layers.Input(shape=(latent_dim,))

        dec_1 = keras.layers.Dense(units=n_latent * n_latent * n_dec, activation=tf.nn.relu)(decoder_input)
        print(dec_1.shape)
        dec_1 = keras.layers.Reshape(target_shape=(n_latent, n_latent, n_dec))(dec_1)
        dec_1 = decoder_X(dec_1)
        print(dec_1.shape)
        
        self.dec_X = keras.Model(decoder_input,dec_1)

        decoder_output = self.dec_X(encoder_output)
        self.enc_X = keras.Model(input_enc, [x_hiden_m,x_hiden_s,encoder_output])
        self.classificator = keras.Model(input_enc, output_class)
        self.model_= keras.Model(input_enc, [decoder_output, output_class])

        print(decoder_output.shape)
        self.classificator.summary()

    def call(self,data):
        #super(ClassCVAE, self).call()
        if isinstance(data,tuple):
            x = data[0]
        else:
            x = data
        return self.model_(x)
    def compile(self,optimizer =tf.keras.optimizers.Adam()):
        super(ClassCVAE, self).compile()

        self.model_.optimizer = optimizer
        self.enc_X.optimizer = optimizer
        self.dec_X.optimizer = optimizer
        self.classificator.optimizer = optimizer
        self.classificator.compile(optimizer = optimizer,loss = keras.losses.BinaryCrossentropy(from_logits = False),metrics =['accuracy'])

    def train_step(self,data):
        #super(ClassCVAE, self).train_step()
        try:
          loss_vae = 1000
          loss_classif = 1
          x,y,label= data
          #y,label = target
          if label.shape[0] is not None :
            #print('label:',label)
            print(data[0].shape)

            try:
                self.classificator.trainable = False
                self.enc_X.trainable = True
                self.dec_X.trainable = True
                with tf.GradientTape() as tape:
                    #print('label: ',label)
                    ind_ = np.where( label.numpy() == 0)[0]
                    print(ind_)
                    if  len(ind_)>0:#0
                        print(x.shape)
                        xp = tf.cast([x[ik,:,:,:] for ik in ind_],dtype = tf.float32)
                        x_m,x_s,x_x = self.enc_X(xp)
                        print(x_x.shape)
                        y_pred = self.dec_X(x_x)
                        print('y-pred :',y_pred.shape)
                        loss_reconstr = self.vae_reconstruction_loss(xp,y_pred)

                    else:
                        x_m,x_s,x_x = self.enc_X(x)
                        y_pred = self.dec_X(x_x)
                        #print('y_pred:',y_pred.shape)
                        loss_reconstr = self.vae_reconstruction_loss(x,y_pred)
                        #print(x.shape,loss_reconstr.shape)
                    #print(x_m.shape,x_s.shape)    
                    loss_kl = self.vae_kl_loss(encoder_mu=x_m,encoder_log_variance=x_s)
                    #print(loss_kl.shape)
                    loss_vae = tf.reduce_mean(loss_kl+loss_reconstr)
                    #print(loss_vae.shape)
                    grad = tape.gradient(loss_vae,self.model_.trainable_variables)
                self.optimizer.apply_gradients(zip(grad, self.model_.trainable_variables))

            except:
                print('error')

            try:
                self.classificator.trainable = True
                #self.enc_X.trainable = False
                self.dec_X.trainable = False
                #self.model_.trainable = False
                with tf.GradientTape() as tape:
                    label_pred = self.classificator(x)
                    loss_classif = self.loss_fn(label,label_pred)
                    grad = tape.gradient(loss_classif,self.classificator.trainable_variables)
                self.optimizer.apply_gradients(zip(grad, self.classificator.trainable_variables))
                self.model_.trainable = True
                self.enc_X.trainable = True
                self.dec_X.trainable = True
            except:
                print('error1')
                loss_vae = 1000
                loss_classif = 1
          else:
            loss_vae = 1000
            loss_classif = 1
        except:
            print('error2')
        return  {  "loss": loss_vae, "loss_clf":loss_classif }
    
    def test_step(self,data):
        #super(ClassCVAE, self).test_step()
        loss_vae = 1000
        loss_classif = 1
        try:

            x,y,label = data

            if 1:#x.shape[0] is not None:
                try:
                    x_m,x_s,x_x = self.enc_X(x)
                    y_pred = self.dec_X(x_x)
                    loss_reconstr = self.vae_reconstruction_loss(x,y_pred)
                    #print('loss_reconstr',loss_reconstr.shape)
                    loss_kl = self.vae_kl_loss(encoder_mu=x_m,encoder_log_variance=x_s)
                    #print('loss_kl:',loss_kl.shape)
                    loss_vae = tf.cast(np.sum(loss_kl.numpy()+loss_reconstr.numpy()),dtype = tf.float32)
                    #print('loss_vae:',loss_vae.shape)
                    label_pred = self.classificator(x)
                    #print('test cl:',y_pred.shape)
                    
                    loss_classif = self.loss_fn(label,label_pred)
                except:
                    print('error')
                    loss_vae = 1000
                    loss_classif = 1
            else:
                loss_vae = 1000
                loss_classif = 1
        except:
            print('error3')
        return  {  "loss": loss_vae, "loss_clf":loss_classif }


###  CVAE+Classificator
class ClassConditionVAE(tf.keras.Model):
    #save model parametr from experiment
    def model_parametr(self, model_name):
        with open(model_name + '.txt', 'w') as f:
            f.write(str(self.latent_dim) + '\n')
            f.write(self.backborne + '\n')
            f.write(self.outlevel + '\n')
            f.write(str(self.input_shape_im[0]) + '\n')
            f.write(str(self.input_shape_im[1]) + '\n')
            f.write(str(self.input_shape_im[2]) + '\n')
            f.close()

    def vae_reconstruction_loss(self, y_true, y_predict):
        reconstruction_loss_factor = 1.000
        print('***',y_true.shape,y_predict.shape,np.max(y_predict.numpy()))
        reconstruction_loss = self.mae(y_true,y_predict) #tf.math.reduce_mean(tf.math.square(y_true - y_predict), axis=[1, 2, 3]) * 224 * 224
        print(reconstruction_loss)
        return tf.cast(reconstruction_loss_factor * reconstruction_loss, dtype=tf.float32)

    def vae_kl_loss(self, encoder_mu, encoder_log_variance):
        dd = (encoder_mu) ** 2 - tf.exp(encoder_log_variance)
        # print('dd:',dd.shape)
        dd = 1.0 + encoder_log_variance - (encoder_mu) ** 2 - tf.exp(encoder_log_variance)
        # print('dd:',dd.shape,np.sum(tf.math.is_nan(dd)),np.sum(dd.numpy()),np.max(dd.numpy()),np.min(dd.numpy()))
        dd = np.sum(dd.numpy(), axis=1)
        # print('dd:',dd.shape)
        kl_loss = -0.5 * np.sum(
            1.0 + encoder_log_variance.numpy() - (encoder_mu.numpy()) ** 2 - np.exp(encoder_log_variance.numpy()),
            axis=1)
        # print(kl_loss)
        return tf.cast(np.mean(kl_loss), dtype=tf.float32)

    def vae_kl_loss_metric(self, y_true, y_predict, encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * tf.math.reduce_sum(
            1.0 + encoder_log_variance - tf.math.square(encoder_mu) - tf.math.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_loss(self, y_true, y_predict):
        reconstruction_loss = self.vae_reconstruction_loss(y_true, y_predict)
        kl_loss = self.vae_kl_loss(y_true, y_predict)

        loss = reconstruction_loss + kl_loss
        return tf.cast(loss, dtype=tf.float32)

    def sampling(self, mu_log_variance):
        mu, log_variance = mu_log_variance
        epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
        random_sample = mu + tf.keras.backend.exp(log_variance / 2) * epsilon
        return random_sample

    def __init__(self, latent_dim, encoder_X, decoder_X, input_shape_im=[224, 224, 9], n_latent=8, n_dec=32,
                 model_name='', outlevel='block_5_expand_relu', backborne='mob'):
                        
        '''
        encoder_X,decoder_X - энкодер, декодер
        input_shape_im = [224,224,3], размер входа
        lambda_=100.0, - коэффициент для потерь
        lambda_identity=1.5, - коэффициент потерь идентичности
        latent_dim = 256, - размер латентного слоя
        n_latent=8, n_dec=32, - параметр латентного слоя, параметр декодера
        mod_ = 'mod', - использовать предобученную сеть с именем из mod 
        deep =  3
        outlevel='block_5_expand_relu' - выходной слой претренированной сети
        '''           

        print('init')
        super(ClassConditionVAE, self).__init__()
        self.optimizer = keras.optimizers.Adam(1e-4)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
        self.input_shape_im = input_shape_im
        self.mse = keras.losses.MeanSquaredError()
        if model_name != '':
            with open(model_name + '.txt', 'r') as f:
                self.latent_dim = int(f.readline())
                self.backborne = f.readline().replace('\n', '')
                self.outlevel = f.readline().replace('\n', '')
                i1 = int(f.readline())
                i2 = int(f.readline())
                i3 = int(f.readline())
                self.input_shape_im = [i1, i2, i3]
                f.close()
        else:
            self.latent_dim = latent_dim
            self.backborne = backborne
            self.outlevel = outlevel
            self.input_shape_im = input_shape_im
        # model CVAE+Classificator
        input_dm = keras.layers.Input(shape=(1,), name="class_input")
        input_enc = keras.layers.Input(shape=(input_shape_im[0], input_shape_im[1], input_shape_im[2]),
                                       name="VAE_input")
        if self.backborne == 'mob':
            base_classif = keras.applications.mobilenet_v2.MobileNetV2(
                input_shape=(input_shape_im[0], input_shape_im[1], input_shape_im[2]),
                alpha=1.0, include_top=False, weights='imagenet')
            x_base_end = base_classif.get_layer(self.outlevel).output  # 3-56 13-14 6 - 28
        else:
            base_classif = tf.keras.models.load_model(self.backborne, custom_objects={"Swish": Swish})
            x_base_end = base_classif.get_layer(self.outlevel).output  # 1920

        down_model = keras.Model(base_classif.input, x_base_end)
        x_base = down_model(input_enc)

        x_f = keras.layers.Flatten()(x_base)
        x_f = keras.layers.concatenate([x_f,input_dm])
        #x_f = keras.layers.Dropout(0.2)(x_f)
        x_f = keras.layers.BatchNormalization()(x_f)
        print(x_f.shape, x_base.shape)
        x_f = keras.layers.Dense(latent_dim)(x_f)

        x_hiden_m = keras.layers.Dense(latent_dim)(x_f)
        x_hiden_s = keras.layers.Dense(latent_dim)(x_f)

        x_class = keras.layers.Dense(latent_dim, activation=tf.nn.leaky_relu)(x_f)
        #x_class = keras.layers.Dropout(0.2)(x_class)
        x_class = keras.layers.BatchNormalization()(x_class)



        encoder_output = keras.layers.Lambda(self.sampling, name="encoder_output")([x_hiden_m, x_hiden_s])
        print(encoder_output.shape)

        x_class = keras.layers.concatenate([encoder_output, x_class])

        x_class = keras.layers.BatchNormalization()(x_class)
        x_class = keras.layers.Dense(latent_dim , activation=tf.nn.leaky_relu)(x_class)
        #x_class = keras.layers.Dropout(0.2)(x_class)
        x_class = keras.layers.BatchNormalization()(x_class)

        output_class = keras.layers.Dense(1, activation='sigmoid')(x_class)


        decoder_input = keras.layers.Input(shape=(latent_dim,))
        decoder_input_cl = input_dm #keras.layers.Input(shape=(1,))
        dec_1= keras.layers.concatenate([decoder_input, decoder_input_cl])
        dec_1 = keras.layers.Dense(units=n_latent * n_latent * n_dec, activation=tf.nn.relu)(dec_1)
        #dec_1= keras.layers.Dropout(0.2)(dec_1)
        dec_1 = keras.layers.BatchNormalization()(dec_1)
        print(dec_1.shape)
        dec_1 = keras.layers.Reshape(target_shape=(n_latent, n_latent, n_dec))(dec_1)
        dec_1 = decoder_X(dec_1)
        print(dec_1.shape)

        self.dec_X = keras.Model([decoder_input, decoder_input_cl], dec_1)

        decoder_output = self.dec_X([encoder_output,decoder_input_cl])
        self.enc_X = keras.Model([input_enc, input_dm], [x_hiden_m, x_hiden_s, encoder_output])
        self.classificator = keras.Model([input_enc,input_dm], output_class)
        self.model_ = keras.Model([input_enc,input_dm], [decoder_output, output_class])
        #self.vae = keras.Model([input_enc, input_dm], [decoder_output])

        print(decoder_output.shape)
        self.classificator.summary()
        self.dec_X.summary()
        decoder_X.summary()
        #self.vae.summary()

    def call(self, data):
        #
        if isinstance(data, tuple):
            x,dm = data[0],data[1]
            return self.model_([x, dm])
        else:
            x = data
            return self.model_(x)

    def compile(self, optimizer=tf.keras.optimizers.Adam()):
        super(ClassConditionVAE, self).compile()

        self.model_.optimizer = optimizer
        self.enc_X.optimizer = optimizer
        self.dec_X.optimizer = optimizer
        #self.vae.optimizer =optimizer
        self.classificator.optimizer = optimizer
        self.classificator.compile(optimizer = optimizer,loss = keras.losses.BinaryCrossentropy(from_logits = False),metrics =['accuracy'])

    def train_step(self, data):
        # super(ClassCVAE, self).train_step()
        try:
            loss_vae = 1000
            loss_classif = 1
            x_dm, label = data
            x,dm = x_dm
            # y,label = target
            if label.shape[0] is not None:
                try:
                    # train CVAE model
                    self.classificator.trainable = False
                    #self.vae.trainable = True
                    self.dec_X.trainable = True
                    self.enc_X.trainable = True
                    #self.model_.trainable = True
                    with tf.GradientTape() as tape:
                        #
                        ind_ = np.where(label.numpy() == 0)[0]
                        
                        if 0:#len(ind_) > 0:  # 0
                            
                            xp = tf.cast([x[ik, :, :, :] for ik in ind_], dtype=tf.float32)

                            x_dm = tf.cast([dm[ik] for ik in ind_], dtype=tf.float32)
                            
                            x_m, x_s, x_x = self.enc_X([xp,x_dm])
                            
                            y_pred = self.dec_X([x_x,x_dm])
                            x_x = tf.clip_by_value(x_x, clip_value_min=-10.0, clip_value_max=10.0)

                            y_pred = self.dec_X([x_x,x_dm])
                            
                            print('y-pred :', y_pred.shape)

                            loss_reconstr = tf.reduce_sum(keras.losses.binary_crossentropy(xp, y_pred), axis=(1, 2)) /xp.shape[1]/xp.shape[1]
                        else:
                            x_m, x_s, x_x = self.enc_X([x,dm])
                            #print(x_x.shape)
                            x_x = tf.clip_by_value(x_x, clip_value_min=-10.0, clip_value_max=10.0)
                            y_pred = self.dec_X([x_x,dm])
                            
                            loss_reconstr = tf.reduce_sum(keras.losses.binary_crossentropy(x, y_pred), axis=(1, 2))/x.shape[1]/x.shape[1]

                        loss_reconstr = tf.clip_by_value(loss_reconstr, clip_value_min=-1.0, clip_value_max=1.0)
                        loss_kl = self.vae_kl_loss(encoder_mu=x_m, encoder_log_variance=x_s)/self.latent_dim
                        loss_kl = tf.clip_by_value(loss_kl , clip_value_min=-1.0, clip_value_max=1.0)
                       
                        loss_vae = tf.reduce_mean(loss_kl*10 + loss_reconstr)/10000.0
                        
                        loss_vae = tf.clip_by_value(loss_vae, clip_value_min=-10.0, clip_value_max=10.0)
                       
                        grad = tape.gradient(loss_vae, self.model_.trainable_variables)

                        #print(len(grad))
                    self.optimizer.apply_gradients(zip(grad, self.model_.trainable_variables))


                except:
                    print('error train 1')

                try:
                    #train top of Classificator

                    self.dec_X.trainable = False
                    self.classificator.trainable = True
                    self.enc_X.trainable = False
                    with tf.GradientTape() as tapec:
                        label_pred = self.classificator([x,dm])
                        label_e= self.enc_X([x, dm])
                        loss_classif = self.loss_fn(tf.reshape(label,[-1,1]), label_pred)/x.shape[0]
                        print('::',loss_classif)
                        loss_classif = tf.clip_by_value(loss_classif , clip_value_min=-1.0, clip_value_max=1.0)
                        gradc = tapec.gradient(loss_classif, self.classificator.trainable_variables)

                    self.optimizer.apply_gradients(zip(gradc, self.classificator.trainable_variables))
                    self.model_.trainable = True
                    self.enc_X.trainable = True
                    self.dec_X.trainable = True
                except:
                    print('error train 2')
                    loss_vae = 1000
                    loss_classif = 1
            else:
                loss_vae = 1000
                loss_classif = 1
        except:
            print('error train')
        return {"loss": loss_vae, "loss_clf": loss_classif}

    def test_step(self, data):
        # super(ClassCVAE, self).test_step()
        loss_vae = 1000
        loss_classif = 1
        try:
            x_dm, label = data
            x, dm = x_dm

            if 1:  # x.shape[0] is not None:
                try:
                    x_m, x_s, x_x = self.enc_X([x,dm])
                    x_x = tf.clip_by_value(x_x, clip_value_min=-10.0, clip_value_max=10.0)
                    y_pred = self.dec_X([x_x,dm])
                    loss_reconstr = tf.reduce_sum(keras.losses.binary_crossentropy(x, y_pred), axis=(1, 2)) /x.shape[1]/x.shape[1]
                    loss_kl = self.vae_kl_loss(encoder_mu=x_m, encoder_log_variance=x_s)/self.latent_dim
                    loss_vae = tf.cast(np.sum(loss_kl.numpy() + loss_reconstr.numpy()), dtype=tf.float32)
                    label_pred = self.classificator([x,dm])

                    loss_classif = self.loss_fn(tf.reshape(label,[-1,1]), label_pred)/x.shape[0]
                except:
                    print('error')
                    loss_vae = 1000
                    loss_classif = 1
            else:
                loss_vae = 1000
                loss_classif = 1
        except:
            print('error3')
        return {"loss": loss_vae, "loss_clf": loss_classif}

# merquri backborne
class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
# Вариационный автоэнкодер
class VAE(keras.Model):
    def __init__(self, latent_dim,encoder_X, decoder_X, input_shape_im=[224, 224, 3], n_latent=8, n_dec=32,
                 model_name='', outlevel='block_5_expand_relu', backborne='mob',**kwargs):
                
        '''
        encoder_X,decoder_X - энкодер, декодер
        input_shape_im = [224,224,3], размер входа
        lambda_=100.0, - коэффициент для потерь
        lambda_identity=1.5, - коэффициент потерь идентичности
        latent_dim = 256, - размер латентного слоя
        n_latent=8, n_dec=32, - параметр латентного слоя, параметр декодера
        mod_ = 'mod', - использовать предобученную сеть с именем из mod 
        deep =  3
        outlevel='block_5_expand_relu' - выходной слой претренированной сети
        '''           

        super(VAE, self).__init__(**kwargs)


        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.backborne = backborne
        self.outlevel = outlevel

        x_encoder_inp = keras.layers.Input(shape = input_shape_im)

        if self.backborne == 'mob':
            base_classif = keras.applications.mobilenet_v2.MobileNetV2(
                input_shape=(input_shape_im[0], input_shape_im[1], input_shape_im[2]),
                alpha=1.0, include_top=False, weights='imagenet')
            x_base_end = base_classif.get_layer(self.outlevel).output  # 3-56 13-14 6 - 28
        else:
            base_classif = tf.keras.models.load_model(self.backborne, custom_objects={"Swish": Swish})
            x_base_end = base_classif.get_layer(self.outlevel).output  # 1920
            #base_classif = encoder_X
            x_base_end = base_classif.output
        base_classif.summary()
        base = keras.Model(base_classif.input,x_base_end,name='base_encoder')

        x_f = base(x_encoder_inp)

        x_f = keras.layers.Flatten()(x_f)
        x_f = keras.layers.Dense(latent_dim*3)(x_f)
        x_f = keras.layers.Dense(latent_dim)(x_f)
        x_m =  keras.layers.Dense(latent_dim)(x_f)
        x_lv = keras.layers.Dense(latent_dim)(x_f)
        x_x = Sampling()([x_m,x_lv])

        self.encoder =keras.Model(x_encoder_inp,[x_m,x_lv,x_x],name='encoder')

        decoder_input = keras.layers.Input(shape=(latent_dim,))


        dec_1 = keras.layers.Dense(units=n_latent * n_latent * n_dec, activation=tf.nn.relu)(decoder_input)
        # dec_1= keras.layers.Dropout(0.2)(dec_1)
        dec_1 = keras.layers.BatchNormalization()(dec_1)
        print(dec_1.shape)
        dec_1 = keras.layers.Reshape(target_shape=(n_latent, n_latent, n_dec))(dec_1)
        dec_out = decoder_X(dec_1)
        print(dec_1.shape)
        self.decoder = keras.Model(decoder_input ,dec_out,name='decoder')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        if isinstance(data,tuple):
            data = data[0][0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            z = tf.clip_by_value(z, clip_value_min=-10.0, clip_value_max=10.0)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            ) / data.shape[0]/data.shape[1]/data.shape[1]
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.clip_by_value(kl_loss, clip_value_min=-1.0, clip_value_max=1.0)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))/128.0
            total_loss = (reconstruction_loss + kl_loss)/10000.0
        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0][0]
        z_mean, z_log_var, z = self.encoder(data)
        z = tf.clip_by_value(z, clip_value_min=-10.0, clip_value_max=10.0)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum( keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)))/data.shape[1]/data.shape[0]/data.shape[1]
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        kl_loss = tf.clip_by_value(kl_loss, clip_value_min=-1.0, clip_value_max=1.0)
        total_loss = reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    def compile(self, optimizer=tf.keras.optimizers.Adam()):
        super(VAE, self).compile()
        self.encoder.optimizer = optimizer
        self.decoder.optimizer = optimizer

    def call(self, data):
        #super(VAE, self).__call__()
        if isinstance(data,tuple):
            data = data[0]
        if data.shape[0]>0:
            z_mean, z_log_var, z = self.encoder(data)
            return self.decoder(z)

# Автоэнкодер + пирамида признаков 
class Autoencoder_fpn(keras.Model):
    def __init__(
            self,
            encoder_X = None,
            input_shape_im=[224, 224, 3],
            lambda_=100.0,
            lambda_identity=1.5,
            latent_dim=256,
            mod_='mod',
            deep = 3,
            outlevel='block_6_expand_relu'
    ):
                
        '''
        encoder_X, - энкодер
        input_shape_im = [224,224,3], размер входа
        lambda_=100.0, - коэффициент для потерь
        lambda_identity=1.5, - коэффициент потерь идентичности
        latent_dim = 256, - размер латентного слоя
        n_latent=8, n_dec=32, - параметр латентного слоя, параметр декодера
        mod_ = 'mod', - использовать предобученную сеть с именем из mod 
        deep =  3
        outlevel='block_5_expand_relu' - выходной слой претренированной сети
        '''           

        super(Autoencoder_fpn, self).__init__()

        self.outlevel = outlevel

        if mod_ == 'mod':
            base_classif = keras.applications.mobilenet_v2.MobileNetV2(
                input_shape=(input_shape_im[0], input_shape_im[1], input_shape_im[2]),
                alpha=1.0, include_top=False, weights='imagenet')
            # base_classif.summary()
            x_base_end = base_classif.get_layer(self.outlevel).output  # 3-56, 13-14, 6 - 28
            self.enc_X = keras.Model(base_classif.input, x_base_end)
            base_classif.trainable = False
            List_back = ['block_16_expand_relu',  # 7
                         'block_13_expand_relu',  # 14
                         'block_6_expand_relu',  # 28
                         'block_3_expand_relu',  # 56
                         'block_1_expand_relu']  # 112
        else:
            self.backborne = mod_
            # backborne = 'full.best.0600-0.0831-0.9774.hdf5', outlevel = 'out_swish'
            if mod_ != '':
                base_classif = tf.keras.models.load_model(self.backborne, custom_objects={"Swish": Swish})
                base_classif.summary()
                x_base_end = base_classif.get_layer(self.outlevel).output
                self.enc_X = keras.Model(base_classif.input, x_base_end)
                List_back = ['block_16_expand',  # 7
                             'block_13_expand',  # 14
                             'block_6_expand_swish',  # 28
                             'block_3_expand_swish',  # 56
                             'block_1_expand_swish']  # 112

            else:
                self.enc_X = encoder_X
                List_back = outlevel  # 112
        self.enc_X.summary()
        print(List_back[len(List_back) - deep:])

        self.latent_dim = latent_dim
        #
        input_enc = self.enc_X.layers[0].input
        x_enc_end = self.enc_X.layers[-1].output#(input_enc)
        x  = keras.layers.Conv2D(self.latent_dim * 4,(3,3),activation = 'relu',kernel_regularizer ='l2' , padding = 'same')(x_enc_end )
        x = keras.layers.Conv2D(self.latent_dim * 4, (3, 3), activation='relu',kernel_regularizer ='l2' , padding = 'same')(x)
        x = keras.layers.Conv2D(self.latent_dim * 4, (3, 3), activation='relu',kernel_regularizer ='l2' , padding = 'same')(x)
        x = keras.layers.BatchNormalization()(x)
        x_latent_enc = x
        list_fpn = []
        for jk,x_block in enumerate(List_back[len(List_back)-deep:]):
            x_skip = self.enc_X.get_layer(x_block).output
            print(x_skip)
            x = layers.Conv2D(x_skip.shape[-1], (1, 1),
                              strides=(1, 1),
                              activation='relu',
                              padding="same", kernel_regularizer='l2', name='t_skip' + str(jk)[0])(x)
            print(x.shape,x_skip.shape)
            x = keras.layers.Add()([x,x_skip ])
            x.shape
            list_fpn += [x]

            x = upsample(x,
                         filters= self.latent_dim * 4 //(2**jk),
                         activation=layers.LeakyReLU(0.2),
                         kernel_size=(4, 4),
                         strides=(2, 2))
            x = layers.Conv2D(latent_dim * 4 //(2**jk), (2, 2),
                                  strides=(1, 1),
                                  activation='relu',
                                  padding="same", kernel_regularizer ='l2' ,name = 'transpose_Conv2d'+str(jk)[0])(x)

        x_out = layers.Conv2D( input_shape_im[2], (1, 1),
                               strides=(1, 1),
                               activation = 'tanh',
                               padding="same",kernel_regularizer ='l2' )(x)
        print(x_out)
        print(input_enc)
        #
        self.model_auto = tf.keras.Model( input_enc ,x_out )
        self.model_auto.summary()


        x_rez = keras.layers.Add( name = 'substr_input-output_unet')([input_enc,x_out])

        list_fpn += [x_rez]
        list_out = []
        for x_f in list_fpn:
            x_c2 = keras.layers.BatchNormalization()(x_f)

            num_filters = 16
            n_r = int(np.log2(x_c2.shape[1])) - 1
            print(n_r)
            if n_r>0:
                for num_downsample_block in range(n_r):

                    num_filters *= 2
                    x_c2 = downsample(x_c2, filters=num_filters, activation=layers.LeakyReLU(0.2),
                    kernel_size=(4, 4),strides=(2, 2),  )
                    x_c2 = layers.Conv2D(latent_dim * 4 //(2**jk), (2, 2),
                                  strides=(1, 1),
                                  activation='relu',
                                  padding="same", kernel_regularizer ='l2' ,name = 'transpose_Conv2d'+str(n_r)[:1]+str(num_filters )[:2])(x_c2)

            x_c2 = keras.layers.GlobalAveragePooling2D()(x_c2)
            x_c2 = keras.layers.BatchNormalization()(x_c2)
            x_c2 = keras.layers.Dropout(0.2)(x_c2)

            list_out += [x_c2]

        x_c2 = tf.keras.layers.concatenate(list_out)
        x_c2 = keras.layers.Dense(1, activation='sigmoid', kernel_regularizer='l2', name='rez_class_out')(x_c2)

        self.classificator= keras.Model(input_enc,x_c2)

        self.lambda_ = lambda_
        self.lambda_identity = lambda_identity
        self.input_shape_im = input_shape_im

        ## super(Autoencoder_v, self).build(input_shape = input_shape_im)
        self.model_auto.summary()
        self.classificator.summary()

    def compile(
            self,
            optimizer,
            loss_fn
    ):

        self.optimizer = optimizer

        self.loss_fn = loss_fn

        self.identity_loss_fn = keras.losses.MeanAbsoluteError()
        self.loss_class = keras.losses.BinaryCrossentropy()
        self.metric_class = keras.metrics.BinaryCrossentropy()
        super(Autoencoder_fpn, self).compile(optimizer=self.optimizer, loss=loss_fn, metrics=[loss_fn])
        self.model_auto.compile(run_eagerly=True, optimizer=self.optimizer, loss=self.loss_fn, metrics=[self.loss_fn])
        self.classificator.compile(run_eagerly=True, optimizer=self.optimizer, loss=self.loss_class,
                                   metrics=[self.metric_class])
    def call(self, x):
        if isinstance(x, tuple):
            x = x[0]
            print('call:', x.shape)
            x = self.model_auto(x)
            return x

    def train_step(self, batch_data):
        # x desktop and y mobile
        if isinstance(batch_data, tuple):
            real_x_real_y, y_label = batch_data
            real_x, real_y = real_x_real_y

            # print(": : ",real_x.shape,real_y.shape)
            if real_x.shape[0] > 0:
                self.enc_X.trainable = False
                with tf.GradientTape(persistent=True) as tape:
                    # desk to mobile
                    ind_ = np.where(y_label.numpy() == 0)[0]
                    #print(ind_)
                    if len(ind_) > 0:  # 0
                        #print(x.shape)
                        real_xp = tf.cast([real_x[ik, :, :, :] for ik in ind_], dtype=tf.float32)
                        real_yp = tf.cast([real_x[ik,:,:,:] for ik in ind_], dtype=tf.float32)
                        #print(': - : ',real_xp.shape,real_yp.shape)
                    else:
                        real_xp = real_x
                        real_yp = real_x

                    fake_y = self.model_auto(real_xp)  # , training=True)
                    # <<<<<<< HEAD
                    # print(real_x[0,:2,:2,:3].shape,fake_y.shape)
                    if real_xp.shape[0] > 0:
                        #print('good')
                        # >>>>>>> dad92cb79778f74bd69380a6b591be42b9bd602b
                        if real_xp.shape[-1] > 3:
                            fake_fft_y = generator_auto.fft_y(fake_y[:, :, :, :3])
                        else:
                            fake_fft_y = fake_y
                    # =======
                    # >>>>>>> 26c05f395c14a934f344506d24fdc26cdfe8c8e2
                    else:
                        print('not good')

                        fake_fft_y = fake_y

                    if real_xp.shape[0] > 0:
                        # decoder loss
                        loss_G = self.loss_fn(real_xp[:, :, :, :3], fake_y[:, :, :, :3])

                        if real_xp.shape[-1] > 3:
                            id_loss_G = (self.identity_loss_fn(real_xp[:, :, :, 3:6], fake_fft_y[:, :, :,3:6]))
                        # Total loss
                        id_loss_G = 0.0
                        total_loss = loss_G  # + id_loss_G
                    else:
                        total_loss = self.loss_fn(real_xp[:, :, :, :3], fake_y[:, :, :, :3])
                        loss_G, id_loss_G = 0.0, 0.0

                # Get the gradients
                grads_G = tape.gradient(total_loss, self.model_auto.trainable_variables)

                # Update the weights rs
                self.optimizer.apply_gradients(zip(grads_G, self.model_auto.trainable_variables) )

                self.model_auto.trainable = False
                #self.dec_X.tarinable = False
                #self.latent_X.trainable = False

                with tf.GradientTape() as tape:

                    y_pred = self.classificator(real_x)
                    y_loss = self.loss_class(y_label, y_pred)
                    #print(y_loss)
                    y_grad = tape.gradient(y_loss, self.classificator.trainable_variables)
                self.optimizer.apply_gradients(zip(y_grad, self.classificator.trainable_variables))


                self.model_auto.trainable = True
                #self.dec_X.tarinable = True
                #self.latent_X.trainable = True

                return {
                    "loss": total_loss,
                    "loss_clf": y_loss,
                }
            else:
                return {
                    "loss": 1000,
                    "loss_clf": 1,
                }
        else:
            return {
                "loss": 1000,
                "loss_clf": 1,
            }

    def test_step(self, batch_data):
        # x desctop - y
        # print('test',batch_data[0].shape, batch_data[1].shape)
        if isinstance(batch_data, tuple):

            real_x_real_y, y_label = batch_data
            real_x, real_y = real_x_real_y
            #print('test', real_x.shape, real_y.shape, y_label.shape)

            #
            fake_y = self.model_auto(real_x, training=True)

            if fake_y.shape[0] > 0:
                #print('good')
                # <<<<<<< HEAD
                if real_x.shape[-1] > 3:
                    fake_fft_y = generator_auto.fft_y(fake_y)
                else:
                    fake_fft_y = fake_y

            else:
                #print('not good')
                fake_fft_y = fake_y
            #  output
            # decoder loss
            if fake_y.shape[0] > 0:
                # decoder loss
                loss_G = self.loss_fn(real_x[:, :, :, :3], fake_y[:, :, :, :3]) * self.lambda_

                # identity loss

                if real_y.shape[-1] > 3:
                    id_loss_G = (self.identity_loss_fn(real_x[:, :, :, 3:6],
                                                       fake_fft_y[:, :, :, 3:6]) * self.lambda_ * self.lambda_identity)
                else:
                    id_loss_G = 0.0

                # Total loss
                total_loss = loss_G  # + id_loss_G

            else:
                total_loss = self.loss_fn(real_x[:, :, :, :3], fake_y[:, :, :, :3]) * self.lambda_
                loss_G, id_loss_G = 0.0, 0.0

            y_pred = self.classificator(real_x)
            y_loss = self.loss_class(y_label, y_pred)
            return {
                "loss": total_loss,
                "loss_clf": y_loss,
            }
        else:
            return {
                "loss": 1000,
                "loss_clf": 1,
            }

# Автоэнкодер с пирамидой признаков версия 2
class Autoencoder_fpn_b(keras.Model):
    def __init__(
            self,
            encoder_X = None,
            input_shape_im=[224, 224, 3],
            lambda_=100.0,
            lambda_identity=1.5,
            latent_dim=256,
            mod_='../CycleGAN/full.0200-0.0933-0.9710.hdf5',
            deep = 5,
            outlevel='block_16_expand'
    ):
                
        '''
        encoder_X - энкодер
        input_shape_im = [224,224,3], размер входа
        lambda_=100.0, - коэффициент для потерь
        lambda_identity=1.5, - коэффициент потерь идентичности
        latent_dim = 256, - размер латентного слоя
        n_latent=8, n_dec=32, - параметр латентного слоя, параметр декодера
        mod_ = 'mod', - использовать предобученную сеть с именем из mod 
        deep =  3
        outlevel='block_5_expand_relu' - выходной слой претренированной сети
        '''           

        super(Autoencoder_fpn_b, self).__init__()

        self.outlevel = outlevel
        # input_enc = keras.layers.Input(shape=(input_shape_im[0], input_shape_im[1], input_shape_im[2]),
        #                               name="auto_input")
        print('BACKBORN')
        if mod_ == 'mod':
            base_classif = keras.applications.mobilenet_v2.MobileNetV2(
                input_shape=(input_shape_im[0], input_shape_im[1], input_shape_im[2]),
                alpha=1.0, include_top=False, weights='imagenet')
            # base_classif.summary()
            x_base_end = base_classif.get_layer(self.outlevel).output  # 3-56, 13-14, 6 - 28
            self.enc_X = keras.Model(base_classif.input, x_base_end)
            base_classif.trainable = False
            List_back = ['block_16_expand_relu',  # 7
                         'block_13_expand_relu',  # 14
                         'block_6_expand_relu',  # 28
                         'block_3_expand_relu',  # 56
                         'block_1_expand_relu']  # 112
            # x_base = down_model(input_enc)
        else:
            self.backborne = mod_
            # backborne = 'full.best.0600-0.0831-0.9774.hdf5', outlevel = 'out_swish'
            if mod_ != '':
                base_classif = tf.keras.models.load_model(self.backborne, custom_objects={"Swish": Swish})
                base_classif.summary()
                x_base_end = base_classif.get_layer(self.outlevel).output
                self.enc_X = keras.Model(base_classif.input, x_base_end)
                # self.enc_X = encoder_X
                List_back = ['block_16_expand',  # 7
                             'block_13_expand',  # 14
                             'block_6_expand_swish',  # 28
                             'block_3_expand_swish',  # 56
                             'block_1_expand_swish']  # 112

            else:
                self.enc_X = encoder_X
                List_back = outlevel  # 112


        self.enc_X.summary()
        print(List_back[len(List_back) - deep:])

        self.latent_dim = latent_dim

        input_enc = self.enc_X.layers[0].input
        x_enc_end = self.enc_X.layers[-1].output  # (input_enc)
        x = keras.layers.Conv2D(self.latent_dim * 4, (3, 3), activation='relu', kernel_regularizer='l2',
                                padding='same')(x_enc_end)
        x = keras.layers.Conv2D(self.latent_dim * 4, (3, 3), activation='relu', kernel_regularizer='l2',
                                padding='same')(x)
        x = keras.layers.Conv2D(self.latent_dim * 4, (3, 3), activation='relu', kernel_regularizer='l2',
                                padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x_latent_enc = x
        list_fpn = []
        for jk, x_block in enumerate(List_back[len(List_back) - deep:]):
            x_skip = self.enc_X.get_layer(x_block).output
            print(x_skip)
            x = layers.Conv2D(x_skip.shape[-1], (1, 1),
                              strides=(1, 1),
                              activation='relu',
                              padding="same", kernel_regularizer='l2', name='t_skip' + str(jk)[0])(x)

            x = keras.layers.BatchNormalization()(x)

            print(x.shape, x_skip.shape)
            x = keras.layers.Add()([x, x_skip])
            x.shape
            list_fpn += [x]
            x = layers.Conv2D(latent_dim * 4 // (2 ** jk), (2, 2),
                              activation='relu',
                              padding="same", kernel_regularizer='l2', name='Conv2d_1_' + str(jk)[0])(x)
            x = keras.layers.BatchNormalization()(x)
            x = layers.DepthwiseConv2D((4,4),
                              activation='relu',depth_multiplier=1,
                              padding="same", kernel_regularizer='l2', name='Conv2d_2_' + str(jk)[0])(x)
            x = keras.layers.BatchNormalization()(x)
            print('x trans:',x.shape)

            x = upsample(x,
                         filters=self.latent_dim * 4 // (2 ** jk),
                         activation=layers.LeakyReLU(0.2),
                         kernel_size=(4, 4),
                         strides=(2, 2))
            print('x trans:',x.shape)

            x = layers.DepthwiseConv2D((4, 4),
                              activation='relu',depth_multiplier=3,
                              padding="same", kernel_regularizer='l2', name='transpose_Conv2d' + str(jk)[0])(x)
            x = keras.layers.BatchNormalization()(x)
            print('x trans:',x.shape)
        x = layers.Conv2D(input_shape_im[2], (1, 1),
                              strides=(1, 1),
                              padding="same", kernel_regularizer='l2')(x)
        x_out = keras.layers.Add(name='substr_input-output_unet')([input_enc, x])
        x_out = keras.layers.BatchNormalization()(x_out)
        x_out = layers.DepthwiseConv2D(input_shape_im[2], (1, 1),
                              depth_multiplier=3,
                              activation='tanh',
                              padding="same", kernel_regularizer='l2')(x)

        print(x_out)
        # self.enc_X.summary()
        print(input_enc)
        #


        self.model_auto = tf.keras.Model(input_enc, x_out)
        self.model_auto.summary()

        x_rez =  x_out
        x = keras.layers.BatchNormalization()(x)
        list_fpn += [x_rez]
        list_out = []
        for x_f in list_fpn:
            x_c2 = keras.layers.BatchNormalization()(x_f)

            num_filters = 16
            n_r = int(np.log2(x_c2.shape[1])) - 1
            print(n_r)
            if n_r > 0:
                for num_downsample_block in range(n_r):
                    num_filters *= 2
                    x_c2 = downsample(x_c2, filters=num_filters, activation=layers.LeakyReLU(0.2),
                                      kernel_size=(4, 4), strides=(2, 2), )
                    x_c2 = layers.Conv2D(num_filters, (2, 2),
                                         strides=(1, 1),
                                         activation='relu',
                                         padding="same", kernel_regularizer='l2',
                                         name='transpose_Conv2d_1'+ str(n_r)[:1] + str(num_filters)[:2])(x_c2)
                    x_c2 = keras.layers.BatchNormalization()(x_c2)
                    x_c2 = layers.DepthwiseConv2D((3, 3),
                                         strides=(1, 1),
                                         activation='relu',depth_multiplier = 3,
                                         padding="same", kernel_regularizer='l2',
                                         name='depth_Conv2d_1' + str(n_r)[:1] + str(num_filters)[:2])(x_c2)
                    x_c2 = keras.layers.BatchNormalization()(x_c2)
                    x_c2 = layers.Conv2D(num_filters, (2, 2),
                                                  strides=(1, 1),
                                                  activation='relu',
                                                  padding="same", kernel_regularizer='l2',
                                                  name='transpose_Conv2d_2' + str(n_r)[:1] + str(num_filters)[:2])(x_c2)

                    x_c2 = layers.DepthwiseConv2D((3, 3),
                                         strides=(1, 1), depth_multiplier = 3,
                                         activation='relu',
                                         padding="same", kernel_regularizer='l2',
                                         name='depth_Conv2d_2' + str(n_r)[:1] + str(num_filters)[:2])(x_c2)
                    x_c2 = keras.layers.BatchNormalization()(x_c2)

            x_c2 = keras.layers.GlobalAveragePooling2D()(x_c2)
            x_c2 = keras.layers.BatchNormalization()(x_c2)
            x_c2 = keras.layers.Dropout(0.2)(x_c2)

            list_out += [x_c2]

        x_c2 = tf.keras.layers.concatenate(list_out)
        x_c2 = keras.layers.Dense(1, activation='sigmoid', kernel_regularizer='l2', name='rez_class_out')(x_c2)

        self.classificator = keras.Model(input_enc, x_c2)

        self.lambda_ = lambda_
        self.lambda_identity = lambda_identity
        self.input_shape_im = input_shape_im

        self.model_auto.summary()
        self.classificator.summary()

    def compile(
            self,
            optimizer,
            loss_fn
    ):

        self.optimizer = optimizer

        self.loss_fn = loss_fn

        self.identity_loss_fn = keras.losses.MeanAbsoluteError()
        self.loss_class = keras.losses.BinaryCrossentropy()
        self.metric_class = keras.metrics.BinaryCrossentropy()
        super(Autoencoder_fpn_b, self).compile(optimizer=self.optimizer, loss=loss_fn, metrics=[loss_fn])
        self.model_auto.compile(run_eagerly=True, optimizer=self.optimizer, loss=self.loss_fn, metrics=[self.loss_fn])
        self.classificator.compile(run_eagerly=True, optimizer=self.optimizer, loss=self.loss_class,
                                   metrics=[self.metric_class])

    def call(self, x):
        if isinstance(x, tuple):
            x = x[0]
            print('call:', x.shape)
            x = self.model_auto(x)
            return x

    def train_step(self, batch_data):
        # x desktop and y mobile
        if isinstance(batch_data, tuple):
            real_x_real_y, y_label = batch_data
            real_x, real_y = real_x_real_y

            if real_x.shape[0] > 0:
                self.enc_X.trainable = False
                with tf.GradientTape(persistent=True) as tape:
                    # desk to mobile
                    ind_ = np.where(y_label.numpy() == 0)[0]
                    # print(ind_)
                    if len(ind_) > 0:  # 0

                        real_xp = tf.cast([real_x[ik, :, :, :] for ik in ind_], dtype=tf.float32)
                        real_yp = tf.cast([real_x[ik, :, :, :] for ik in ind_], dtype=tf.float32)

                    else:
                        real_xp = real_x
                        real_yp = real_x

                    fake_y = self.model_auto(real_xp)  # , training=True)

                    if real_xp.shape[0] > 0:

                        if real_xp.shape[-1] > 3:
                            fake_fft_y = generator_auto.fft_y(fake_y[:, :, :, :3])
                        else:
                            fake_fft_y = fake_y

                    else:
                        print('not good')

                        fake_fft_y = fake_y

                    if real_xp.shape[0] > 0:
                        # decoder loss
                        loss_G = self.loss_fn(real_xp[:, :, :, :3], fake_y[:, :, :, :3])

                        if real_xp.shape[-1] > 3:
                            id_loss_G = (self.identity_loss_fn(real_xp[:, :, :, 3:6], fake_fft_y[:, :, :, 3:6]))
                        # Total loss
                        id_loss_G = 0.0
                        total_loss = loss_G  # + id_loss_G
                    else:
                        total_loss = self.loss_fn(real_xp[:, :, :, :3], fake_y[:, :, :, :3])
                        loss_G, id_loss_G = 0.0, 0.0

                # Get the gradients
                grads_G = tape.gradient(total_loss, self.model_auto.trainable_variables)

                # Update the weights rs
                self.optimizer.apply_gradients(zip(grads_G, self.model_auto.trainable_variables))

                self.model_auto.trainable = False

                with tf.GradientTape() as tape:

                    y_pred = self.classificator(real_x)
                    y_loss = self.loss_class(y_label, y_pred)
                    # print(y_loss)
                    y_grad = tape.gradient(y_loss, self.classificator.trainable_variables)
                self.optimizer.apply_gradients(zip(y_grad, self.classificator.trainable_variables))

                self.model_auto.trainable = True

                return {
                    "loss": total_loss,
                    "loss_clf": y_loss,
                }
            else:
                return {
                    "loss": 1000,
                    "loss_clf": 1,
                }
        else:
            return {
                "loss": 1000,
                "loss_clf": 1,
            }

    def test_step(self, batch_data):
        # x desctop - y

        if isinstance(batch_data, tuple):

            real_x_real_y, y_label = batch_data
            real_x, real_y = real_x_real_y
            # print('test', real_x.shape, real_y.shape, y_label.shape)

            #
            fake_y = self.model_auto(real_x, training=True)

            if fake_y.shape[0] > 0:
                # print('good')
                if real_x.shape[-1] > 3:
                    fake_fft_y = generator_auto.fft_y(fake_y)
                else:
                    fake_fft_y = fake_y
            else:
                fake_fft_y = fake_y
            #  output
            # decoder loss
            if fake_y.shape[0] > 0:
                # decoder loss
                loss_G = self.loss_fn(real_x[:, :, :, :3], fake_y[:, :, :, :3]) * self.lambda_

                # identity loss

                if real_y.shape[-1] > 3:
                    id_loss_G = (self.identity_loss_fn(real_x[:, :, :, 3:6],
                                                       fake_fft_y[:, :, :, 3:6]) * self.lambda_ * self.lambda_identity)
                else:
                    id_loss_G = 0.0

                # Total loss
                total_loss = loss_G  # + id_loss_G

            else:
                total_loss = self.loss_fn(real_x[:, :, :, :3], fake_y[:, :, :, :3]) * self.lambda_
                loss_G, id_loss_G = 0.0, 0.0

            y_pred = self.classificator(real_x)
            y_loss = self.loss_class(y_label, y_pred)
            return {
                "loss": total_loss,
                "loss_clf": y_loss,
            }
        else:
            return {
                "loss": 1000,
                "loss_clf": 1,
            }
            
# Автоэнкодер версия 2
class Autoencoder_m(keras.Model):
    def __init__(
            self,
            nk  = 4,
            input_shape_im=[224, 224, 3],
            lambda_=100.0,
            lambda_identity=1.5,
            latent_dim=256,
            mod_='mod',
            outlevel='block_6_expand_relu'
    ):        
        '''
        input_shape_im = [224,224,3], размер входа
        lambda_=100.0, - коэффициент для потерь
        lambda_identity=1.5, - коэффициент потерь идентичности
        latent_dim = 256, - размер латентного слоя
        n_latent=8, n_dec=32, - параметр латентного слоя, параметр декодера
        mod_ = 'mod', - использовать предобученную сеть с именем из mod 
        deep =  3
        outlevel='block_5_expand_relu' - выходной слой претренированной сети
        '''           

        super(Autoencoder_m, self).__init__()


        kernel_num = [32,64,128,256]
        kernel_size = [4,3]
        self.latent_dim = latent_dim
        #
        x = keras.layers.Input(shape=input_shape_im)
        input_enc = x
        List_back = []
        n4 = 32
        for jk in range(nk):
            x = downsample(x,
                           filters=n4 * 2 ** (jk ),
                           activation=layers.LeakyReLU(0.2),
                           kernel_size=(4, 4),
                           strides=(2, 2))
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)

            x = layers.Conv2D( 2**(jk+1) * n4, (4, 4),
                              strides=(1, 1),
                              activation='relu',
                              padding="same", kernel_regularizer='l2', name='transpose_Conv2d_down' + str(jk)[0])(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)

            x = layers.DepthwiseConv2D((2, 2),
                                       strides=(1, 1),
                                       activation='relu',
                                       padding="same", kernel_regularizer='l2', name='transpose_DEPTHW2d' + str(jk)[0])(
                x)

        x = downsample(x,
                       filters=2 ** (jk+1) * n4,
                       activation=layers.LeakyReLU(0.2),
                       kernel_size=(4, 4),
                       strides=(2, 2))
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        x_latent_enc = x
        s_x = x_latent_enc.shape

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(latent_dim)(x)

        x = keras.layers.Dense(n4 * s_x[1] *s_x[2] )(x)
        x = keras.layers.Reshape((s_x[1], s_x[2], n4))(x)

        x = upsample(x,
                     filters=n4 * 2 ** (nk + 1),
                     activation=layers.LeakyReLU(0.2),
                     kernel_size=(3, 3),
                     strides=(2, 2))
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)

        x = layers.Conv2D(n4 * 2 ** (nk - jk), (2, 2),  # n4*(2**(nk-jk)), (2, 2),
                          strides=(1, 1),
                          activation='relu',
                          padding="same", kernel_regularizer='l2', name='transpose_Conv2d_hiden' + str(jk)[0])(x)

        for jk in range(nk):
            x = upsample(x,
                         filters=n4 * 2**(nk-jk),
                         activation=layers.LeakyReLU(0.2),
                         kernel_size=(3, 3),
                         strides=(2, 2))
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)

            x = layers.Conv2D(n4 * 2**(nk-jk), (2, 2),  # n4*(2**(nk-jk)), (2, 2),
                              strides=(1, 1),
                              activation='relu',
                              padding="same", kernel_regularizer='l2', name='transpose_Conv2d_up' + str(jk)[0])(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)

            x = layers.DepthwiseConv2D((3, 3),
                                       strides=(1, 1),
                                       activation='relu',
                                       padding="same", kernel_regularizer='l2',
                                       name='transpose_DEPTHW2d_up' + str(jk)[0])(x)

        x_out = layers.Conv2D(input_shape_im[2], (1, 1),
                              strides=(1, 1),
                              activation='tanh',
                              padding="same", kernel_regularizer='l1', name='auto_out')(x)
        print(x_out)
        #
        self.model_auto = tf.keras.Model( input_enc ,x_out )
        self.model_auto.summary()

        self.lambda_ = lambda_
        self.lambda_identity = lambda_identity
        self.input_shape_im = input_shape_im

        ## super(Autoencoder_v, self).build(input_shape = input_shape_im)
        self.model_auto.summary()

    def compile(
            self,
            optimizer,
            loss_fn
    ):

        self.optimizer = optimizer

        self.loss_fn = loss_fn
        super(Autoencoder_m, self).compile(optimizer=self.optimizer, loss=loss_fn, metrics=[loss_fn])
        self.model_auto.compile( optimizer=self.optimizer, loss=self.loss_fn, metrics=[self.loss_fn])
        
    def call(self, x):
        if isinstance(x, tuple):
            x = x[0]
            print('call:', x.shape)
            x = self.model_auto(x)
            return x

    def train_step(self, batch_data):
        # x desktop and y mobile
        if isinstance(batch_data, tuple):
            real_x_real_y, y_label = batch_data
            real_x, real_y = real_x_real_y

            # print(": : ",real_x.shape,real_y.shape)
            if real_x.shape[0] > 0:
                ind_ = np.where(y_label.numpy() == 0)[0]
                # print(ind_)
                if len(ind_) > 0:  # 0
                    # print(x.shape)
                    real_xp = tf.cast([real_x[ik, :, :, :] for ik in ind_], dtype=tf.float32)
                    real_yp = tf.cast([real_x[ik, :, :, :] for ik in ind_], dtype=tf.float32)
                    # print(': - : ',real_xp.shape,real_yp.shape)
                else:
                    real_xp = real_x
                    real_yp = real_x

                with tf.GradientTape(persistent=True) as tape:
                    fake_y = self.model_auto(real_xp)  # , training=True)
                    if real_xp.shape[0] > 0:
                        # decoder loss
                        loss_G = self.loss_fn(real_xp[:, :, :, :3], fake_y[:, :, :, :3])

                        total_loss = loss_G  # + id_loss_G
                    else:
                        total_loss = self.loss_fn(real_xp[:, :, :, :3], fake_y[:, :, :, :3])
                        loss_G, id_loss_G = 0.0, 0.0

                # Get the gradients
                grads_G = tape.gradient(total_loss, self.model_auto.trainable_variables)

                # Update the weights rs
                self.optimizer.apply_gradients(zip(grads_G, self.model_auto.trainable_variables) )


                return {
                    "loss": total_loss,
                }
            else:
                return {
                    "loss": 1000,
                }
        else:
            return {
                "loss": 1000,
            }

    def test_step(self, batch_data):
        # x desctop - y
        if isinstance(batch_data, tuple):

            real_x_real_y, y_label = batch_data
            real_x, real_y = real_x_real_y
            ind_ = np.where(y_label.numpy() == 0)[0]
            if len(ind_) > 0:  # 0
                real_xp = tf.cast([real_x[ik, :, :, :] for ik in ind_], dtype=tf.float32)
                real_yp = tf.cast([real_x[ik, :, :, :] for ik in ind_], dtype=tf.float32)
            else:
                real_xp = real_x
                real_yp = real_x

            #
            fake_y = self.model_auto(real_xp, training=True)

            # decoder loss
            if fake_y.shape[0] > 0:
                # decoder loss
                loss_G = self.loss_fn(real_xp[:, :, :, :3], fake_y[:, :, :, :3])

                total_loss = loss_G  # + id_loss_G

            else:
                total_loss = self.loss_fn(real_xp[:, :, :, :3], fake_y[:, :, :, :3])
                loss_G, id_loss_G = 0.0, 0.0

            return {
                "loss": total_loss,
            }
        else:
            return {
                "loss": 1000,
            }


# Автоэнкодер UNet версия
class Autoencoder_unet_m(keras.Model):
    def __init__(
            self,
            nk  = 2,
            input_shape_im=[224, 224, 3],
            lambda_=100.0,
            lambda_identity=1.5,
            latent_dim=256,
            mod_='mod',
            outlevel='block_6_expand_relu'
    ):
                
        '''
        input_shape_im = [224,224,3], размер входа
        lambda_=100.0, - коэффициент для потерь
        lambda_identity=1.5, - коэффициент потерь идентичности
        latent_dim = 256, - размер латентного слоя
        n_latent=8, n_dec=32, - параметр латентного слоя, параметр декодера
        mod_ = 'mod', - использовать предобученную сеть с именем из mod 
        deep =  3
        outlevel='block_5_expand_relu' - выходной слой претренированной сети
        '''           

        super(Autoencoder_unet_m, self).__init__()

        #input_enc = keras.layers.Input(shape=(input_shape_im[0], input_shape_im[1], input_shape_im[2]),
        #                               name="auto_input")


        self.latent_dim = latent_dim
        #
        x = keras.layers.Input(shape = input_shape_im)
        input_enc = x
        List_back = []
        n4 = 32
        for jk in range(nk):

            x = downsample(x,
                         filters= n4,#2 ** (jk ) * n4,
                         activation=layers.LeakyReLU(0.2),
                         kernel_size=(3, 3),
                         strides=(2, 2))
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)

            x1 = x
            List_back += [x1]

            x = layers.Conv2D(2**(jk+1) * n4, (3, 3),
                              strides=(1, 1),
                              activation='relu',
                              padding="same", kernel_regularizer='l2', name='transpose_Conv2d_down' + str(jk)[0])(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)

            x = layers.DepthwiseConv2D((2, 2),
                                       strides=(1, 1),
                                       activation='relu',
                                       padding="same", kernel_regularizer='l2', name='transpose_DEPTHW2d' + str(jk)[0])(x)
        
        x = downsample(x,
                         filters= 2 ** (jk ) * n4,
                         activation=layers.LeakyReLU(0.2),
                         kernel_size=(3, 3),
                         strides=(2, 2))
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)

        x = keras.layers.Conv2D(latent_dim,1)(x)

        x = upsample(x,
                         filters= n4 * 2**(nk-jk),
                         activation=layers.LeakyReLU(0.2),
                         kernel_size=(3, 3),
                         strides=(2, 2))
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)

        print(List_back)
        for jk in range(nk):
            x_skip = List_back[nk-jk-1]
            print(x_skip)
            x.shape
            x = upsample(x,
                         filters= n4 * 2**(nk-jk),
                         activation=layers.LeakyReLU(0.2),
                         kernel_size=(4, 4),
                         strides=(2, 2))
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)

            x = layers.Conv2D(n4*(2**(nk-jk)), (2, 2),
                                  strides=(1, 1),
                                  activation='relu',
                                  padding="same", kernel_regularizer ='l2' ,name = 'transpose_Conv2d_up'+str(jk)[0])(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)

            x = layers.DepthwiseConv2D((2, 2),
                                  strides=(1, 1),
                                  activation='relu',
                                  padding="same", kernel_regularizer='l2', name='transpose_DEPTHW2d_up' + str(jk)[0])(x)

        x_out = layers.Conv2D( input_shape_im[2], (1, 1),
                               strides=(1, 1),
                               activation = 'tanh',
                               padding="same",kernel_regularizer ='l2' ,name = 'unet_out')(x)
        print(x_out)
        #self.enc_X.summary()
        #
        self.model_auto = tf.keras.Model( input_enc ,x_out )
        self.model_auto.summary()

        self.lambda_ = lambda_
        self.lambda_identity = lambda_identity
        self.input_shape_im = input_shape_im

        self.model_auto.summary()

    def compile(
            self,
            optimizer,
            loss_fn
    ):

        self.optimizer = optimizer

        self.loss_fn = loss_fn

        self.identity_loss_fn = keras.losses.MeanAbsoluteError()
        self.loss_class = keras.losses.BinaryCrossentropy()
        self.metric_class = keras.metrics.BinaryCrossentropy()
        super(Autoencoder_unet_m, self).compile(optimizer=self.optimizer, loss=loss_fn, metrics=[loss_fn])
        self.model_auto.compile(run_eagerly=True, optimizer=self.optimizer, loss=self.loss_fn, metrics=[self.loss_fn])

    def call(self, x):
        if isinstance(x, tuple):
            x = x[0]
            print('call:', x.shape)
            x = self.model_auto(x)
            return x

    def train_step(self, batch_data):
        # x desktop and y mobile
        if isinstance(batch_data, tuple):
            real_x_real_y, y_label = batch_data
            real_x, real_y = real_x_real_y

            if real_x.shape[0] > 0:

                with tf.GradientTape(persistent=True) as tape:
                    # desk to mobile
                    ind_ = np.where(y_label.numpy() == 0)[0]
                    #print(ind_)
                    if len(ind_) > 0:  # 0
                        real_xp = tf.cast([real_x[ik, :, :, :] for ik in ind_], dtype=tf.float32)
                        real_yp = tf.cast([real_x[ik,:,:,:] for ik in ind_], dtype=tf.float32)
                    else:
                        real_xp = real_x
                        real_yp = real_x

                    fake_y = self.model_auto(real_xp)  # , training=True)
                    if real_xp.shape[0] > 0:
                        if real_xp.shape[-1] > 3:
                            fake_fft_y = generator_auto.fft_y(fake_y[:, :, :, :3])
                        else:
                            fake_fft_y = fake_y
                    else:
                        print('not good')

                        fake_fft_y = fake_y

                    if real_xp.shape[0] > 0:
                        # decoder loss
                        loss_G = self.loss_fn(real_xp[:, :, :, :3], fake_y[:, :, :, :3])

                        if real_xp.shape[-1] > 3:
                            id_loss_G = (self.identity_loss_fn(real_xp[:, :, :, 3:6], fake_fft_y[:, :, :,3:6]))
                        # Total loss
                        id_loss_G = 0.0
                        total_loss = loss_G  # + id_loss_G
                    else:
                        total_loss = self.loss_fn(real_xp[:, :, :, :3], fake_y[:, :, :, :3])
                        loss_G, id_loss_G = 0.0, 0.0

                # Get the gradients
                grads_G = tape.gradient(total_loss, self.model_auto.trainable_variables)

                # Update the weights rs
                self.optimizer.apply_gradients(zip(grads_G, self.model_auto.trainable_variables) )


                return {
                    "loss": total_loss,
                }
            else:
                return {
                    "loss": 1000,
                }
        else:
            return {
                "loss": 1000,
            }

    def test_step(self, batch_data):
        # x desctop - y
        if isinstance(batch_data, tuple):

            real_x_real_y, y_label = batch_data
            real_x, real_y = real_x_real_y
            #print('test', real_x.shape, real_y.shape, y_label.shape)

            #
            # desk to mobile
            ind_ = np.where(y_label.numpy() == 0)[0]
            #print(ind_)
            if len(ind_) > 0:  # 0
                        #print(x.shape)
                        real_xp = tf.cast([real_x[ik, :, :, :] for ik in ind_], dtype=tf.float32)
                        real_yp = tf.cast([real_x[ik,:,:,:] for ik in ind_], dtype=tf.float32)
                        #print(': - : ',real_xp.shape,real_yp.shape)
            else:
                        real_xp = real_x
                        real_yp = real_x
           
            fake_y = self.model_auto(real_xp, training=True)

            if fake_y.shape[0] > 0:
                #print('good')
                if real_x.shape[-1] > 3:
                    fake_fft_y = generator_auto.fft_y(fake_y)
                else:
                    fake_fft_y = fake_y

            else:
                #print('not good')
                fake_fft_y = fake_y
            #  output
            # decoder loss
            if fake_y.shape[0] > 0:
                # decoder loss
                loss_G = self.loss_fn(real_xp[:, :, :, :3], fake_y[:, :, :, :3]) * self.lambda_

                # identity loss

                if real_y.shape[-1] > 3:
                    id_loss_G = (self.identity_loss_fn(real_xp[:, :, :, 3:6],
                                                       fake_fft_y[:, :, :, 3:6]) * self.lambda_ * self.lambda_identity)
                else:
                    id_loss_G = 0.0

                # Total loss
                total_loss = loss_G  # + id_loss_G

            else:
                total_loss = self.loss_fn(real_xp[:, :, :, :3], fake_y[:, :, :, :3]) * self.lambda_
                loss_G, id_loss_G = 0.0, 0.0

            return {
                "loss": total_loss,
            }
        else:
            return {
                "loss": 1000,
            }

            
            
# merquri backborne from Alina

class Swish(tf.keras.layers.Layer):
    #     """
    #     Swish activation function from 'Searching for Activation Functions,' https://arxiv.org/abs/1710.05941.
    #     """

    def call(self, x, training=False):
        return x * tf.nn.sigmoid(x)

    # model = mobi2.MobileNetV2_swish(input_shape=(224,224,3),)
#base_model = tf.keras.models.load_model('full.best.0600-0.0831-0.9774.hdf5', custom_objects={"Swish": Swish})

# pure classifier net
class Classifer(tf.keras.Model):
    def model_parametr(self,model_name):
        with open(model_name+'.txt','w') as f:
            f.write(str(self.latent_dim)+'\n')
            f.write(self.backborne+'\n')
            f.write(self.outlevel+'\n')
            f.write(str(self.input_shape_im[0])+'\n')
            f.write(str(self.input_shape_im[1])+'\n')
            f.write(str(self.input_shape_im[2])+'\n')
            f.close()


    def __init__(self, latent_dim = 128, input_shape_im = [ 224, 224, 3], model_name = '', backborne = 'mod', outlevel = 'block_6_expand_relu',lr =0.000001):
        print('init')
        super(Classifer, self).__init__()

        self.latent_dim = latent_dim
        self.backborne = backborne
        self.outlevel = outlevel
        self.input_shape_im = input_shape_im
        self.optimizer = keras.optimizers.Adam(lr)
        self.loss_fn = keras.losses.BinaryCrossentropy( from_logits=False)

        if backborne == 'mod':
            base_classif = keras.applications.mobilenet_v2.MobileNetV2(
                input_shape=(input_shape_im[0], input_shape_im[1], input_shape_im[2]),
                alpha=1.0, include_top=False, weights='imagenet')
            x_base_end = base_classif.get_layer(self.outlevel).output  # 3-56, 13-14, 6 - 28
            self.enc_X = keras.Model(base_classif.input, x_base_end)
            base_classif.trainable = False

        else:
            self.enc_X = encoder_X
        self.enc_X.summary()

        self.latent_dim = latent_dim
        #
        input_enc = self.enc_X.layers[0].input
        x_enc_end = self.enc_X.layers[-1].output#(input_enc)
        x  = keras.layers.Conv2D(self.latent_dim * 4,(3,3),activation = 'relu',kernel_regularizer ='l2' , padding = 'same')(x_enc_end )
        x = keras.layers.Conv2D(self.latent_dim * 4, (3, 3), activation='relu',kernel_regularizer ='l2' , padding = 'same')(x)
        x = keras.layers.Conv2D(self.latent_dim * 4, (3, 3), activation='relu',kernel_regularizer ='l2' , padding = 'same')(x)
        x = keras.layers.BatchNormalization()(x)
        x_latent_enc = x
        
        x_c = layers.Conv2D(latent_dim * 4 // (2 ** 3), (2, 2),
                             strides=(1, 1),
                             activation='relu',
                             padding="same", kernel_regularizer='l2', name='transpose_Conv2d_latent')(x_latent_enc)

        x_c = keras.layers.Flatten()(x_c)
        x_c = keras.layers.Dense(self.latent_dim * 4, activation='relu', kernel_regularizer='l2',
                                  name='rez_class_dense_1')(x_c)
        x_c = keras.layers.BatchNormalization()(x_c)
        x_c = keras.layers.Dropout(0.2)(x_c)

        x_c = keras.layers.Dense(self.latent_dim, activation='relu', kernel_regularizer='l2',
                                  name='rez_class_dense_3')(x_c)
        x_c = keras.layers.BatchNormalization()(x_c)
        x_c = keras.layers.Dropout(0.2)(x_c)

        x_c = keras.layers.Dense(1, activation='sigmoid', kernel_regularizer='l2', name='rez_class_out')(x_c)
        self.classificator = keras.Model(input_enc, x_c)

        self.classificator.summary()

    def call(self, data):
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data
        return self.classificator(x)

    def compile(self, optimizer=tf.keras.optimizers.Adam()):
        super(Classifer ,self).compile()
        self.classificator.compile(optimizer = optimizer,loss = keras.losses.BinaryCrossentropy(from_logits = False ),metrics=['accuracy'])



    def train_step(self, data):
        try:
            loss_vae = 1000
            loss_classif = 1
            xy, label = data
            x = xy[0]

            if label.shape[0] is not None:
                try:
                    self.classificator.trainable = True
                    with tf.GradientTape() as tape:
                        label_pred = self.classificator(x)
                        loss_classif = self.classificator.loss(tf.reshape(label,[-1,1]), label_pred)
                        grad = tape.gradient(loss_classif, self.classificator.trainable_variables)
                    self.optimizer.apply_gradients(zip(grad, self.classificator.trainable_variables))
                except:
                    print('error1')
                    loss_vae = 1000
                    loss_classif = 1
            else:
                loss_vae = 1000
                loss_classif = 1
        except:
            print('error2')
        return {"loss": loss_vae, "loss_clf": loss_classif}

    def test_step(self, data):
        loss_vae = 1000
        loss_classif = 1
        try:

            xy, label = data
            x = xy[0]

            if 1:  # x.shape[0] is not None:
                try:
                    label_pred = self.classificator(x)

                    loss_classif = self.classificator.loss(tf.reshape(label,[-1,1]), label_pred)/x.shape[0]
                except:
                    print('error')
                    loss_vae = 1000
                    loss_classif = 1
            else:
                loss_vae = 1000
                loss_classif = 1
        except:
            print('error3')
        return {"loss": loss_vae, "loss_clf": loss_classif}

class Autoencoder_unet_class(keras.Model):
    def __init__(
            self,
            encoder_X = None,
            input_shape_im=[224, 224, 3],
            lambda_=100.0,
            lambda_identity=1.5,
            latent_dim=256,
            mod_='../CycleGAN/full.0200-0.0933-0.9710.hdf5',
            outlevel='block_6_expand_swish'
    ):
                
        '''
        encoder_X - энкодер
        input_shape_im = [224,224,3], размер входа
        lambda_=100.0, - коэффициент для потерь
        lambda_identity=1.5, - коэффициент потерь идентичности
        latent_dim = 256, - размер латентного слоя
        n_latent=8, n_dec=32, - параметр латентного слоя, параметр декодера
        mod_ = 'mod', - использовать предобученную сеть с именем из mod 
        deep =  3
        outlevel='block_5_expand_relu' - выходной слой претренированной сети
        '''           

        super(Autoencoder_unet_class, self).__init__()

        self.outlevel = outlevel
        #input_enc = keras.layers.Input(shape=(input_shape_im[0], input_shape_im[1], input_shape_im[2]),
        #                               name="auto_input")


        if mod_ == 'mod':
            self.backborne = 'MobileNetV2'
            base_classif = keras.applications.mobilenet_v2.MobileNetV2(
                input_shape=(input_shape_im[0], input_shape_im[1], input_shape_im[2]),
                alpha=1.0, include_top=False, weights='imagenet')
            #base_classif.summary()
            x_base_end = base_classif.get_layer(self.outlevel).output  # 3-56, 13-14, 6 - 28
            self.enc_X = keras.Model(base_classif.input, x_base_end)
            base_classif.trainable = False

            # x_base = down_model(input_enc)
            List_back = [#self.enc_X.get_layer('block_16_expand_relu').output,  # 7
                     #self.enc_X.get_layer('block_13_expand_relu').output,  # 14
                     'block_6_expand_relu',  # 28
                     'block_3_expand_relu',  # 56
                     'block_1_expand_relu']  # 112
        else:
            self.backborne = mod_
            # backborne = 'full.best.0600-0.0831-0.9774.hdf5', outlevel = 'out_swish'
            base_classif = tf.keras.models.load_model(self.backborne, custom_objects={"Swish": Swish})
            base_classif.summary()
            # base_classif = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(input_shape_im[0], input_shape_im[1], input_shape_im[2]), alpha=1.0, include_top=False, weights='imagenet')
            # x_base_end = base_classif.get_layer('block_3_expand_relu').output  # 1-112 3-56 6-28 13 - 14
            x_base_end = base_classif.get_layer(self.outlevel).output
            self.enc_X = keras.Model(base_classif.input, x_base_end)
            #self.enc_X = encoder_X
            List_back = [#self.enc_X.get_layer('block_16_expand').output,  # 7
                     #self.enc_X.get_layer('block_13_expand').output,  # 14
                     'block_6_expand_swish',  # 28
                     'block_3_expand_swish',  # 56
                     'block_1_expand_swish']  # 112

        self.enc_X.summary()

        self.latent_dim = latent_dim
        #
        #end_enc = self.enc_X.get_layer(self.outlevel).output
        input_enc = self.enc_X.layers[0].input
        x_enc_end = self.enc_X.layers[-1].output#(input_enc)
        
        x  = keras.layers.Conv2D(self.latent_dim * 4,(3,3),activation = 'relu',kernel_regularizer ='l2' , padding = 'same')(x_enc_end )
        x = keras.layers.Conv2D(self.latent_dim * 4, (3, 3), activation='relu',kernel_regularizer ='l2' , padding = 'same')(x)
        x = keras.layers.Conv2D(self.latent_dim * 4, (3, 3), activation='relu',kernel_regularizer ='l2' , padding = 'same')(x)
        x = keras.layers.BatchNormalization()(x)
        x_latent_enc = x
        for jk,x_block in enumerate(List_back):
            x_skip = self.enc_X.get_layer(x_block).output
            print(x_skip)
            x = keras.layers.Concatenate(axis=-1)([x,x_skip ])
            x.shape
            x = upsample(x,
                         filters= self.latent_dim * 4 //(2**jk),
                         activation=layers.LeakyReLU(0.2),
                         kernel_size=(4, 4),
                         strides=(2, 2))
            x = layers.Conv2D(latent_dim * 4 //(2**jk), (2, 2),
                                  strides=(1, 1),
                                  activation='relu',
                                  padding="same", kernel_regularizer ='l2' ,name = 'transpose_Conv2d'+str(jk)[0])(x)

        x_out = layers.Conv2D( input_shape_im[2], (1, 1),
                               strides=(1, 1),
                               activation = 'tanh',
                               padding="same",kernel_regularizer ='l2' ,name = 'unet_out')(x)
        print(x_out)
        #
        print(input_enc)
        #
        self.model_auto = tf.keras.Model( input_enc ,x_out )
        self.model_auto.summary()

        x_rez = keras.layers.Subtract( name = 'substr_input-output_unet')([input_enc,x_out])
        x_c2 = keras.layers.BatchNormalization()(x_rez)

        num_filters = 32

        for num_downsample_block in range(3):

            num_filters *= 2
            x_c2 = downsample(
                x_c2,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
                )
            x_c2 = layers.Conv2D(latent_dim * 4 //(2**jk), (2, 2),
                                  strides=(1, 1),
                                  activation='relu',
                                  padding="same", kernel_regularizer ='l2' ,name = 'transpose_Conv2d'+str(num_filters )[:2])(x_c2)

        x_c2 = keras.layers.Flatten()(x_c2)
        x_c2 = keras.layers.Dense(self.latent_dim *4 , activation='relu',kernel_regularizer ='l2' ,name = 'rez_class_dense_1')(x_c2)
        x_c2 = keras.layers.BatchNormalization()(x_c2)
        x_c2 = keras.layers.Dropout(0.2)(x_c2)

        x_c2 = keras.layers.Dense(self.latent_dim *2, activation='relu',name = 'rez_class_dense_2')(x_c2)
        x_c2 = keras.layers.BatchNormalization()(x_c2)
        x_c2 = keras.layers.Dropout(0.2)(x_c2)

        x_c2 = keras.layers.Dense(self.latent_dim , activation='relu',kernel_regularizer ='l2' ,name = 'rez_class_dense_3')(x_c2)
        x_c2 = keras.layers.BatchNormalization()(x_c2)
        x_c2 = keras.layers.Dropout(0.2)(x_c2)

        x_c2 = keras.layers.Dense(1, activation='sigmoid',kernel_regularizer ='l2' ,name = 'rez_class_out')(x_c2)
        self.classificator_rez = keras.Model(input_enc,x_c2)

        x_c = base_classif.output

        self.classificator = keras.Model(input_enc, x_c)

        self.lambda_ = lambda_
        self.lambda_identity = lambda_identity
        self.input_shape_im = input_shape_im

        ##
        self.model_auto.summary()
        self.classificator.summary()

    def compile(
            self,
            optimizer,
            loss_fn
    ):

        self.optimizer = optimizer

        self.loss_fn = loss_fn

        self.identity_loss_fn = keras.losses.MeanAbsoluteError()
        self.loss_class = keras.losses.BinaryCrossentropy()
        self.metric_class = keras.metrics.BinaryCrossentropy()
        super(Autoencoder_unet_class, self).compile(optimizer=self.optimizer, loss=loss_fn, metrics=[loss_fn])
        self.model_auto.compile(run_eagerly=True, optimizer=self.optimizer, loss=self.loss_fn, metrics=[self.loss_fn])
        self.classificator.compile(run_eagerly=True, optimizer=self.optimizer, loss=self.loss_class,
                                   metrics=[self.metric_class])
        self.classificator_rez.compile(run_eagerly=True, optimizer=self.optimizer, loss=self.loss_class,
                                   metrics=[self.metric_class])
    def call(self, x):
        if isinstance(x, tuple):
            x = x[0]
            print('call:', x.shape)
            x = self.model_auto(x)
            return x

    def train_step(self, batch_data):
        # x desktop and y mobile
        if isinstance(batch_data, tuple):
            real_x_real_y, y_label = batch_data
            real_x, real_y = real_x_real_y

            # print(": : ",real_x.shape,real_y.shape)
            if real_x.shape[0] > 0:
                self.enc_X.trainable = False
                with tf.GradientTape(persistent=True) as tape:
                    # desk to mobile
                    ind_ = np.where(y_label.numpy() == 0)[0]
                    #print(ind_)
                    if len(ind_) > 0:  # 0
                        #print(x.shape)
                        real_xp = tf.cast([real_x[ik, :, :, :] for ik in ind_], dtype=tf.float32)
                        real_yp = tf.cast([real_x[ik,:,:,:] for ik in ind_], dtype=tf.float32)
                        #print(': - : ',real_xp.shape,real_yp.shape)
                    else:
                        real_xp = real_x
                        real_yp = real_x

                    fake_y = self.model_auto(real_xp)  # , training=True)
                    if real_xp.shape[0] > 0:
                        if real_xp.shape[-1] > 3:
                            fake_fft_y = generator_auto.fft_y(fake_y[:, :, :, :3])
                        else:
                            fake_fft_y = fake_y
                    else:
                        print('not good')

                        fake_fft_y = fake_y

                    if real_xp.shape[0] > 0:
                        # decoder loss
                        loss_G = self.loss_fn(real_xp[:, :, :, :3], fake_y[:, :, :, :3])

                        if real_xp.shape[-1] > 3:
                            id_loss_G = (self.identity_loss_fn(real_xp[:, :, :, 3:6], fake_fft_y[:, :, :,3:6]))
                        # Total loss
                        id_loss_G = 0.0
                        total_loss = loss_G  # + id_loss_G
                    else:
                        total_loss = self.loss_fn(real_xp[:, :, :, :3], fake_y[:, :, :, :3])
                        loss_G, id_loss_G = 0.0, 0.0

                # Get the gradients
                grads_G = tape.gradient(total_loss, self.model_auto.trainable_variables)

                # Update the weights rs
                self.optimizer.apply_gradients(zip(grads_G, self.model_auto.trainable_variables) )

                self.enc_X.trainable = False
                #self.dec_X.tarinable = False
                #self.latent_X.trainable = False

                with tf.GradientTape() as tape:

                    y_pred = self.classificator(real_x)
                    y_loss = self.loss_class(y_label, y_pred)
                    #print(y_loss)
                    y_grad = tape.gradient(y_loss, self.classificator.trainable_variables)
                self.optimizer.apply_gradients(zip(y_grad, self.classificator.trainable_variables))

                y_loss2=0.0
                self.enc_X.trainable = True
                #self.dec_X.tarinable = True
                #self.latent_X.trainable = True

                return {
                    "loss": total_loss,
                    "loss_clf": y_loss,
                    "loss_cls_rez":y_loss2,
                }
            else:
                return {
                    "loss": 1000,
                    "loss_clf": 1,
                    "loss_cls_rez": 1,
                }
        else:
            return {
                "loss": 1000,
                "loss_clf": 1,
                "loss_cls_rez": 1,
            }

    def test_step(self, batch_data):
        # x desctop - y
        # print('test',batch_data[0].shape, batch_data[1].shape)
        if isinstance(batch_data, tuple):

            real_x_real_y, y_label = batch_data
            real_x, real_y = real_x_real_y
            #print('test', real_x.shape, real_y.shape, y_label.shape)

            #
            fake_y = self.model_auto(real_x, training=True)

            if fake_y.shape[0] > 0:
                #print('good')
                # <<<<<<< HEAD
                if real_x.shape[-1] > 3:
                    fake_fft_y = generator_auto.fft_y(fake_y)
                else:
                    fake_fft_y = fake_y

            else:
                #print('not good')
                fake_fft_y = fake_y
            #  output
            # decoder loss
            if fake_y.shape[0] > 0:
                # decoder loss
                loss_G = self.loss_fn(real_x[:, :, :, :3], fake_y[:, :, :, :3]) * self.lambda_

                # identity loss

                if real_y.shape[-1] > 3:
                    id_loss_G = (self.identity_loss_fn(real_x[:, :, :, 3:6],
                                                       fake_fft_y[:, :, :, 3:6]) * self.lambda_ * self.lambda_identity)
                else:
                    id_loss_G = 0.0

                # Total loss
                total_loss = loss_G  # + id_loss_G

            else:
                total_loss = self.loss_fn(real_x[:, :, :, :3], fake_y[:, :, :, :3]) * self.lambda_
                loss_G, id_loss_G = 0.0, 0.0

            y_pred = self.classificator(real_x)
            y_loss = self.loss_class(y_label, y_pred)
            #y_pred2 = self.classificator_rez(real_x)
            #y_loss2 = self.loss_class(y_label, y_pred2)
            y_loss2=0
            return {
                "loss": total_loss,
                "loss_clf": y_loss,
                "loss_cls_rez": y_loss2,
            }
        else:
            return {
                "loss": 1000,
                "loss_clf": 1,
                "loss_cls_rez": 1,
            }








class CustomSaveCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))



# метрика
def calc_metrics(targets_scores, imposter_scores):
    """
    targets_scores - значения выхода модели через метрику для пар,
    imposter_scores - значения выхода модели через метрику для не пар,
    """
    # диапазоны
    min_score = np.minimum(np.min(targets_scores), np.min(imposter_scores))
    max_score = np.maximum(np.max(targets_scores), np.max(imposter_scores))
    #print(min_score,max_score)
    n_tars = len(targets_scores)
    n_imps = len(imposter_scores)

    N = 100

    fars = np.zeros((N,))
    frrs = np.zeros((N,))
    dists = np.zeros((N,))

    mink = float('inf')
    eer = 0
    min_i = 0
    # поиск оптимальной точки
    for i, dist in enumerate(np.linspace(min_score, max_score, N)):
        far = len(np.where(imposter_scores > dist)[0]) / n_imps
        frr = len(np.where(targets_scores < dist)[0]) / n_tars
        # добавили оценки по этому порогу для dist
        fars[i] = far
        frrs[i] = frr
        dists[i] = dist

        k = np.abs(far - frr)

        if k < mink:
            mink = k
            min_i = i
            eer = (far + frr) / 2
    # accuracy

    if n_tars>n_imps:
      nn = n_imps
    else:
      nn = n_tars
    TN = np.sum(imposter_scores[:nn] > dists[min_i])
    TP = np.sum(targets_scores[:nn] < dists[min_i])
    acc = (TP+TN)/(nn*2)


    return eer, fars, frrs, dists, min_i, acc



'''
Пример обращения
# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


enc_ = get_encoder(
    latent_dim = [16,16,32],
    filters=64, kernel_initializer=kernel_init,  name='encoder',
    input_img_size = [input_img_size[0],input_img_size[1],3])
enc_.summary()  


dec_ =  get_decoder( latent_dim = [28,28,32],
    filters=64, kernel_initializer=kernel_init, name='decod',
    input_img_size = [input_img_size[0],input_img_size[1],3])
dec_.summary()   

lat_ = get_latent(
    latent_dim1 = [28,28,32],
    latent_dim2 = [28,28,32],
    latent_dim = 128,
    filters=64,
    
    gamma_initializer=gamma_init,
    name='latent_FC',
    )
lat_.summary()


model_ = Autoencoder_v(dec_,lat_,enc_)
model_.summary()
'''
