## test net

from tensorflow import keras

import tensorflow as tf

import numpy as np

import pandas as pd
import os


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


from readcsvdata import read_data
import auto_enc
from generator_auto import CustomDataGenFace_dm_label_image

from init_confir import config_model, tostring_model_name, fromstring_model_name


gpu_ =0
fft_ = True
image_ = False
print(tf.__version__)
# текущие каталоги куда запишем промежуточные результаты
file_desktopt, file_mobilet, file_desktop, file_mobile, file_name, cwd, path_label, path_image, \
numb_step, n_train, n_test, epoch, \
filters_disc, down_bloc_disc, up_bloc_gen, down_bloc_gen, rez_bloc_gen,\
gpu_, SIZE, load_models, batch_size_model, lr_model, \
model_name, model_path, file_name, desctop_train, mobile_train, desctop_test, mobile_test = config_model(gpu_)

print(down_bloc_disc)

m1=n_test # batch_size тест

m2= numb_step # train+test subset number
m3=1 # train subset number
list_mob=[]
list_mobt = []
list_desk = []
list_deskt=[]
n_test1 = 0
n_test2 = 10 
n_train1,n_train2 = 0,10
gpu_ = 0# test from local computer
if gpu_ == 0:
    # test from local computer
    # file_mobile  -  path to class 1
    # file_desktop - path to class 0
    # print(os.path.join(file_name,file_mobile) )
    batch_size_model = 2
    file_name =  ''
    list_mob = os.listdir(os.path.join(file_name, file_mobile))
    list_desk = os.listdir(os.path.join(file_name, file_desktop))
    list_mobt = os.listdir(os.path.join(file_name, file_mobilet))
    list_deskt = os.listdir(os.path.join(file_name, file_desktopt))
    tm = np.random.randint(0, 2, (len(list_mob)))
    tmt = np.random.randint(0, 2, (len(list_mobt)))
    td = np.random.randint(0, 2, (len(list_desk)))
    tdt = np.random.randint(0, 2, (len(list_deskt)))

if gpu_:
    # test from server computer
    list_mob, list_mobt, list_desk, list_deskt, n_test1, n_test2, n_train1, n_train2, tm, tmt, td, tdt = read_data(
        path_name=file_name)

N_split = 3
lab = 1
num_filter = 1
num_up=1

model_n = input('enter model name:')
#model_n = 'model/auto3f__128_224_64_0_3_0_64_3class_X.h5'
try:
    num_disc, up_bloc_gen ,down_bloc_gen, num_residual_blocks , filters_gen = fromstring_model_name(model_n = model_n , shift = 0)
except:
    num_disc, up_bloc_gen, down_bloc_gen, num_residual_blocks, filters_gen = 2,2,2,0,8
print('SIZE:',SIZE, filters_gen,up_bloc_gen)

kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


fft_ = True
if fft_&image_:
  n_chanel = 9
else :
  n_chanel = 3
latent_dim = 128
# Data loader
data_gen = CustomDataGenFace_dm_label_image(tm=tmt,td=tdt,
                                            file_mobile=list_mobt,file_desk=list_deskt,
                                            path_m=os.path.join(file_name,file_mobilet),
                                            path_d=os.path.join(file_name,file_desktopt),
                                            n_train = n_test,
                                            W = SIZE[0],
                                            input_size =(SIZE[0],SIZE[0],n_chanel) ,
                                            batch_size = batch_size_model,
                                            fft_=fft_, image_=image_, random_ind=False )

hystoryG, hystoryD = [], []
# Get the encoder
enc_ = auto_enc.get_encoder(
    latent_dim=[28, 28, 64],
    filters=filters_disc, kernel_initializer=kernel_init, name='encoder',
    input_img_size=[SIZE[0], SIZE[1], n_chanel])

# Get the decoder
dec_ = auto_enc.get_decoder(latent_dim=[28, 28, 64],
                            filters=filters_gen, kernel_initializer=kernel_init, name='decod',
                            output_img_size=[SIZE[0], SIZE[1], n_chanel])


mode = auto_enc.ClassConditionVAE(latent_dim, enc_, dec_, input_shape_im=[224, 224, 3], n_latent=28, n_dec=64,
                                  model_name=model_n)
mode.compile()

mode_classificator = model_n
try:
    mode.classificator.load_weights(mode_classificator)
    print('weght load')
except:
    pass

# generator test
i1,s1 =data_gen[0]
print(i1[0].shape,i1[1].shape)

if len(list_mobt)>len(list_deskt):
    N1=len(list_deskt) // batch_size_model
else:
    N1 = len(list_mobt) // batch_size_model
    
result_predict = np.array([])
target_ = np.array([])
print(N1)
N = int(input('enter test number ('+str(N1)+'):'))
for nk1 in range(N):
    try:
      if nk1 < N // 2 :
          n = nk1
      else:
          n = N1 - nk1 -1
      i1_dm,label=data_gen[n]

      i1,dm= i1_dm

      prediction = mode.classificator.predict([i1,dm])

      if nk1 == 0:
          target_ =  label.reshape(-1,1)
          result_predict = prediction
      else:
          target_ =np.vstack(   (target_,label.reshape(-1,1)))
          result_predict = np.vstack((result_predict, prediction))

    except:
        print('error')



print(result_predict)
print(result_predict.shape,target_.shape)
result_predict = np.hstack((result_predict,target_))
print(result_predict[0,:])

df = pd.DataFrame(np.array(result_predict),columns=['vae_cond_classif','target'])


df.to_csv('model_00_'+model_n.split(os.path.sep)[-1]+'.csv')


