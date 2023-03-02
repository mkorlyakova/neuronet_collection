
import os
import pandas as pd


l_dir = os.listdir('mobile')
print(l_dir)
df = pd.DataFrame(l_dir, columns = ['dir'])
df['tag'] = 1
print(df)
df.to_csv('mobile_train.csv')

l_dir = os.listdir('mobile')
print(l_dir)
df = pd.DataFrame(l_dir, columns = ['dir'])
df['tag'] = 1
print(df)
df.to_csv('mobile_test.csv')

l_dir = os.listdir('desktop')
print(l_dir)
df = pd.DataFrame(l_dir, columns = ['dir'])
df['tag'] = 1
print(df)
df.to_csv('desktop_train.csv')
l_dir = os.listdir('desktop')
print(l_dir)
df = pd.DataFrame(l_dir, columns = ['dir'])
df['tag'] = 1
print(df)
df.to_csv('desktop_test.csv')


