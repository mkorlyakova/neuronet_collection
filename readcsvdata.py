import pandas as pd



def read_data(path_name='',without_=2):

    file_mobile = ''
    file_mobilet = ''
    file_desktop = ''
    file_desktopt = ''



    n_test1,n_test2 = 0,1010

    n_train1,n_train2 = 0,1000

    
    df1 = pd.read_csv(path_name+'mobile_train.csv')
    df2 = pd.read_csv(path_name+'desktop_train.csv')
    #print(df1.head())
    #print(df2.head())
    df3 = pd.read_csv(path_name+'mobile_test.csv')
    df4 = pd.read_csv(path_name+'desktop_test.csv')
     
    #print("bfore df1:",df1.shape)
    df1 = df1.loc[df1.iloc[:,1] != without_,:]
    df2 = df2.loc[df2.iloc[:,1] != without_,:]
    df3 = df3.loc[df3.iloc[:,1] != without_,:]
    df4 = df4.loc[df4.iloc[:,1] != without_,:]
    #print("after df1:",df1.shape)


    list_mob = df1.iloc[:,0].values.tolist()
    list_mobt = df3.iloc[:,0].values.tolist()
    list_desk = df2.iloc[:,0].values.tolist()
    list_deskt = df4.iloc[:,0].values.tolist()

    tag_mob = df1.iloc[:,1].values.tolist()
    tag_mobt = df3.iloc[:,1].values.tolist()
    tag_desk = df2.iloc[:,1].values.tolist()
    tag_deskt = df4.iloc[:,1].values.tolist()

    if len(list_mob)>len(list_desk):
        n_train2 = len(list_desk)
    else:
        n_train2 = len(list_mob)

    return   list_mob,list_mobt , list_desk,list_deskt  ,n_test1,n_test2, n_train1,n_train2, tag_mob, tag_mobt, tag_desk, tag_deskt
