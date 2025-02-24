##############################################################
#            Synthetic Dataset Generators
#
# Data Characteristics -- Multivariate
# Tasks -- Binary classification
# 
# For an overview of XAI methods, we refer to:
# 10.1109/ACCESS.2024.3409843
# 10.48550/arXiv.2404.16495
#
##############################################################

import math
import numpy as np
import pandas as pd


##############################################################
def data_generator_4ft(n=10, df=False):
    """
    Synthetic generator -- all Feature Types are Real
    n is the number of instances in each class

    RETURNS: a 4-features binary synthetic dataset -- 2 core features and 2 random noise features
    """
    n= int(n)

    mean_n1= np.array([-2,2.0])
    mean_n2= np.array([0.0,0.0])

    D_n1= np.array([[1.0,0.0],[0.0,1.0]]).T
    D_n2= np.array([[1.0,1.0],[-1.0,1.0]]).T

    S_n1= np.diag([2.0,1.0])
    S_n2= np.diag([4.0,0.1])

    C_n1= np.linalg.multi_dot([D_n1,S_n1,D_n1.T])
    C_n2= np.linalg.multi_dot([D_n2,S_n2,D_n2.T])

    X1= np.random.multivariate_normal(mean_n1,C_n1,size=(n))
    X2= np.random.multivariate_normal(mean_n2,C_n2,size=(n))

    X= np.zeros((X1.shape[0] + X2.shape[0],5))
    
    X[:X1.shape[0],:2]= X1
    X[:X1.shape[0],2:-1]= np.random.uniform(-4,4,size=(X1.shape[0],2))
    X[:X1.shape[0],-1]= 1
    X[X1.shape[0]:X1.shape[0] + X2.shape[0],:2]= X2
    X[X1.shape[0]:X1.shape[0] + X2.shape[0],2:-1]= np.random.uniform(-4,4,size=(X2.shape[0],2))
    X= X[:(n * 2),:]
    
    if (df is True):
        columns_syn= ['core_1','core_2','noise_1','noise_2','target']
        X_df= pd.DataFrame(data= X, columns= columns_syn)
        
        return X_df
    
    return X


##############################################################
def data_generator_1cat2_3ft(n=10, p=0.95, df=False):
    """
    Synthetic generator -- Mixing Real and Categorical data
    n is the number of instances in each class
    p is the probability threshold of correct placement on categorical variables

    RETURNS: a 3-features binary synthetic dataset -- 1 core (2 categorical values one-hot-encoded 
             style) and 2 random noise features
    """
    n= int(n)
    
    X1= np.zeros((n, 2))
    X2= np.zeros((n, 2))
    
    for i in range(n):
        rp= np.random.random_sample()

        if (rp <= p):
            X1[i][0]= 1
        else:
            X1[i][1]= 1
            
        rp= np.random.random_sample()
        
        if (rp <= p):
            X2[i][1]= 1
        else:
            X2[i][0]= 1
            

    X= np.zeros((X1.shape[0] + X2.shape[0],5))
    
    X[:X1.shape[0],:2]= X1
    X[:X1.shape[0],-1]= 1
    X[X1.shape[0]:X1.shape[0] + X2.shape[0],:2]= X2
    X[:,2:-1]= np.random.uniform(-4,4,size=(X1.shape[0] + X2.shape[0],2))
    X= X[:(n * 2),:]
    
    if (df is True):
        columns_syn= ['cat_core_1','cat_core_2','noise_1','noise_2','target']
        X_df= pd.DataFrame(data= X, columns= columns_syn)
        
        return X_df
    
    return X


##############################################################
def data_generator_1cat3_3ft(n=10, p=0.95, df=False):
    """
    Synthetic generator -- Mixing Real and Categorical data
    n is the number of instances in each class
    p is the probability threshold of correct placement on categorical variables

    RETURNS: a 3-features binary synthetic dataset -- 1 core (3 categorical values one-hot-encoded 
             style) and 2 random noise features
    """
    n= int(n)
    
    X1= np.zeros((math.ceil(n / 2), 3))
    X2= np.zeros((n, 3))
    X3= np.zeros((math.ceil(n / 2), 3))
    
    for i in range(n):
        if (i < math.ceil(n / 2)):
            rp= np.random.random_sample()

            if (rp <= p):
                X1[i][0]= 1
            elif (rp < (p + (1 - p) / 2)):
                X1[i][1]= 1
            else:
                X1[i][2]= 1
                
            rp= np.random.random_sample()
        
            if (rp <= p):
                X3[i][2]= 1
            elif (rp < (p + (1 - p) / 2)):
                X3[i][0]= 1
            else:
                X3[i][1]= 1
            
        rp= np.random.random_sample()
        
        if (rp <= p):
            X2[i][1]= 1
        elif (rp < (p + (1 - p) / 2)):
            X2[i][0]= 1
        else:
            X2[i][2]= 1
            

    X= np.zeros((X1.shape[0] + X2.shape[0] + X3.shape[0],6))
    
    X[:X1.shape[0],:3]= X1
    X[:X1.shape[0],-1]= 1
    X[X1.shape[0]:X1.shape[0] + X2.shape[0],:3]= X2
    X[X1.shape[0] + X2.shape[0]:X1.shape[0] + X2.shape[0] + X3.shape[0],:3]= X3
    X[X1.shape[0] + X2.shape[0]:X1.shape[0] + X2.shape[0] + X3.shape[0],-1]= 1
    X[:,3:-1]= np.random.uniform(-4,4,size=(X1.shape[0] + X2.shape[0] + X3.shape[0],2))
    X= X[:(n * 2),:]
    
    if (df is True):
        columns_syn= ['cat_core_1','cat_core_2','cat_core_3','noise_1','noise_2','target']
        X_df= pd.DataFrame(data= X, columns= columns_syn)
        
        return X_df
    
    return X


##############################################################
def data_generator_1cat2_5ft(n=10, p=0.95, df=False):
    """
    Synthetic generator -- Mixing Real and Categorical data
    n is the number of instances in each class
    p is the probability threshold of correct placement on categorical variables

    RETURNS: a 5-features binary synthetic dataset -- 3 core (2 categorical values one-hot-encoded 
             style and 2 numerical) and 2 random noise features
    """
    cat_data= data_generator_1cat2_3ft(n=n, p=p, df=True)
    num_data= data_generator_4ft(n=n, df=True)
    
    cat_columns= ['cat_core_1','cat_core_2']
    
    cat_num_6ft= pd.concat([cat_data[cat_columns], num_data], axis=1)

    if not (df):
        return np.asarray(cat_num_6ft)
    
    return cat_num_6ft


##############################################################
def data_generator_1cat3_5ft(n=10, p=0.95, df=False):
    """
    Synthetic generator -- Mixing Real and Categorical data
    n is the number of instances in each class
    p is the probability threshold of correct placement on categorical variables

    RETURNS: a 5-features binary synthetic dataset -- 3 core (3 categorical values one-hot-encoded 
             style and 2 numerical) and 2 random noise features
    """
    cat_data= data_generator_1cat3_3ft(n=n, p=p, df=False)
    cat_data1= cat_data[:math.ceil(n / 2),:3]
    cat_data2= cat_data[math.ceil(n / 2):(math.ceil(n / 2) + n),:3]
    cat_data3= cat_data[(math.ceil(n / 2) + n):,:3]
    
    cat_data= np.concatenate((cat_data1, cat_data3), axis=0)
    cat_data= np.concatenate((cat_data, cat_data2), axis=0)
    
    cat_columns= ['cat_core_1','cat_core_2','cat_core_3']
    cat_data= pd.DataFrame(data=cat_data, columns=cat_columns)
    
    num_data= data_generator_4ft(n=n, df=True)
    
    cat_num_7ft= pd.concat([cat_data, num_data], axis=1)

    if not (df):
        return np.asarray(cat_num_7ft)
    
    return cat_num_7ft
