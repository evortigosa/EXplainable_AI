##############################################################
#           Data (pre)processing methods
# 
# For an overview of XAI methods, we refer to:
# http://dx.doi.org/10.1109/ACCESS.2024.3409843
# http://dx.doi.org/10.48550/arXiv.2404.16495
#
##############################################################

import numpy as np
import pandas as pd


##############################################################
def normalize(x, norm_range=(0, 1)):
    """
    Normalize a numpy array to a specified range.
    x: numpy.ndarray
    norm_range tuple (min, max) with the desired range of normalized data, default=(0, 1)

    RETURNS: a normalized numpy.ndarray
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input x must be a numpy array.")
    
    n_min, n_max= norm_range
    
    max_value= np.max(x)
    min_value= np.min(x)

    if max_value == min_value:
        # return a constant array if all values are the same
        return np.full_like(x, (n_max + n_min) / 2)
    
    normalized= (x- min_value)/ (max_value - min_value)
    normalized= normalized * (n_max - n_min) + n_min
        
    return normalized


##############################################################
def normalize_selected(df, selected_cols=[], norm_range=(0,1)):
    """
    Normalize selected columns of a Pandas DataFrame to a specified range.
    if selected_cols is not defined as argument, all df columns will be normalized
    norm_range tuple (min, max) with the desired range of normalized data, default=(0, 1)

    - df: DataFrame containing data to normalize.
    - selected_cols: list, optional List of column names to normalize. If None, all 
        numeric columns are normalized.
    - norm_range: tuple, optional (min, max) range for normalization, default=(0, 1).

    RETURNS: df with selected_cols normalized
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame.")
    
    result= df.copy()
    n_min, n_max= norm_range
    
    if selected_cols is None:
        # automatically select numeric columns
        selected_cols= df.select_dtypes(include=[np.number]).columns
    
    for col in selected_cols:
        max_value= df[col].max()
        min_value= df[col].min()

        if max_value == min_value:
            # assign mid-range value if all values are the same
            result[col] = (n_max + n_min) / 2
        else:
            result[col]= (df[col]- min_value)/ (max_value - min_value)
            result[col]= result[col] * (n_max - n_min) + n_min
        
    return result


##############################################################
# Scaling after the train-test split is an essential practice to keep the validation phase as 
# realistic as an actual deployment and maintain the integrity of your model's performance metrics. 
# Always remember: scale based on the training set, and then apply those transformations to the test 
# set to maintain the purity of the test environment.
##############################################################
def get_norm_parameters(train):
    """
    Compute normalization parameters for a given dataset.
    train is a pd.DataFrame n-dimensional dataset

    RETURNS: a n-rows np.array with [min,max] values of each feature from train for apply 
             in a normalization procedure
    """
    if not isinstance(train, pd.DataFrame):
        raise TypeError("train must be a pandas DataFrame.")
    
    norm_parameters= []
    
    for col in train.columns:
        max_value= train[col].max()
        min_value= train[col].min()
        
        norm_parameters.append([min_value,max_value])
    
    return np.asarray(norm_parameters)


##############################################################
def normalize_train_test(train, test, norm_parameters=[], selected_cols=[], norm_range=(0,1)):
    """
    Best practice sequence to prevent data leakage: Split, Then Scale
    This method apply the scaling transformation to the training data and scale 
    the test data using the same parameters computed from the training data.

    train and test are pd.DataFrame n-dimensional datasets resulting from a data split 
        (must have the same features)
    norm_parameters is an n-rows np.array with [min,max] values of each feature from train data
    if selected_cols is not defined as argument, all df columns will be normalized
    norm_range tuple (min, max) with the desired range of normalized data, default=(0, 1)

    RETURNS: training data normalized and test data normalized using the parameters from the 
             training data.
    """
    if not isinstance(train, pd.DataFrame) or not isinstance(test, pd.DataFrame):
        raise TypeError("train and test must be pandas DataFrames.")
    
    if not np.array_equal(train.columns, test.columns):
        raise ValueError('Train and test must match!')
        
    if not (len(norm_parameters)== 0):
        if (np.asarray(norm_parameters).shape[0]!= train.shape[1]):
            raise ValueError('norm_parameters and train must match!')
    
    norm_train= train.copy()
    norm_test= test.copy()
    
    n_min, n_max= norm_range
    
    if selected_cols is None:
        # normalize only numeric columns
        selected_cols= train.select_dtypes(include=[np.number]).columns
    
    parameters= True
    if (len(norm_parameters)== 0):
        parameters= False
    
    for col in selected_cols:
        if (parameters):
            min_value, max_value= norm_parameters[train.columns.get_loc(col)]
        else:
            min_value, max_value= train[col].min(), train[col].max()
        
        if max_value == min_value:
            # if all values are the same, set them to mid-range
            norm_train[col]= (n_max + n_min) / 2
            norm_test[col] = (n_max + n_min) / 2
        else:
            norm_train[col]= (train[col]- min_value)/ (max_value - min_value)
            norm_train[col]= norm_train[col] * (n_max - n_min) + n_min
            
            norm_test[col]= (test[col]- min_value)/ (max_value - min_value)
            norm_test[col]= norm_test[col] * (n_max - n_min) + n_min

    return norm_train, norm_test


##############################################################
def get_std_parameters(train):
    """
    Compute standardization parameters (mean, std) for each feature in the training dataset.
    train is a pd.DataFrame n-dimensional dataset

    RETURNS: a n-rows np.array with [mean, std] values of each feature from train for apply 
    in a Standard Scale procedure
    """
    if not isinstance(train, pd.DataFrame):
        raise TypeError("train must be a pandas DataFrame.")
    
    std_parameters= []
    
    for col in train.columns:
        mean_value= train[col].mean()
        std_value = train[col].std(ddof=0)
        
        std_parameters.append([mean_value,std_value])
    
    return np.asarray(std_parameters)


##############################################################
def standarize_train_test(train, test, std_parameters=[], selected_cols=[]):
    """
    Standardize training and test datasets using parameters computed from training data.
    Best practice sequence to prevent data leakage: Split, Then Scale
    This method apply the scaling transformation to the training data and scale 
    the test data using the same parameters computed from the training data.

    train and test are pd.DataFrame n-dimensional datasets resulting from a data split 
        (must have the same features)
    norm_parameters is an n-rows np.array with [mean,std] values of each feature from train data
    if selected_cols is not defined as argument, all df columns will be standarized

    RETURNS: training data standardized and test data standardized using the parameters from the 
             training data.
    """
    if not isinstance(train, pd.DataFrame) or not isinstance(test, pd.DataFrame):
        raise TypeError("train and test must be pandas DataFrames.")
    
    if not np.array_equal(train.columns, test.columns):
        raise ValueError('Train and test must match!')
        
    if not (len(std_parameters)== 0):
        if (np.asarray(std_parameters).shape[0]!= train.shape[1]):
            raise ValueError('std_parameters must match the number of features in train!')
    
    std_train= train.copy()
    std_test= test.copy()

    if selected_cols is None:
        # standardize only numeric columns
        selected_cols = train.select_dtypes(include=[np.number]).columns
    
    parameters= True
    if (len(std_parameters)== 0):
        parameters= False
        
    for col in selected_cols:
        if (parameters):
            mean_value, std_value= std_parameters[train.columns.get_loc(col)]
        else:
            mean_value, std_value= train[col].mean(), train[col].std(ddof=0)
        
        if std_value == 0:
            # avoid division by zero: if std=0, set values to zero (data has no variation)
            std_train[col]= 0
            std_test[col] = 0
        else:
            std_train[col]= (train[col]- mean_value)/ std_value
            std_test[col] = (test[col] - mean_value)/ std_value
    
    return std_train, std_test


##############################################################
def check_null_values(df, text_info=False):
    """
    Check for missing (null) values in a DataFrame.
    df is the DataFrame to check for missing values.
    text_info True returns a textual message about the null checking

    RETURNS: True if there are some missing values in df or False if there are no missing values in df
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    
    null_values= df.isnull().values.any()

    if (text_info and null_values):
        print("There are some missing values in the dataset")
    elif text_info:
        print("There are no missing values in the dataset")

    return null_values


##############################################################
def pre_proc_fillna_num_fts(df, num_cols, num_type='mean'):
    """
    Fill NaN values in numerical columns using mean, median, mode, or zero.
    receive numerical features: fill nan's with median/mean/mode/zeros
    num_cols list indicating the columns names from df are numerical
    num_type defines the fill type - mode/mean/median/zeros

    RETURNS: df without nan's on num_cols
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(num_cols, list):
        raise TypeError("num_cols must be a list of column names.")
    
    df_train= df.copy()

    if (num_type=='median'):
        for col in num_cols:
            ft_median= df_train[col].median()
            df_train[col]= df_train[col].fillna(ft_median)
    elif (num_type=='mode'):
        for col in num_cols:
            ft_mode= df_train[col].value_counts().index[0]
            df_train[col]= df_train[col].fillna(ft_mode)
    elif (num_type=='zeros'):
        for col in num_cols:
            ft_zero= 0
            df_train[col]= df_train[col].fillna(ft_zero)
    else:
        for col in num_cols:
            ft_mean= df_train[col].mean()
            df_train[col]= df_train[col].fillna(ft_mean)

    return df_train


##############################################################
def pre_proc_fillna_num_fts_train_test(train, test, num_cols, num_type='mean'):
    """
    Receive numerical features: fill NaN values with median/mean/mode/zeros
    This method fills NaN values in the training data. NaN values in the test data are filled 
    using the same parameters computed from the training data.

    train and test are pd.DataFrame n-dimensional datasets resulting from a data split 
        (must have the same features)
    num_cols list indicating the columns names from df are numerical
    num_type defines the fill type - mode/mean/median/zeros

    RETURNS: train and test without NaN values in num_cols using the parameters from the training data.
    """
    if not isinstance(train, pd.DataFrame) or not isinstance(test, pd.DataFrame):
        raise TypeError("train and test must be pandas DataFrames.")
    
    if not isinstance(num_cols, list) or not all(isinstance(col, str) for col in num_cols):
        raise TypeError("num_cols must be a list of column names (strings).")
    
    if not np.array_equal(train.columns, test.columns):
        raise ValueError('Train and test must match!')
    
    # ensure all specified columns exist in train/test
    missing_cols = [col for col in num_cols if col not in train.columns]
    if missing_cols:
        raise ValueError(f"The following columns are missing from train/test: {missing_cols}")
    
    df_train= train.copy()
    df_test = test.copy()

    if (num_type=='median'):
        for col in num_cols:
            ft_median= df_train[col].median()
            df_train[col]= df_train[col].fillna(ft_median)
            df_test[col]= df_test[col].fillna(ft_median)
    elif (num_type=='mode'):
        for col in num_cols:
            ft_mode= df_train[col].value_counts().index[0]
            df_train[col]= df_train[col].fillna(ft_mode)
            df_test[col]= df_test[col].fillna(ft_mode)
    elif (num_type=='zeros'):
        for col in num_cols:
            ft_zero= 0
            df_train[col]= df_train[col].fillna(ft_zero)
            df_test[col]= df_test[col].fillna(ft_zero)
    else:
        for col in num_cols:
            ft_mean= df_train[col].mean()
            df_train[col]= df_train[col].fillna(ft_mean)
            df_test[col]= df_test[col].fillna(ft_mean)

    return df_train, df_test


##############################################################
def pre_proc_fillna_cat_fts(df, cat_cols, cat_type='mode'):
    """
    Fill NaN values in categorical features using mode.
    cat_cols list indicating the columns names from df are categorical
    cat_type (str, optional) defines the fill type - mode

    RETURNS: df without nan's on cat_cols
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if not isinstance(cat_cols, list) or not all(isinstance(col, str) for col in cat_cols):
        raise TypeError("cat_cols must be a list of column names (strings).")
    
    missing_cols = [col for col in cat_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"These columns are missing from df: {missing_cols}")
    
    df_train= df.copy()
    
    # apply mode for categorical columns
    for col in cat_cols:
        if df_train[col].dtype == "object" or df_train[col].dtype.name == "category":
            mode_value= df_train[col].dropna().mode()[0] if not df_train[col].dropna().empty else ""
            df_train[col].fillna(mode_value, inplace=True)
        else:
            raise ValueError(f"Column '{col}' is not categorical.")

    return df_train


##############################################################
def pre_proc_fillna_cat_fts_train_test(train, test, cat_cols, cat_type='mode'):
    """
    Fill NaN values in categorical features using mode, ensuring test data 
    uses the same mode value computed from the training data.

    train and test are pd.DataFrame n-dimensional datasets resulting from a data split 
        (must have the same features)
    cat_cols list indicating the columns names from df are numerical
    cat_type (str, optional) defines the fill type - mode

    RETURNS: train and test without nan's in num_cols using the parameters from the training data.
    """
    if not isinstance(train, pd.DataFrame) or not isinstance(test, pd.DataFrame):
        raise TypeError("train and test must be pandas DataFrames.")

    if not isinstance(cat_cols, list) or not all(isinstance(col, str) for col in cat_cols):
        raise TypeError("cat_cols must be a list of column names (strings).")

    if not np.array_equal(train.columns, test.columns):
        raise ValueError("Train and test must have the same features.")

    missing_cols = [col for col in cat_cols if col not in train.columns]
    if missing_cols:
        raise ValueError(f"These columns are missing from train/test: {missing_cols}")
    
    df_train= train.copy()
    df_test = test.copy()
    
    for col in cat_cols:
        if df_train[col].dtype == "object" or df_train[col].dtype.name == "category":
            mode_value= df_train[col].dropna().mode()[0] if not df_train[col].dropna().empty else ""
            df_train[col].fillna(mode_value, inplace=True)
            df_test[col].fillna(mode_value, inplace=True)
        else:
            raise ValueError(f"Column '{col}' is not categorical.")

    return df_train, df_test


##############################################################
def replace_values(df, num_cols, num_type='mean', cat_type='none'):
    """
    Create a single row instance based on replacing each column (numerical and categorical) of the 
    original data by statistical values.
    - df: (pd.DataFrame) The dataset.
    - num_cols: (list) List of numerical column names.
    - num_type: (str, optional) Replacement strategy for numerical columns: 'mean', 'median', 'mode', 
        or 'zeros' (default is 'mean').
    - cat_type: (str, optional) Replacement strategy for categorical columns: 'mode'
        (default is 'none', meaning no change).

    RETURNS:
    - a single row pd.DataFrame with the values of replacement.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    
    if not isinstance(num_cols, list) or not all(isinstance(col, str) for col in num_cols):
        raise TypeError("num_cols must be a list of column names (strings).")

    missing_cols= [col for col in num_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"These numerical columns are missing from df: {missing_cols}")
    
    cat_values= None
    num_values= None
    
    # numerical column replacement
    if (num_type== 'mode'):
        num_values= df.mode(axis=0)
    elif (num_type== 'median'):
        num_values= df.median(axis=0).to_frame().T
    elif (num_type== 'mean'):
        num_values= df.mean(axis=0).to_frame().T
    elif (num_type== 'zeros'):
        num_values= df.loc[0:0,:].copy()
        num_values.loc[:,:]= 0
    
    # categorical column replacement
    if (cat_type== 'mode'):
        cat_values= df.mode(axis=0)

        cat_values[num_cols]= num_values[num_cols]
        return cat_values
    
    return num_values


##############################################################
def get_dummies_drop_first_only_binary_atts(data, cat_cols):
    """
    One-hot encodes categorical columns, dropping the first category 
    only for binary attributes.
    
    receive data as a pandas.DataFrame with categorical attributes
    cat_cols list indicating the columns names from df are numerical

    RETURNS: (pd.DataFrame) one-hot encoded dataset
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if not isinstance(cat_cols, list) or not all(isinstance(col, str) for col in cat_cols):
        raise TypeError("cat_cols must be a list of column names (strings).")

    missing_cols = [col for col in cat_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"These categorical columns are missing from data: {missing_cols}")
    
    # determine which columns are binary
    drop_first_cols= [col for col in cat_cols if data[col].nunique() == 2]

    # apply get_dummies with selective drop_first
    df_encoded= pd.get_dummies(data, columns=cat_cols, drop_first=False)
    df_encoded= pd.get_dummies(data, columns=drop_first_cols, drop_first=True)

    return df_encoded


##############################################################
def save_dataset(df, file_name='synthetic_data.csv'):
    """
    Save a pd.DataFrame dataset in a csv file
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    try:
        df.to_csv(file_name, index=False)
        print(f"Dataset saved as '{file_name}'.")
    except Exception as e:
        print(f"Error saving {file_name}: {e}")
