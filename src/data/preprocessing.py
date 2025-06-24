import pandas as pd
from sklearn.model_selection import train_test_split

def read_new_data(path: str) -> pd.DataFrame:
    "Reads the new data and creates a dataframe"
    return pd.read_excel(path)

def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns considered irrelevant for churn prediction, as identified during EDA.

    """
    useless_columns = ['CustomerID' ,'Lat Long', 'Churn Score', 'CLTV', 'Churn Reason', 'Churn Label']
    return df.drop(columns=useless_columns)

def drop_single_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns from the DataFrame that contain only a single unique value.
    These columns are not informative for predictive models.

    Returns:
        pd.DataFrame: A copy of the DataFrame without single-value columns.
    """
    single_value_cols = [col for col in df.columns if df[col].nunique() == 1]
    
    if single_value_cols:
        print(f"Dropping single-value columns: {single_value_cols}")
    
    return df.drop(columns=single_value_cols)

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces spaces in string values with underscores,
    and cleans column names by replacing spaces with underscores.
    """
    df_clean = df.copy()
    
    for col in df_clean.select_dtypes(include='object').columns:
        df_clean[col] = df_clean[col].str.replace(' ', '_')
    
    df_clean.columns = df_clean.columns.str.replace(' ', '_')
    
    return df_clean

def impute_numeric_missing(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Imputes missing values in numeric columns using the specified strategy.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        strategy (str): Strategy to use for imputation. Options are:
            - 'mean': Fill missing values with the mean of the column.
            - 'median': Fill missing values with the median of the column.

    Returns:
        pd.DataFrame: DataFrame with missing numeric values imputed.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            if strategy == 'mean':
                df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df[col].fillna(df[col].median())
            else:
                raise ValueError(f"Unsupported strategy '{strategy}' for numeric imputation.")

    return df

def impute_categorical_missing(df: pd.DataFrame, strategy: str = 'most_frequent') -> pd.DataFrame:
    """
    Imputes missing values in categorical columns using the specified strategy.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        strategy (str): Strategy to use for imputation. Options are:
            - 'most_frequent': Fill missing values with the mode of the column.
            - 'constant': Fill missing values with the string 'missing'.

    Returns:
        pd.DataFrame: DataFrame with missing categorical values imputed.
    """
    categorical_cols = df.select_dtypes(include='object').columns

    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            if strategy == 'most_frequent':
                df[col].fillna(df[col].mode()[0])
            elif strategy == 'constant':
                df[col].fillna('missing')
            else:
                raise ValueError(f"Unsupported strategy '{strategy}' for categorical imputation.")

    return df

def cast_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes column types by converting categorical object columns to 'category'
    and ensuring Total_Charges is numeric.

    Returns:
        pd.DataFrame: DataFrame with corrected types.
    """
    # Convert Total_Charges to numeric
    if df['Total_Charges'].dtype == 'object':
        df['Total_Charges'] = pd.to_numeric(df['Total_Charges'], errors='coerce')

    # Cast only real categorical columns
    categorical_cols = df.select_dtypes(include='object').columns.drop('Total_Charges', errors='ignore')
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    return df

def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies one-hot encoding to predefined categorical columns
    from the Telco Customer Churn dataset.

    Returns:
        pd.DataFrame: The one-hot encoded DataFrame.
    """
    cols_to_encode = [
        'City',
        'Gender',
        'Senior_Citizen',
        'Partner',
        'Dependents',
        'Phone_Service',
        'Multiple_Lines',
        'Internet_Service',
        'Online_Security',
        'Online_Backup',
        'Device_Protection',
        'Tech_Support',
        'Streaming_TV',
        'Streaming_Movies',
        'Contract',
        'Paperless_Billing',
        'Payment_Method'
    ]
    
    return pd.get_dummies(df, columns=cols_to_encode, dtype=int)

def split_data(df: pd.DataFrame, target_col: str):
    """
    Splits the input DataFrame into training, validation, and test sets with stratified sampling.
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test