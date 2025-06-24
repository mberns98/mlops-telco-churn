import sys
sys.path.append('./') # Ensures local modules in 'src/' can be imported

from src.data.preprocessing import (
    read_new_data,
    drop_useless_columns,
    drop_single_value_columns,
    clean_column_names,
    impute_numeric_missing,
    impute_categorical_missing,
    cast_column_types,
    one_hot_encode
)

def run_pipeline(path: str):
    """
    Runs the full preprocessing pipeline on the input data.
    
    Args:
        path (str): Path to the raw Excel file.
    
    Returns:
        None.
    """
    df = (
        read_new_data(path)
        .pipe(drop_useless_columns)
        .pipe(drop_single_value_columns)
        .pipe(clean_column_names)
        .pipe(impute_numeric_missing)
        .pipe(impute_categorical_missing)
        .pipe(cast_column_types)
        .pipe(one_hot_encode)
    )
    print(df.head()) 

if __name__ == "__main__":
    # Check if the script was run with the required file path argument
    if len(sys.argv) < 2:
        print("Usage: poetry run python run_pipeline.py <path_to_excel>")
    else:
        path = sys.argv[1] # Extract the input file path
        run_pipeline(path) # Run the preprocessing pipeline