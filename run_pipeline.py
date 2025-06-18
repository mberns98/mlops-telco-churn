import sys
sys.path.append('./')

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

    if len(sys.argv) < 2:
        print("Usage: poetry run python run_pipeline.py <path_to_excel>")
    else:
        path = sys.argv[1]
        run_pipeline(path)