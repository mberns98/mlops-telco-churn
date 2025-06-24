import pandas as pd
from src.data import preprocessing as prep
from models.logreg_model import tune_logistic_regression, train_final_model
from models.inference import predict_and_evaluate
from models.evaluation_utils import plot_roc_curve

# ⚠️ Convertir target si está en formato texto
def convert_target(df):
    if df["Churn"].dtype == "object":
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df

def main():
    # 🧪 Leer y procesar
    df_raw = prep.read_new_data("data/Telco_Customer_Churn.xlsx")

    df_clean = (
        df_raw
        .pipe(convert_target)
        .pipe(prep.drop_useless_columns)
        .pipe(prep.drop_single_value_columns)
        .pipe(prep.clean_column_names)
        .pipe(prep.impute_numeric_missing)
        .pipe(prep.impute_categorical_missing)
        .pipe(prep.cast_column_types)
        .pipe(prep.one_hot_encode)
    )

    # 🔀 Split
    X_train, X_val, X_test, y_train, y_val, y_test = prep.split_data(df_clean, target_col="Churn")

    # 🧠 Tuning + entrenamiento
    print("🔍 Tuning Logistic Regression...")
    best_params = tune_logistic_regression(X_train, y_train)

    print("\n🛠️ Entrenando modelo final...")
    model = train_final_model(X_train, y_train, best_params)

    # 📊 Evaluación
    print("\n📊 Evaluación en test set:")
    predict_and_evaluate(model, X_test, y_test, labels=["No", "Yes"])

    # 📈 Mostrar curva ROC
    plot_roc_curve(model, X_test, y_test)

if __name__ == "__main__":
    main()