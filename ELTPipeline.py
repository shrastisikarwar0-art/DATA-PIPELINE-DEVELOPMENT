# etl_pipeline.py
# ------------------------------
# A simple automated ETL pipeline using Pandas and Scikit-learn
# ------------------------------

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# =====================
# Step 1: Extract
# =====================
def extract_data(file_path):
    """Load raw data from a CSV file"""
    print("Extracting data...")
    df = pd.read_csv(file_path)
    print(f"Data shape: {df.shape}")
    return df


# =====================
# Step 2: Transform
# =====================
def transform_data(df):
    """Preprocess the data: handle missing values, encode, and scale"""

    print("Transforming data...")

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    # Numeric pipeline: impute + scale
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline: impute + one-hot encode
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine both using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Final pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    # Fit and transform the data
    X_transformed = pipeline.fit_transform(X)

    print("Transformation complete.")
    return X_transformed, y, pipeline


# =====================
# Step 3: Load
# =====================
def load_data(X_transformed, y, output_path="processed_data.csv"):
    """Save processed data to a CSV file"""
    print("Loading (saving) processed data...")
    processed_df = pd.DataFrame(X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed)
    processed_df["target"] = y.values
    processed_df.to_csv(output_path, index=False)
    print(f"Data successfully saved to {output_path}")


# =====================
# Main ETL Runner
# =====================
def run_etl_pipeline(file_path):
    df = extract_data(file_path)
    X_transformed, y, pipeline = transform_data(df)
    load_data(X_transformed, y)
    print("\n✅ ETL Pipeline Completed Successfully!")


# =====================
# Run Example
# =====================
if __name__ == "__main__":
    # Example: Suppose we have a dataset "sample_data.csv" with a 'target' column
    # You can replace it with your own CSV file path
    sample_data = {
        "age": [25, 30, None, 22, 28],
        "income": [50000, 60000, 55000, None, 52000],
        "city": ["Delhi", "Mumbai", "Delhi", "Chennai", None],
        "target": [1, 0, 1, 0, 1]
    }

    # Creating a temporary CSV to simulate extraction
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv("sample_data.csv", index=False)

    # Run ETL
    run_etl_pipeline("sample_data.csv")
