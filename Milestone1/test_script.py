import pandas as pd
import numpy as np
import re
import joblib
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer
import warnings

warnings.filterwarnings('ignore')


# ===== Helper Cleaning and Encoding Functions =====

def drop_columns(df):
    to_drop = joblib.load("drop.joblib")
    return df.drop(columns=to_drop, errors='ignore')

def clean_numeric_columns(df):
    numeric_cols = joblib.load('numerical_cols.joblib')
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^0-9\.\-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def impute_numerical(df):
    imputer = joblib.load('numerical_imputer.joblib')
    cols = joblib.load('numerical_cols.joblib')
    df[cols] = imputer.transform(df[cols])
    return df

def impute_categorical(df):
    modes = joblib.load('categorical_modes.joblib')
    cols = joblib.load('categorical_cols.joblib')
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(modes.get(col, 'Unknown'))
    return df

def clean_text_columns(df):
    columns = joblib.load('columns_to_clean.joblib')
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)))
    return df

def clean_host_location(df):
    def expand(loc):
        if pd.isnull(loc): return 'Unknown'
        loc = loc.lower().strip()
        loc = re.sub(r'[^\w\s,]', '', loc)
        loc = re.sub(r'\b(ny|ca|tx|us|usa|united states)\b', 'United States', loc)
        return loc.title()
    if 'host_location' in df.columns:
        df['host_location'] = df['host_location'].apply(expand)
    return df

def load_and_apply_url_ids(df):
    try:
        urls = joblib.load('urls.joblib')
        for col in urls.columns:
            if col in df.columns:
                df[col] = urls[col]
    except:
        pass
    return df

def process_date_columns(df):
    cols = joblib.load('date_columns.joblib')
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_year"] = df[col].dt.year
    return df.drop(columns=cols, errors='ignore')

def clean_state_column(df):
    mapping = joblib.load('state_mapping.joblib')
    if 'state' in df.columns:
        df['state'] = df['state'].str.upper().str.strip()
        df['state'] = df['state'].replace(mapping)
    return df

def clean_zipcode(df):
    if 'zipcode' in df.columns:
        df['zipcode'] = df['zipcode'].astype(str).str.split('-').str[0]
    return df

def apply_label_encoding(df):
    encoders = joblib.load('label_encoders.joblib')
    for col, le in encoders.items():
        if col in df.columns:
            known = list(le.classes_)
            df[col] = df[col].astype(str)
            new_vals = df[col][~df[col].isin(known)].unique()
            if len(new_vals):
                le.classes_ = np.append(le.classes_, new_vals)
            df[col] = le.transform(df[col])
    return df

def apply_word_count(df):
    cols = joblib.load('wordcount_columns.joblib')
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: len(str(x).split()))
    return df

def clean_and_apply_amenities_bow(df):
    try:
        vectorizer = joblib.load("amenities_vectorizer.joblib")
        if "amenities" in df.columns:
            bow_matrix = vectorizer.transform(df["amenities"].fillna(""))
            bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
            bow_df.index = df.index
            df = pd.concat([df.drop(columns=["amenities"], errors='ignore'), bow_df], axis=1)
    except:
        pass
    return df

def scale_target(df):
    scaler_dict = joblib.load('scaler_target.joblib')
    scaler = scaler_dict['scaler_target']
    scaler_cols = scaler_dict['target']
    df[scaler_cols] = scaler.transform(df[scaler_cols])
    return df


def save_cleaned_df(df):
    df.to_csv("cleaned_output.csv", index=False)

def load_and_clean_dataframe(file_path):
    df = pd.read_csv(file_path)
    has_target = 'review_scores_rating' in df.columns
    y_true = None

    if has_target:
        df = scale_target(df)
        y_true = df['review_scores_rating'].copy()

    df = drop_columns(df)
    df = clean_numeric_columns(df)
    df = impute_numerical(df)
    df = impute_categorical(df)
    df = clean_text_columns(df)
    df = clean_host_location(df)
    df = load_and_apply_url_ids(df)
    df = process_date_columns(df)
    df = clean_state_column(df)
    df = clean_zipcode(df)
    df = apply_label_encoding(df)
    df = apply_word_count(df)
    df = clean_and_apply_amenities_bow(df)

    feature_cols = joblib.load('feature_cols.joblib')
    df = df[[col for col in feature_cols if col in df.columns]]

    scaler_dict = joblib.load('scaler.joblib')
    scaler = scaler_dict['scaler']
    scaler_cols = scaler_dict['columns']
    X_scaled = scaler.transform(df[scaler_cols])

    with open('best_regression_models.pkl', 'rb') as f:
        models = pickle.load(f)
    all_predictions={}
    predictions = pd.DataFrame()
    all_mse = {}
    all_r2 = {}

    predictions_df = pd.DataFrame()
    for name, model in models.items():
        predictions_df[name] = model.predict(X_scaled)

    for name, model in models.items():
        preds = model.predict(X_scaled)
        predictions[name] = preds
        all_predictions=preds

        if y_true is not None:
            mse = mean_squared_error(y_true, preds)
            r2 = r2_score(y_true, preds)
            all_mse[name] = mse
            all_r2[name] = r2

    return predictions, all_mse if has_target else None, all_r2 if has_target else None


# ===== Run Script =====

if __name__ == "__main__":
    predictions, mse_dict, r2_dict = load_and_clean_dataframe("reglabel3.csv")

    print("\n📊 Model Predictions:")
    print(predictions.to_markdown(index=False))

    if mse_dict is not None and r2_dict is not None:
        print("\n📉 Model MSE:")
        for model, mse in mse_dict.items():
            print(f"  {model}: {mse:.4f}")

        print("\n📈 Model R²:")
        for model, r2 in r2_dict.items():
            print(f"  {model}: {r2:.4f}")
