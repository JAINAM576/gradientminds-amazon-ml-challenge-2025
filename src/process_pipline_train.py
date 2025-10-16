import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sentence_transformers import SentenceTransformer
import joblib


df=pd.read_csv("final_category.csv")

numeric_features = [
    'unit_qty', 'pack_count', 'total_qty',
    'num_bullet_points', 'num_product_desc',
    'total_chars_bullet_points', 'total_chars_product_desc', 'avg_bullet_point_len'
]

categorical_features = ['unit', 'brand_name',"category"]
text_feature = 'catalog_content'


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', TargetEncoder(cols=categorical_features, handle_unknown='impute'), categorical_features)
    ])


def generate_text_embeddings(df, model_name='Lajavaness/bilingual-embedding-large', batch_size=128, device='cuda'):
    model = SentenceTransformer(model_name, device=device,trust_remote_code=True)
    texts = df[text_feature].astype(str).tolist()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, device=device)
    return np.array(embeddings)


def process_catalog_df(df, target_col='price', test=False, pipeline_path='process_pipeline.pkl', model_name='Lajavaness/bilingual-embedding-large'):
    if not test:
        preprocessor = build_preprocessor()
        X_non_text = df[numeric_features + categorical_features]
        y = df[target_col]
        preprocessor.fit(X_non_text, y)
        joblib.dump(preprocessor, pipeline_path)
        print(f"Non-text pipeline saved to {pipeline_path}")
    else:
        preprocessor = joblib.load(pipeline_path)
        print(f"Non-text pipeline loaded from {pipeline_path}")

    X_non_text_processed = preprocessor.transform(df[numeric_features + categorical_features])

    text_embeddings = generate_text_embeddings(df, model_name=model_name)

    X_processed = np.hstack((X_non_text_processed, text_embeddings))
    return X_processed


X_train = process_catalog_df(df, target_col='price', test=False)

