
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Create grade categories
    grade_labels = ['F', 'D', 'D+', 'C', 'C+', 'B', 'B+', 'A', 'A+']
    score_bins = [0, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    df['Grade_Category'] = pd.cut(df['Exam_Score'], bins=score_bins,
                                  labels=grade_labels, right=False)

    df.dropna(inplace=True)

    X = df.drop(['Grade_Category', 'Exam_Score'], axis=1)
    y = df['Grade_Category']

    num_cols = X.select_dtypes(include=['int64','float64']).columns
    cat_cols = X.select_dtypes(include='object').columns

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor
