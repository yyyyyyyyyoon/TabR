import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(paths):
    if isinstance(paths, str):
        paths = [paths]

    splits = {}

    for path in paths:
        dataset_name = os.path.splitext(os.path.basename(path))[0]  # 확장자 제거한 파일명 추출

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue

        target_col = "class"
        if target_col not in df.columns:
            print(f"Warning: {path} 파일에서 '{target_col}' 컬럼을 찾을 수 없습니다.")
            continue

        #class 제외한 컬럼 (입력 데이터)
        features = df.drop(target_col, axis=1)
        #class 컬럼 (정답 데이터)
        labels = df[target_col]

        #학습 데이터와 테스트 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=0.2,
            stratify=labels,
            random_state=42
        )

         # Min-Max 스케일링
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

         # SMOTE 오버샘플링
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

        splits[dataset_name] = {
            "X_train": X_train_resampled, "X_test": X_test_scaled,
            "y_train": y_train_resampled, "y_test": y_test
        }
    return splits
