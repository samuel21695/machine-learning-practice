# 라이브러리 import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

# 데이터 불러오기
iris_df = pd.read_csv("./data/iris.csv")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(iris_df.drop("iris_species", axis=1), iris_df["iris_species"], test_size=0.2)

# 모델 생성 및 학습
model = KNeighborsClassifier(n_neighbors=3)

# 에러 해결 코드
columns_to_convert = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# 데이터 전처리: "..." 값을 NaN으로 변환, NaN 값을 평균값으로 대체
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 데이터 전처리: "..." 값을 숫자 값으로 변환
X_train_copy = X_train.copy()
for column in columns_to_convert:
    X_train_copy[column].replace("...", float('nan'), inplace=True)  # "..." 값을 NaN으로 변환

# Imputation with SimpleImputer
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train_copy)
X_test = imputer.transform(X_test)

# Convert to DataFrame and ensure numeric data type
X_train = pd.DataFrame(X_train, columns=columns_to_convert)
X_test = pd.DataFrame(X_test, columns=columns_to_convert)

# Check data types (optional)
print(X_train.dtypes)

# 문제가 있는 행을 찾아 수정 또는 제거
for index, row in X_train.iterrows():
    for column in columns_to_convert:
        if not isinstance(row[column], float):
            # 수정이 가능한 경우:
            row[column] = "올바른 숫자 값"
            # 수정이 불가능한 경우:
            X_train = X_train.drop(index)  # 행 제거
            break  # 다음 행으로 이동
        
# 문자열 열을 숫자로 변환 (NaN 값 포함)
X_train = pd.DataFrame(X_train, columns=columns_to_convert)
X_test = pd.DataFrame(X_test, columns=columns_to_convert)

model.fit(X_train, y_train)

# 예측 수행
y_pred = model.predict(X_test)

# 정확도 평가
accuracy = model.score(X_test, y_test)
print("정확도:", accuracy)

from flask import Flask, request, jsonify

# Flask 앱 설정
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = pd.DataFrame(data)
    prediction = model.predict(features)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
