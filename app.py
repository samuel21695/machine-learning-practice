# 라이브러리 import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 데이터 불러오기
iris_df = pd.read_csv("./data/iris.csv")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(iris_df.drop("iris_species", axis=1), iris_df["iris_species"], test_size=0.2)

# 모델 생성 및 학습
model = KNeighborsClassifier(n_neighbors=3)

# 에러 해결 코드
columns_to_convert = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# 문제 데이터 확인
print(X_train[columns_to_convert])

# 문제가 있는 행을 찾아 수정 또는 제거
for index, row in X_train.iterrows():
    for column in columns_to_convert:
        if "..." in row[column]:
            # 수정이 가능한 경우:
            row[column] = "올바른 숫자 값"
            # 수정이 불가능한 경우:
            X_train = X_train.drop(index)  # 행 제거
            break  # 다음 행으로 이동
        
# 문자열 열을 숫자로 변환
X_train[columns_to_convert] = X_train[columns_to_convert].apply(pd.to_numeric, errors='coerce')

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
