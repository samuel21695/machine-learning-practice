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
