import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("D:\\EXP-19ML\\fake_news_detection_1000.csv")
df = df.dropna()
le = LabelEncoder()
df["news_type"] = le.fit_transform(df["news_type"])

df= df.drop("label",axis = 1)
x = df.drop("news_type",axis = 1)
y = df["news_type"]

print(y.info())
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
news_model = LogisticRegression()
news_model.fit(x_train,y_train) 
y_pred = news_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Prediction", y_pred[0])
print("Accuracy:", accuracy)
with open("news_model.pkl", "wb") as file:
    pickle.dump(news_model, file)

