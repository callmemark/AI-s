import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv("Color_data.csv")
pd.set_option("display.max_rows", 117)


R = data["R"]
G = data["G"]
B = data["B"]

X = data.iloc[:, :-2].values
y = data["Id"]



model = KNeighborsClassifier(n_neighbors = 7)
model.fit(X, y)


input = [94, 108, 204]



prediction = model.predict([input])
print(prediction)


if prediction == [1]: print("red-shade")
if prediction == [2]: print("orange-shade")
if prediction == [3]: print("yellow-shade")
if prediction == [4]: print("yellowgreen-shade")
if prediction == [5]: print("green-shade")
if prediction == [6]: print("bluegreen-shade")
if prediction == [7]: print("neon-shade")
if prediction == [8]: print("lightblue-shade")
if prediction == [9]: print("blue-shade")
if prediction == [10]: print("bluviolet-shade")
if prediction == [11]: print("violet-shade")
if prediction == [12]: print("pink-shade")
if prediction == [13]: print("black-shade")
