import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("https://raw.githubusercontent.com/whitehatjr/datasets/master/pro-c116/Admission_Predict.csv")
TOEFL = df["TOEFL Score"].tolist()
chance = df["Chance of admit"].tolist()
GRE = df["GRE Score"].tolist()

colors = []
for data in chance:
    if data == 1:
        colors.append("green")
    else:
        colors.append("red")

fig = go.Figure(data=go.Scatter(
    x=TOEFL, y=GRE, mode='markers', marker=dict(color=colors)
))
fig.show()

scores = df[["TOEFL Score", "GRE Score"]]
chances = df["Chance of admit"]

scores_train, scores_test, chances_train, chances_test = train_test_split(scores, chances, test_size = 0.25, random_state = 0)
print(scores_train[0:10])
sc_x = StandardScaler()

hours_train = sc_x.fit_transform(scores_train)
hours_test = sc_x.transform(scores_test)

print(scores_train[0:10])

classifier = LogisticRegression(random_state=0)
classifier.fit(scores_train, chances_train)

chances_pred = classifier.predict(scores_test)

print("Accuracy: ", accuracy_score(chances_test, chances_pred))

sc_x = StandardScaler() 
scores_train = sc_x.fit_transform(scores_train)  

TOEFL_score = int(input("Enter TOEFL score -> "))
GRE_score = int(input("Enter GRE score -> "))

user_test = sc_x.transform([[GRE_score, TOEFL_score]])

user_result_pred = classifier.predict(user_test)

if user_result_pred[0] == 1:
  print("This user may be admitted!")
else:
  print("This user may not be admitted!")