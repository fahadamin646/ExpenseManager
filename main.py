from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

df = pd.read_csv (r'exp_dataset.csv')

X = df.iloc[:, 0:9].values
y = df.iloc[:, 9].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc = StandardScaler()
sc.fit(X_train)
with open('stand_scalar','wb') as f:
  pickle.dump(sc,f)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = RandomForestClassifier(n_estimators=100,
bootstrap = True,
max_features = 'sqrt')

ppn.fit(X_train_std, y_train)
y_pred=ppn.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy Score")
print(accuracy_score(y_test, y_pred))

with open('model','wb') as f:
  pickle.dump(ppn,f)