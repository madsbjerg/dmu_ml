import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# skipping the header
data = pd.read_csv('titanic_800.csv', sep=',', header=0)

# show the data
#print(data.describe(include='all'))
#the describe is a great way to get an overview of the data

# Remove useless data from set
data.drop('PassengerId', axis=1, inplace=True)
data.drop('Name', axis=1, inplace=True)
data.drop('Cabin', axis=1, inplace=True)
data.drop('Ticket', axis=1, inplace=True)

# Replace unknown values. Unknown class set to 3rd class
data["Pclass"].fillna(3, inplace=True)
data["Embarked"].fillna('S', inplace=True)
data["Embarked"] = data["Embarked"].replace('S', 0.0)
data["Embarked"] = data["Embarked"].replace('C', 1.0)
data["Embarked"] = data["Embarked"].replace('Q', 2.0)

data["Sex"] = data["Sex"].replace('female', 0.0)
data["Sex"] = data["Sex"].replace('male', 1.0)

avg = data["Age"].mean() #set it to avg instead
data["Age"].fillna(avg, inplace=True)

# Replace unknown values. Unknown survival set to survived
data["Survived"].fillna(1, inplace=True)

yvalues = pd.DataFrame(dict(Survived=[]), dtype=int)
yvalues["Survived"] = data["Survived"].copy()
#now the yvalues should contain just the survived column

print(data.describe(include='all'))

x = data["Age"]
y = data["Pclass"]
plt.figure()
plt.scatter(x.values, y.values, color='green', s=20)
plt.show()

#now we can delete the survived column from the data (because
#we have copied that already into the yvalues.
data.drop('Survived', axis=1, inplace=True)

# show the data
print(data.describe(include='all'))

X_train, X_test, y_train, y_test = train_test_split(data, yvalues, train_size=0.8, stratify=yvalues, )
print(y_train)

#scaler til data for the memes
scaler = StandardScaler()
scaler.fit(X_train)

xtrain = scaler.transform(X_train) # uden skalering er accuracy faldet med 5%
xtest = scaler.transform(X_test)

#batch size 5 = 0,73, batch size 10 = 0,71
#learning rate init default = 0.001
mlp = MLPClassifier(hidden_layer_sizes=(12, 8, 4), max_iter=2000)
mlp.fit(X_train, y_train.values.ravel())

predictions = mlp.predict(X_test)
matrix = confusion_matrix(y_test, predictions)
print(matrix)

tn, fp, fn, tp = matrix.ravel()
print('tp: ' + str(tp))
print('tn: ' + str(tn))
print('fp: ' + str(fp))
print('fn: ' + str(fn))
accuracy = (tp + tn) / (tp+fp+fn+tn)
print("Accuracy: " + str(accuracy))

print(classification_report(y_test, predictions))

# EXPERIMENTING
svc = SVC(kernel='rbf', C=100)
svc.fit(X_train, y_train.values.ravel())

predictions = svc.predict(X_test)
matrix = confusion_matrix(y_test, predictions)
print(matrix)

tn, fp, fn, tp = matrix.ravel()
print('tp: ' + str(tp))
print('tn: ' + str(tn))
print('fp: ' + str(fp))
print('fn: ' + str(fn))
accuracy = (tp + tn) / (tp+fp+fn+tn)
print("Accuracy: " + str(accuracy))

print(classification_report(y_test, predictions))
