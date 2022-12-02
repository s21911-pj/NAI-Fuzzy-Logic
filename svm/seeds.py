import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


from utilities import visualize_classifier

# Load input data from file
df = pd.read_csv("seeds.csv", delimiter=";")
# separate input data - X and output data - Y
X = df.drop('Class', axis=1)
y = df['Class']
# create data for test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# we create an instance of SVM and fit out data.
svc = SVC().fit(X_train, y_train)

# calculate accuracy score and display score
y_pred = svc.predict(X_test)
accuracy_score(y_test, y_pred)
print('Accuracy of model is', accuracy_score(y_test, y_pred))


# display score
X = df[["Length of kernel", "Length od kernel groove"]].to_numpy()
y = df['Class'].to_numpy()
svc = SVC().fit(X, y)
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Reds)
plt.xlabel('Length of kernel')
plt.ylabel('Length od kernel groove')
plt.xticks([x_min,x_max])
plt.yticks([y_min, y_max])
plt.title('SVC')
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap = plt.cm.Paired, alpha = 0.1)
plt.show()



# Separate input data into two classes based on labels
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])

# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=100)

# Decision Trees classifier
classifier = DecisionTreeClassifier(random_state=0,max_depth=8)
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
accuracy_score(y_test, y_test_pred)

# Visualize data
visualize_classifier(classifier, X_train, y_train, 'Training dataset')
visualize_classifier(classifier, X_test, y_test, 'Test dataset')
plt.show()

# Evaluate classifier performance
class_names = ['1', '2', '3']
print("\n" + "#"*40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
print("#"*40 + "\n")

print("#"*40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")