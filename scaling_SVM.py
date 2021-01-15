"""
--- Computational Scaling of SVMs ---
- Python function from which you can generate synthetic datasets of an arbitrary size (e.g. a mixture model of two Gaussians).
- By training SVMs on datasets of different sizes showing
         -How the training time scales
         -How the classification time scales (on a sample of 1000 unseen data points)
         -The accuracy of the classifier (on a sample of 1000 unseen data points)
- Chose an appropriate kernel for the task at hand.
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm
import pandas as pd

def getdata(n):
    class_plus = np.random.normal(loc=1,scale=2,size=n)
    class_minus = np.random.normal(loc=2,scale=4,size=n)
    label_plus, label_minus = [],[]

    for i in range(n):
        label_plus.append("Plus")
        label_minus.append("Minus")

    labels = list(label_plus) + list(label_minus)
    points = list(class_plus) + list(class_minus)

    data = pd.DataFrame(data=labels, columns=['Label'])
    data["Points"] = points
    
    return(data)

def fitData(num):
    data = getdata(num)
    X = data[['Points']]
    y = data[['Label']]
    y = np.ravel(y)
    clf = svm.SVC(kernel='rbf')
    clf.fit(X,y)
    return clf

def pred(clf,num):
    data = getdata(num)
    X = np.ravel(data[["Points"]])
    X = X.reshape(-1, 1)
    y_pred = clf.predict(X)
    return y_pred

def accuracy(clf, num):
    data = getdata(num)
    X,y = data[["Points"]], data[["Label"]]
    y,X = np.ravel(y), np.ravel(X)
    y,X = y.reshape(-1, 1), X.reshape(-1, 1) 
    y_pred = clf.predict(X)
    return accuracy_score(y, y_pred)

for i in [100, 1000, 10000]:
    print("SVM:", i, "data points\n")
    clf = fitData(i)
    %timeit fitData(i)
    %timeit pred(clf, 1000) #predicting
    print(accuracy(clf, 1000)) #metric

### OUTPUT
#SVM: 100 data points
#3.02 ms ± 39.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
#6.4 ms ± 304 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
#0.662
#SVM: 1000 data points
#153 ms ± 4.2 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
#41.9 ms ± 4.54 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
#0.6625
#SVM: 10000 data points
#25.9 s ± 2.94 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
#503 ms ± 47 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
#0.678
