import matplotlib.pyplot as plt # for plotting
from sklearn import datasets, svm, metrics, neighbors # for dataset, classifier & metrics

#loading digits dataset
digits = datasets.load_digits()

#getting images & labels
digit_classes = list(zip(digits.images, digits.target))

#plotting all classes 0-9 digits
for index, (image, label) in enumerate(digit_classes[0:10]):
    plt.subplot(3, 4, index + 1) #subplot for each digit
    plt.tight_layout() #spacing
    plt.imshow(image, cmap=plt.cm.binary, interpolation='nearest') #image
    plt.title('Digit: %i' % label) #label
plt.show()

#choosing two digit classes (2 & 4) for training KNN Classifier
trainImages = digits.images[np.where(np.logical_or(digits.target == 2 , digits.target == 4 ))]
trainLabels = digits.target[np.where(np.logical_or(digits.target == 2 , digits.target == 4 ))]

#data preprocessing
n = len(trainImages)
X = trainImages.reshape((n, -1)) #image as its own list
y = trainLabels.reshape((n, -1)) #image class label

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#splitting data into training set (80% data) and test set (20% data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = KNeighborsClassifier(4, weights='uniform') # KNN classifier, k = 4 (randomly chosen)
classifier.fit(X_train, y_train) #model training

y_pred = classifier.predict(X_test) #predicting on test set

print("Accuracy Score: %s" % metrics.accuracy_score(y_test, y_pred)) #evaluating prediction
#Got 100% accuracy (0% error rate). Even playing with different k values the classifier model was able to distinguish between 2 and 4
