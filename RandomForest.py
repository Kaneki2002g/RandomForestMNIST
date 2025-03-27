import idx2numpy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

train_images = idx2numpy.convert_from_file('Data/emnist-mnist-train-images-idx3-ubyte')
train_labels = idx2numpy.convert_from_file('Data/emnist-mnist-train-labels-idx1-ubyte')

test_images = idx2numpy.convert_from_file('Data/emnist-mnist-test-images-idx3-ubyte')
test_labels = idx2numpy.convert_from_file('Data/emnist-mnist-test-labels-idx1-ubyte')


X_train = train_images.reshape(train_images.shape[0], -1)
X_test = test_images.reshape(test_images.shape[0], -1)

y_train = train_labels
y_test = test_labels


clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Precision: {precision:.4f}')