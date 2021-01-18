from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from matplotlib import pyplot as plt
import pandas as pd
import os
import csv
import numpy as np

# Load Data
df = pd.read_csv('dataset_speech.csv')
fitur = []
for i in range(len(df['fitur0'])):
    temp = []
    # 13 Jumlah Fitur MFCC
    for f in range(13):
        temp.append(df['fitur'+str(f)][i])
    fitur.append(temp)
fitur = np.array(fitur)
target = np.array([k for k in df['kelas']])

#Model Adaboost Implementasi Machine Learning
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), \
                             n_estimators=200,algorithm="SAMME.R", \
                             learning_rate=1)
# Training
ada_clf.fit(fitur, target)

# Testing
y_pred=ada_clf.predict(fitur)
print(y_pred)
print(target)

# F1 Score
cm = confusion_matrix(y_pred, target)
print(cm)
plt.matshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Prediksi")
plt.xlabel("Target")
plt.colorbar()
plt.show()
