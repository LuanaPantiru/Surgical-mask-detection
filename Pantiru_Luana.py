from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import seaborn as sn
from glob import glob
import librosa as lr
from sklearn import preprocessing, svm
import matplotlib.pyplot as plt


def plotare(train):
    audio1, freq1 = lr.load(train[0])
    time1 = np.arange(0, len(audio1)) / freq1
    audio2, freq2 = lr.load(train[2])
    time2 = np.arange(0, len(audio2)) / freq2
    audio3, freq3 = lr.load(train[1])
    time3 = np.arange(0, len(audio3)) / freq3
    audio4, freq4 = lr.load(train[6])
    time4 = np.arange(0, len(audio4)) / freq4
    fig1, ax1 = plt.subplots()
    ax1.plot(time1, audio1)
    plt.title('100001.wav,0')
    ax1.set(xlabel='time (s)', ylabel='sound amplitude')
    plt.show()
    fig2, ax2 = plt.subplots()
    ax2.plot(time2, audio2)
    plt.title('100003.wav,0')
    ax2.set(xlabel='time (s)', ylabel='sound amplitude')
    plt.show()
    fig3, ax3 = plt.subplots()
    ax3.plot(time3, audio3)
    plt.title('100002.wav,1')
    ax3.set(xlabel='time (s)', ylabel='sound amplitude')
    plt.show()
    fig4, ax4 = plt.subplots()
    ax4.plot(time4, audio4)
    plt.title('100007.wav,1')
    ax4.set(xlabel='time (s)', ylabel='sound amplitude')
    plt.show()


def confusion_matrix_plot(label, pred):
    conf_matrix = confusion_matrix(label, pred)
    fig, ax = plt.subplots()
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(conf_matrix, annot=True, annot_kws={"size": 16})  # font size
    ax.set(xlabel='Actual', ylabel='Predicted')
    plt.show()
    print(conf_matrix)


def formare(file, array, label_np):
    aux = np.zeros((len(array), 128), dtype='float32')
    i = 0
    for wav in file:
        label_np = np.append(label_np, array[wav[-10:]])
        audio, freq = lr.load(wav)
        mfccs = lr.feature.mfcc(y=audio, sr=freq, n_mfcc=128)
        mfccsscaled = np.mean(mfccs.T, axis=0)
        aux[i] = mfccsscaled
        i += 1
    return aux, label_np


def normalizare(train_data, validation_data, test_data, type=None):
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()
    elif type == 'min-max':
        scaler = preprocessing.MinMaxScaler()
    elif type == 'l1':
        scaler = preprocessing.Normalizer(norm='l1')
    elif type == 'l2':
        scaler = preprocessing.Normalizer(norm='l2')
    else:
        print("type incorect")
        return
    scaler.fit(train_data)
    # scaler.fit(audio_validation)
    scaled_train_data = scaler.transform(train_data)
    scaled_validation_data = scaler.transform(validation_data)
    scaled_test_data = scaler.transform(test_data)
    return scaled_train_data, scaled_validation_data, scaled_test_data

# setam path-ul pentru fiecare set de date
data_validare = "./validation/validation"
data_train = "./train/train"
data_test = "./test/test"
# pentru fiecare set de date memoram fisierele audio
validare = glob(data_validare + '/*.wav')
train = glob(data_train + '/*.wav')
test = glob(data_test + '/*.wav')
# citim fisierele de tip .txt pentru a putea sa ne scoatel label-urile
file1 = open("validation.txt", "r")
file2 = open("train.txt", "r")
array1 = dict()
array2 = dict()
# intr-un dictionar tinem minte numele fisierului si label-ul specific
for fileline in file1:
    aux = fileline.split(",")
    array1[aux[0]] = int(aux[1][0])
for fileline in file2:
    aux = fileline.split(",")
    array2[aux[0]] = int(aux[1][0])
# plotam cateva fisiere audio in functie amplitudine si timp
plotare(train)
# ne stabilim feature-urile pe fiecare set de date si a label-urilor
label_validation = np.array([])
label_train = np.array([])
audio_validation, label_validation = formare(validare, array1, label_validation)
audio_train, label_train = formare(train, array2, label_train)
audio_test = np.zeros((3000, 128), dtype='float32')
array_test = []
i = 0
for wav in test:
    array_test.append(wav[-10:])
    audio, freq = lr.load(wav)
    mfccs = lr.feature.mfcc(y=audio, sr=freq, n_mfcc=128)
    mfccsscaled = np.mean(mfccs.T, axis=0)
    audio_test[i] = mfccsscaled
    i += 1
print("gata citire datelor")
# normalizam features-urile
train_norm, validation_norm, test_norm = normalizare(audio_train, audio_validation, audio_test, 'standard')
print("gata normalizarea features-utilor")
'''
# cautarea valorilor bune pentru hiperparametrii  
param = {'C':[0.1,1,10,100,1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],'kernel': ['linear','poly','rbf']}
search = GridSearchCV(svm.SVC(),param,refit = True, verbose=3)
search.fit(train_norm,label_train)
print(svm.best_params_)
print(svm.best_estimator_)
'''
# definirea modelului
svm = svm.SVC(C=10, kernel='linear', gamma=1)
# antrenam datele de train pe model
svm.fit(train_norm, label_train)
print("gata antrenarea")
# prezicem label-urile pentru datele de validation
pred_validation = svm.predict(validation_norm)
print("gata predictia pentru validare")
confusion_matrix_plot(label_validation, pred_validation)
acc = accuracy_score(label_validation, pred_validation)
print("accuracy : " + str(acc))
f1score = f1_score(label_validation, pred_validation)
print("f1_score : " + str(f1score))
# prezicem label-urile pentru datele de test
pred = svm.predict(test_norm)
# afisam prezicerile in fisier
file3 = open("predictie.txt", "w")
file3.write("name,label\n")
for i in range(len(array_test)):
    elem = array_test[i] + "," + str(int(pred[i])) + "\n"
    file3.write(elem)
'''
# MLPClassifier
train_feats_sc, validation_feats_sc , test_feats_sc = normalizare(audio_train, audio_validation, audio_test, 'standard')
print("gata normalizare")
results = []


def train_test_model(model, X_train, y_train, X_validare, y_validare ,X_test):
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    validare_accuracy = model.score(X_validare, y_validare)
    y_test = model.predict(X_test)
    no_iter = model.n_iter_

    results.append(( train_accuracy, validare_accuracy, no_iter))
    return y_test


mlp = MLPClassifier(activation= 'relu', alpha= 0.005, hidden_layer_sizes= (100, 100), learning_rate_init= 0.01)
pred = train_test_model(mlp, train_feats_sc, label_train, validation_feats_sc, label_validation,test_feats_sc)
print(tabulate(results,headers=["Accuracy for train set","Accuracy for test set","No. of iterations until convergence"]))
print(pred)
'''
'''
# cautarea valorilor bune pentru hiperparametrii lui MLPClassifier
param = {'hidden_layer_sizes': [(1,), (10,), (10, 10), (100, 100), (100,), (50,), (50, 50)],
         'activation': ['tanh', 'relu', 'logistic', 'identity'], 'solver': ['sgd', 'adam'],
         'learning_rate_init': [0.01, 0.00001, 10], 'momentum': [0, 0.9], 'alpha': [0.0001, 0.005]}
clf = GridSearchCV(mlp, param, n_jobs=-1, cv=3)
clf.fit(train_feats_sc,label_train)
print("gata fit")
print(clf.best_params_)
print(clf.best_estimator_)
'''

