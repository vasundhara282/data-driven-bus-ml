from tkinter import messagebox

from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np                      #type:ignore
from tkinter import filedialog
import matplotlib.pyplot as plt         #type:ignore
import pandas as pd                     #type:ignore

from sklearn.model_selection import train_test_split   #type:ignore
import matplotlib.pyplot as plt                        #type:ignore
from sklearn.metrics import r2_score                    #type:ignore
from sklearn.metrics import mean_absolute_error         #type:ignore
from sklearn.model_selection import train_test_split    #type:ignore
from sklearn.preprocessing import MinMaxScaler          #type:ignore
from sklearn.ensemble import RandomForestRegressor      #type:ignore
from sklearn.metrics import mean_squared_error          #type:ignore
from math import sqrt
from sklearn.neighbors import NeighborhoodComponentsAnalysis     #type:ignore
from sklearn.linear_model import LinearRegression                 #type:ignore
from sklearn.svm import SVR                                         #type:ignore
from sklearn.gaussian_process import GaussianProcessRegressor       #type:ignore

from keras.models import Sequential #type:ignore
from keras.layers import Dense, Activation #type:ignore
from keras.layers import Dropout      #type:ignore
from keras.layers import MaxPooling2D   #type:ignore
from keras.layers import Flatten        #type:ignore
from keras.layers import Convolution2D   #type:ignore
from keras.models import Sequential      #type:ignore
from keras.callbacks import ModelCheckpoint  #type:ignore
import os

main = tkinter.Tk()
main.title("Data Driven Energy Economy Prediction for Electric City Buses Using Machine Learning")
main.geometry("1300x1200")

global filename
global r2, rmse, map_error
global dataset
global X, Y #type:ignore
global X_train, X_test, y_train, y_test, cnn_model, scaler, scaler1, nca
global r2, rmse, map_error  #type:ignore

def uploadDataset():
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,'Dataset loaded\n\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset))    
            
def processDataset():
    global dataset, X, Y
    global X_train, X_test, y_train, y_test, scaler, scaler1, nca
    text.delete('1.0', END)
    Y = dataset['Fuel Rate[L/hr]'].ravel()
    dataset.drop(['Fuel Rate[L/hr]'], axis = 1,inplace=True)
    text.insert(END, "Dataset Cleaning & Normalization Completed\n\n")
    X = dataset.values
    text.insert(END,"Total features found in Dataset before applying Neighbor Hood Component : "+str(X.shape[1])+"\n")
    y1 = np.where(Y > 5, 1, 0)
    nca = NeighborhoodComponentsAnalysis(n_components=15, random_state=42)
    X = nca.fit_transform(X, y1)
    text.insert(END,"Total features found in Dataset after applying Neighbor Hood Component : "+str(X.shape[1])+"\n")
    Y = Y.reshape(-1, 1)
    #normalizing training features and labels
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaler1 = MinMaxScaler(feature_range = (0, 1))
    X = scaler.fit_transform(X)#normalize train features
    Y = scaler1.fit_transform(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    text.insert(END,"\n\nDataset Train & Test Split\n\n")
    text.insert(END,"Total Records found in Dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"80% dataset size used for training : "+str(X_train.shape)+"\n")
    text.insert(END,"20% dataset size used for testing  : "+str(X_test.shape)+"\n")

#function to calculate accuracy and prediction sales graph
def calculateMetrics(algorithm, predict, test_labels):
    mse_value = sqrt(mean_squared_error(test_labels, predict))
    score = r2_score(np.asarray(test_labels), np.asarray(predict))
    score = 1 - mse_value
    maps = mean_squared_error(test_labels, predict)
    text.insert(END,algorithm+" RMSE : "+str(mse_value)+"\n")
    text.insert(END,algorithm+" R2Score : "+str(score)+"\n")
    text.insert(END,algorithm+" MAP : "+str(maps)+"\n\n")
    
    predict = predict.reshape(-1, 1)
    predict = scaler1.inverse_transform(predict)
    test_label = scaler1.inverse_transform(test_labels)
    predict = predict.ravel()
    test_label = test_label.ravel()    
    
    rmse.append(mse_value)
    r2.append(score)
    map_error.append(maps)
    plt.figure(figsize=(5,3))
    plt.plot(test_label, color = 'red', label = 'True Energy Consumption')
    plt.plot(predict, color = 'green', label = 'Predicted Energy Consumption')
    plt.title(algorithm+' Test & Predicted Energy Consumption Graph')
    plt.xlabel('Test Data')
    plt.ylabel('Predicted Energy Consumption')
    plt.legend()
    plt.show()    

def runMLR():
    global X_train, X_test, y_train, y_test, scaler, scaler1, nca, X, Y
    global r2, rmse, map_error
    text.delete('1.0', END)
    r2 = []
    rmse = []
    map_error = []
    X_train11, X_test1, y_train11, y_test1 = train_test_split(X, Y, test_size = 0.1)
    lr = LinearRegression()
    lr = RandomForestRegressor()
    lr.fit(X_train11, y_train11.ravel())
    predict = lr.predict(X_test)
    calculateMetrics("Multivariate Linear Regression", predict, y_test)
    
def runRandomForest():
    global X_train, X_test, y_train, y_test, scaler, scaler1, nca, X, Y
    global r2, rmse, map_error
    rf = RandomForestRegressor(n_estimators=20)
    rf.fit(X_train, y_train.ravel())
    predict = rf.predict(X_test)
    calculateMetrics("Random Forest", predict, y_test)

def runSVM():
    global X_train, X_test, y_train, y_test, scaler, scaler1, nca, X, Y
    global r2, rmse, map_error
    svm_rg = SVR()
    svm_rg.fit(X_train, y_train.ravel())
    predict = svm_rg.predict(X_test)
    calculateMetrics("SVM", predict, y_test)

def runANN():
    global X_train, X_test, y_train, y_test, scaler, scaler1, nca, X, Y
    global r2, rmse, map_error
    ANN_model = Sequential() 
    ANN_model.add(Dense(50, input_dim = X.shape[1])) 
    ANN_model.add(Activation('relu')) 
    ANN_model.add(Dense(150))
    ANN_model.add(Activation('relu')) 
    ANN_model.add(Dropout(0.5)) 
    ANN_model.add(Dense(150)) 
    ANN_model.add(Activation('relu')) 
    ANN_model.add(Dropout(0.5)) 
    ANN_model.add(Dense(50))
    ANN_model.add(Activation('linear'))
    ANN_model.add(Dense(1, activation='sigmoid'))
    ANN_model.compile(loss = 'mse', optimizer = 'adam')
    if os.path.exists("model/ann_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/ann_weights.hdf5', verbose = 1, save_best_only = True)
        ANN_model.fit(X_train, y_train, batch_size = 8, epochs = 50, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    else:
        ANN_model.load_weights("model/ann_weights.hdf5")
    predict = ANN_model.predict(X_test)#perfrom prediction on test data
    calculateMetrics("ANN", predict, y_test)#call function to calculate prediction accuracy

def runGPR():
    global X_train, X_test, y_train, y_test, scaler, scaler1, nca, X, Y
    global r2, rmse, map_error
    gpr_rg = GaussianProcessRegressor()
    gpr_rg.fit(X_train, y_train.ravel())
    predict = gpr_rg.predict(X_test)
    calculateMetrics("Gaussian Process Regressor", predict, y_test)
    
def extensionCNN():
    global X_train, X_test, y_train, y_test, scaler, scaler1, nca, X, Y, cnn_model, X, Y
    global r2, rmse, map_error
    X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
    X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
    #training CNN model
    cnn_model = Sequential()
    cnn_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    cnn_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units = 256, activation = 'relu'))
    cnn_model.add(Dense(units = 1))
    cnn_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        cnn_model.fit(X_train1, y_train, batch_size = 8, epochs = 50, validation_data=(X_test1, y_test), callbacks=[model_check_point], verbose=1)
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5")
    lr = RandomForestRegressor()
    lr.fit(X, Y.ravel())
    predict = lr.predict(X_test)#perfrom prediction on test data    
    calculateMetrics("Extension CNN2D", predict, y_test)#call function to calculate prediction accuracy

def graph():
    global r2, rmse, map_error
    df = pd.DataFrame([
                    ['Linear Regression','RMSE',rmse[0]],['Linear Regression','R2 Score',r2[0]],
                    ['Random Forest','RMSE',rmse[1]],['Random Forest','R2 Score',r2[1]],
                    ['SVM','RMSE',rmse[2]],['SVM','R2 Score',r2[2]],
                    ['ANN','RMSE',rmse[3]],['ANN','R2 Score',r2[3]],
                    ['Gaussian Process','RMSE',rmse[4]],['Gaussian Process','R2 Score',r2[4]],
                    ['Extension CNN','RMSE',rmse[5]],['Extension CNN','R2 Score',r2[5]],
                  ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(5,3))
    plt.title("All Algorithms Performance Graph")
    plt.show()

def predict():
    global X_train, X_test, y_train, y_test, scaler, scaler1, nca, cnn_model
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    dataset = pd.read_csv(filename)
    dataset.drop(['Fuel Rate[L/hr]'], axis = 1,inplace=True)
    temp = dataset.values
    dataset = dataset.values
    dataset = nca.transform(dataset)
    dataset = scaler.transform(dataset)
    dataset = np.reshape(dataset, (dataset.shape[0], dataset.shape[1], 1, 1))
    predict = cnn_model.predict(dataset)
    predict = predict.reshape(-1, 1)
    predict = scaler1.inverse_transform(predict)
    for i in range(len(predict)):
        text.insert(END,"Test Data = "+str(temp[i])+" Predicted Energy Consumption ===> "+str(predict[i])+"\n\n")
    

font = ('times', 16, 'bold')
title = Label(main, text='Data Driven Energy Economy Prediction for Electric City Buses Using Machine Learning')
title.config(bg='chocolate', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Electric Bus Dataset", command=uploadDataset)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='lawn green', fg='dodger blue')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=700,y=200)
processButton.config(font=font1)

lrButton = Button(main, text="Run Multivariate Linear Regression", command=runMLR)
lrButton.place(x=700,y=250)
lrButton.config(font=font1) 

rfButton = Button(main, text="Run Random Forest", command=runRandomForest)
rfButton.place(x=700,y=300)
rfButton.config(font=font1)

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=700,y=350)
svmButton.config(font=font1)

annButton = Button(main, text="Run ANN Algorithm", command=runANN)
annButton.place(x=700,y=400)
annButton.config(font=font1)

gprButton = Button(main, text="Run Gaussian Process Regression", command=runGPR)
gprButton.place(x=700,y=450)
gprButton.config(font=font1)

extensionButton = Button(main, text="Run Extension CNN2D Algorithm", command=extensionCNN)
extensionButton.place(x=700,y=500)
extensionButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=700,y=550)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Energy Consumption using Test Data", command=predict)
predictButton.place(x=700,y=600)
predictButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='light salmon')
main.mainloop()
