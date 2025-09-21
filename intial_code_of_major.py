from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
import os
# Create the main window
main = Tk()
main.title("Electric Bus Dataset Prediction")
main.geometry("1000x700")

# Title Label
ont = ('times', 16, 'bold')
title = Label(main, text='Data Driven Energy Economy Prediction for Electric City Buses Using Machine Learning')
title.config(bg='chocolate', fg='white')  
title.config(font=ont)           
title.config(height=3, width=120)       
title.place(x=0, y=5)

# Button font configuration
font1 = ('times', 13, 'bold')

# Upload Dataset Button
upload = Button(main, text="Upload Electric Bus Dataset")
upload.place(x=700, y=100)
upload.config(font=font1)

# Path Label
pathlabel = Label(main)
pathlabel.config(bg='lawn green', fg='dodger blue')  
pathlabel.config(font=font1)           
pathlabel.place(x=700, y=150)

# Preprocess Dataset Button
processButton = Button(main, text="Preprocess Dataset")
processButton.place(x=700, y=200)
processButton.config(font=font1)

# Multivariate Linear Regression Button
lrButton = Button(main, text="Run Multivariate Linear Regression")
lrButton.place(x=700, y=250)
lrButton.config(font=font1)

# Random Forest Button
rfButton = Button(main, text="Run Random Forest")
rfButton.place(x=700, y=300)
rfButton.config(font=font1)

# SVM Algorithm Button
svmButton = Button(main, text="Run SVM Algorithm")
svmButton.place(x=700, y=350)
svmButton.config(font=font1)

# ANN Algorithm Button
annButton = Button(main, text="Run ANN Algorithm")
annButton.place(x=700, y=400)
annButton.config(font=font1)

# Gaussian Process Regression Button
gprButton = Button(main, text="Run Gaussian Process Regression")
gprButton.place(x=700, y=450)
gprButton.config(font=font1)

# Extension CNN2D Algorithm Button
extensionButton = Button(main, text="Run Extension CNN2D Algorithm")
extensionButton.place(x=700, y=500)
extensionButton.config(font=font1)

# Comparison Graph Button
graphButton = Button(main, text="Comparison Graph")
graphButton.place(x=700, y=550)
graphButton.config(font=font1)

# Predict Energy Consumption Button
predictButton = Button(main, text="Predict Energy Consumption using Test Data")
predictButton.place(x=700, y=600)
predictButton.config(font=font1)

# Text Widget for displaying data
font1 = ('times', 12, 'bold')
text = Text(main, height=30, width=80)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=100)
text.config(font=font1)

# Run the Tkinter event loop
main.config(bg='light salmon')
main.mainloop()