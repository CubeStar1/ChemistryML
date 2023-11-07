import os
import rdkit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem  # To extract information of the molecules
from rdkit.Chem import Draw  # To draw the molecules
from mordred import Calculator, descriptors  # To calculate descriptors
from molml.features import CoulombMatrix  # To calculate the coulomb matrix
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ReduceLROnPlateau

# Reading the dataset
df = pd.read_csv('qm9.csv')


# Using a descriptor that considers the 3D structure of the molecules
calc = Calculator(descriptors, ignore_3D=True)

# Using only the molecules with 19 atoms
df_desc = calc.pandas(df['mol'][df['num_of_atoms']==19])

# Saving the resulting file
df_desc.to_csv('descriptors.csv')

df_desc = df_desc.select_dtypes(include=np.number).astype('float32')

# Removing columns with variance = 0
df_desc = df_desc.loc[:, df_desc.var() > 0.0]

# Normalizing the descriptors
df_descN = pd.DataFrame(MinMaxScaler().fit_transform(df_desc), columns = df_desc.columns)

# Selecting the initial properties for molecules with only 19 atoms
df_19 = df[df['num_of_atoms']==19]

# Testing with the "mu" (dipole moment) property
x_train, x_test, y_train, y_test = train_test_split(df_descN, df_19['mu'],
                                                    test_size=0.2, random_state=42)

# Creating the model

def neural_model(x, y, x_test, y_test, neurons):
    """
    Neural network model

    Inputs
    x: descriptors values for training and validation
    y: properties values for training and validation
    x_test: descriptors values for test
    y_test: properties values for test


    Outputs
    model: trained neural network model
    score: a list with the score values for each fold
    """
    np.random.seed(1)
    score = []
    kfold = KFold(n_splits=5, shuffle=True)

    model = Sequential()
    model.add(Dense(neurons, input_dim=x.shape[1], activation='relu'))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='linear'))

    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['mean_absolute_error'])

    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=10)

    for train, validation in kfold.split(x, y):
        model.fit(x.iloc[train], y.iloc[train],
                  epochs=100,
                  batch_size=128,
                  callbacks=[rlrop],
                  verbose=0,
                  validation_data=(x.iloc[validation], y.iloc[validation]))

        score.append(model.evaluate(x_test, y_test))

    return model, score

model, score = neural_model(x_train, y_train, x_test, y_test, neurons=64)

print(f'mse: {np.mean(score):.3f} \u00B1 {np.std(score):.3f} ')