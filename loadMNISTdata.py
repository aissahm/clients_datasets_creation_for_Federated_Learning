import numpy as np
from tensorflow import keras
import dataloader

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

number_clients = 100
dominant_class_percentage = 0.5

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

clients_datasets_obj = dataloader.returnClientDatasetsNonIIDdata(y_train, number_clients, dominant_class_percentage)

