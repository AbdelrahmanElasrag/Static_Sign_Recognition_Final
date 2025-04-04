#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import csv

class KeyPointClassifier:
    def __init__(self, model_path, label_path):
        # Load the Keras model
        self.model = tf.keras.models.load_model(model_path)

        # Load class labels from the CSV file
        with open(label_path, encoding="utf-8-sig") as f:
            self.labels = [row[0] for row in csv.reader(f)]

    def __call__(self, landmark_list):
        # Perform inference with the Keras model
        input_data = np.array([landmark_list], dtype=np.float32)  # Prepare input
        predictions = self.model.predict(input_data)  # Get model predictions

        # Get the index of the highest predicted value (class index)
        result_index = np.argmax(np.squeeze(predictions))

        return result_index, self.labels[result_index]  # Return both index and label
