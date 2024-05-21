#!/usr/bin/env /opt/conda/bin/python
# Ensure compatibility
#from __future__ import absolute_import, division, print_function

###########################################################################
print('Content-Type:text/html') #HTML is following
print("")                          #Leave a blank line

###########################################################################

# Ensure compatibility
#from __future__ import absolute_import, division, print_function

# Import packages
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

# Import TensorFlow and Keras packages
import tensorflow as tf
from tensorflow import keras

priority_labels= ['Low', 'Medium', 'High', 'Critical']

# Load saved model
new_model = tf.keras.models.load_model("Models")

###########################################################################

import cgi
import cgitb
cgitb.enable()

input_data=cgi.FieldStorage()

print('<h1>','Prediction Results','</h1>')
try:
  QuantityNormFloat = float(input_data["QuantityNorm"].value)
  SalesNormFloat = float(input_data["SalesNorm"].value)
  ShippingCostNormFloat = float(input_data["ShippingCostNorm"].value)
except:
  print('<p>Sorry, we cannot turn your inputs into numbers (floats).</p>')

print('<p>',QuantityNormFloat, SalesNormFloat, ShippingCostNormFloat,'</p>')

#QunatityNorm, SalesNorm, ShippingCostsNorm
#Each ranges from 0 to 1
SingleObservation = np.array([[QuantityNormFloat, SalesNormFloat, ShippingCostNormFloat]])
SingleObservationFloat = SingleObservation.astype(float)

SinglePredictionNewModel = new_model.predict(SingleObservationFloat)

# Print prediction
print('<p>',SinglePredictionNewModel[0],'</p>')

print('<p>',priority_labels[np.argmax(SinglePredictionNewModel[0])],'</p>')

print('<h1>','Results are complete','</h1>')




