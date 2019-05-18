#!/usr/bin/env python
# coding: utf-8

# # Train and deploy on Kubeflow from Notebooks
# 
# This notebook introduces you to using Kubeflow Fairing to train and deploy a model to Kubeflow on Google Kubernetes Engine (GKE), and Google Cloud ML Engine. This notebook demonstrate how to:
#  
# * Train an XGBoost model in a local notebook,
# * Use Kubeflow Fairing to train an XGBoost model remotely on Kubeflow,
#   * Data is read from a PVC
#   * The append builder is used to rapidly build a docker image
# * Use Kubeflow Fairing to deploy a trained model to Kubeflow, and
# * Call the deployed endpoint for predictions.
# 
# To learn more about how to run this notebook locally, see the guide to [training and deploying on GCP from a local notebook][gcp-local-notebook].
# 
# [gcp-local-notebook]: https://kubeflow.org/docs/fairing/gcp-local-notebook/

# ## Set up your notebook for training an XGBoost model
# 
# Import the libraries required to train this model.

# In[1]:





# In[2]:





# In[3]:


# fairing:include-cell
import ames
import argparse
import fire
import logging
import nbconvert
import os
import joblib
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from importlib import reload


# In[4]:





# In[5]:





# In[6]:





# In[7]:





# In[8]:





# In[9]:


# fairing:include-cell
class HousingServe(object):    
    def __init__(self, model_file=None):
        self.n_estimators = 50
        self.learning_rate = 0.1
        if not model_file:
            print("model_file not supplied; checking environment variable")
            model_file = os.getenv("MODEL_FILE")
        
        self.model_file = model_file
        print("model_file={0}".format(self.model_file))
        
        self.model = None
                

    def train(self, train_input, model_file):
        (train_X, train_y), (test_X, test_y) = ames.read_input(train_input)
        model = ames.train_model(train_X,
                                 train_y,
                                 test_X,
                                 test_y,
                                 self.n_estimators,
                                 self.learning_rate)

        ames.eval_model(model, test_X, test_y)
        ames.save_model(model, model_file)

    def predict(self, X, feature_names):
        """Predict using the model for given ndarray."""
        if not self.model:
            print("Loading model {0}".format(self.model_file))
            self.model = joblib.load(self.model_file)
        # Do any preprocessing
        prediction = self.model.predict(data=X)
        # Do any postprocessing
        return [[prediction.item(0), prediction.item(0)]]


# ## Train your Model Locally
# 
# * Train your model locally inside your notebook

# In[10]:





# ## Use Fairing to Launch a K8s Job to train your model

# ### Set up Kubeflow Fairing for training and predictions
# 
# Import the `fairing` library and configure the environment that your training or prediction job will run in.

# In[11]:





# ## Use fairing to build the docker image
# 
# * This uses the append builder to rapidly build docker images

# In[12]:





# In[13]:





# ## Launch the K8s Job
# 
# * Use pod mutators to attach a PVC and credentials to the pod

# In[14]:





# In[15]:





# ## Deploy the trained model to Kubeflow for predictions

# In[17]:





# In[22]:





# ## Call the prediction endpoint
# 
# Create a test dataset, then call the endpoint on Kubeflow for predictions.

# In[23]:





# In[24]:





# ## Clean up the prediction endpoint
# 
# Delete the prediction endpoint created by this notebook.

# In[ ]:






if __name__ == "__main__":
  import fire
  import logging
  logging.basicConfig(format='%(message)s')
  logging.getLogger().setLevel(logging.INFO)
  fire.Fire(HousingServe)
