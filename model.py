#Medical Cost 
#https://www.kaggle.com/datasets/mirichoi0218/insurance?resource=download
#saul
import pandas as pd
import numpy as np 
from sklear.model_selection import train_test_split
from sklear.linear_model import LinearRegression
import statsmodels.api as sm 
from sklearn import preprocessing
import matplotlib.pyplot as plt 

insurance = pd.read_csv('./insurance.csv')
insurance = insurance[:-5]
