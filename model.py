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

age = 50
sex = 1
bmi = 30.97
children = 3
smoker = 0
region = 0.4

def regionCond(x):
    if x == 'northeast':
        return 0.1
    elif x == 'southeast':
        return 0.2
    elif x == 'southwest':
        return 0.3
    elif x == 'northwest':
        return 0.4

insurance['smoker'] = insurance['smoker'].apply(lambda x: 1 if x=='yes' else 0)
insurance['sex'] = insurance['sex'].apply(lambda x: 1 if x=='male' else 0)
insurance['region'] = insurance['region'].apply(regionCond)
