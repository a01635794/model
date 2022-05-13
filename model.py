# Modelo de costos médicos individuales facturados por el seguro de salud
# https://www.kaggle.com/datasets/mirichoi0218/insurance?resource=download

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme(color_codes=True)

# Leer Archivo
insurance = pd.read_csv('./insurance.csv')
insurance = insurance[:-5]

def regionCond(x):
    if x == 'northeast':
        return 0.1
    elif x == 'southeast':
        return 0.2
    elif x == 'southwest':
        return 0.3
    elif x == 'northwest':
        return 0.4

# Cambiar a valores numéricos
insurance['smoker'] = insurance['smoker'].apply(lambda x: 1 if x=='yes' else 0)
insurance['sex'] = insurance['sex'].apply(lambda x: 1 if x=='male' else 0)
insurance['region'] = insurance['region'].apply(regionCond)
print(insurance.head())

# Establecer X y Y -- Multilineal
X = np.array(insurance.drop(['charges'], axis=1))
y = np.array(insurance[['charges']])

# Separar dataframe
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Elegir regresión, entrenar modelo y realizar predicciones
LR = LinearRegression()
LR.fit(x_train, y_train)
y_prediction = LR.predict(x_test)

# Imprimir resultados
olsmod = sm.OLS(y_test, sm.add_constant(x_test)).fit()
print(olsmod.summary())

print(insurance.head())

# Hacer gráficas para los datos
graphInfo = pd.DataFrame()
graphInfo['y_test'] = y_test.flatten()
graphInfo['y_prediction'] = y_prediction.flatten()

predicted = sns.lmplot(x='y_test', y='y_prediction', data=graphInfo)
predicted.fig.suptitle('Test Value vs Prediction')
predicted.fig.show()

# Predecir perfil
print(' Profile Calculator '.center(40, '='))
age = int(input('Age: '))
sex = int(input('Male(1) - Female(0): '))
bmi = float(input('BMI: '))
children = int(input('Children: '))
smoker = int(input('Smoker yes(1) - no(0): '))
print('''   Northeast(0.1)
   Southeast(0.2)
   Southwest(0.3)
   Northwest(0.4)''')
region = float(input('Region: '))

print(f'Medical Cost: ${LR.predict([[age,sex,bmi,children,smoker,region]])[0][0]}')