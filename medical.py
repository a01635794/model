
#insurance['children'] = preprocessing.maxabs_scale(insurance[['children']])

print(insurance.head())

# Establecer X y Y -- Multilineal
X = np.array(insurance.drop(['charges'], axis=1))
y = np.array(insurance[['charges']])

# Separar dataframe
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
