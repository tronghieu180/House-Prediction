import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers

train_data = pd.read_csv('X_train.csv')
y_train_data = pd.read_csv('Y_train.csv')  # Load target data
test_data = pd.read_csv('X_test.csv')

train_data['tradeTime'] = pd.to_datetime(train_data['tradeTime'], errors='coerce')


print(train_data.head())
print(train_data.describe())
print(train_data.isnull().sum().sort_values(ascending=False))


plt.figure(figsize=(12, 8))
sns.heatmap(train_data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.show()


train_data.fillna(train_data.median(), inplace=True)
test_data.fillna(test_data.median(), inplace=True)


train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)


train_data, test_data = train_data.align(test_data, join='left', axis=1, fill_value=0)


X = train_data  # Use all features for training
y = y_train_data['target']  # Use target from Y_train
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])


model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))


val_predictions = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f'Validation RMSE: {val_rmse}')


test_predictions = model.predict(test_data)

# Step 8: Prepare the Submission File
submission = pd.DataFrame({
    'Id': y_train_data['Id'],  # Use Id from Y_train for submission
    'target': test_predictions.flatten()  # Flatten to make it a 1D array
})

submission.to_csv('submission.csv', index=False)
print("Submission file created.")
