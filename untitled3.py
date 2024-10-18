import csv
from datetime import datetime
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

import matplotlib.pyplot as plt

def log_car_count(count):
    with open('car_counts.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), count])



# Load the data
data = pd.read_csv('car_counts.csv', header=None, names=['timestamp', 'car_count'])

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Calculate average congestion
average_congestion = data['car_count'].mean()
print(f'Average congestion: {average_congestion}')

# Calculate moving average
data['moving_average'] = data['car_count'].rolling(window=5).mean()  # 5-cycle moving average



# Prepare data
data['cycle'] = np.arange(len(data))  # Create a cycle index
X = data[['cycle']]  # Features
y = data['car_count']  # Target

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Print predictions
for actual, predicted in zip(y_test, predictions):
    print(f'Actual: {actual}, Predicted: {predicted}')

plt.plot(data['timestamp'], data['car_count'], label='Car Count')
plt.plot(data['timestamp'], data['moving_average'], label='Moving Average', color='orange')
plt.xlabel('Time')
plt.ylabel('Number of Cars')
plt.legend()
plt.show()