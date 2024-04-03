import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv

class KalmanFilter:
    def __init__(self, F, H, Q, R):
        self.F = F  # State transition matrix
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = np.eye(F.shape[0])  # Initial state covariance
        self.x = np.zeros((F.shape[0], 1))  # Initial state

    def predict(self):
        # Predict state and covariance
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        # Update step
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Compute association probabilities
        association_probabilities = self.compute_association_probabilities(S)

        # Update state based on association probabilities
        predicted_states = []
        for i, prob in enumerate(association_probabilities):
            K = np.dot(np.dot(self.P, self.H.T), inv(S))
            x_i = self.x + np.dot(K, y)
            self.P = self.P - np.dot(np.dot(K, self.H), self.P)
            predicted_states.append(x_i)
        
        # Select the predicted state with highest association probability
        max_prob_index = np.argmax(association_probabilities)
        self.x = predicted_states[max_prob_index]

    def compute_association_probabilities(self, S):
        # Compute association probabilities using Mahalanobis distance
        num_measurements = self.H.shape[0]
        det_S = np.linalg.det(S)
        inv_S = np.linalg.inv(S)
        association_probabilities = np.zeros(num_measurements)
        for i in range(num_measurements):
            d = np.dot(np.dot((self.H[i] - np.dot(self.H[i], self.x)).T, inv_S), (self.H[i] - np.dot(self.H[i], self.x)))
            association_probabilities[i] = np.exp(-0.5 * d) / ((2 * np.pi) ** (self.H.shape[0] / 2) * np.sqrt(det_S))
        return association_probabilities

def main():
    # Read data from CSV file, skip first row
    data = pd.read_csv("test.csv", header=None, skiprows=1)

    # Extract target ID column
    target_ids = data[0].unique()

    # Define state transition matrix
    F = np.eye(4)  # Assume constant velocity model for simplicity

    # Define measurement matrix
    H = np.eye(4)  # Identity matrix since measurement directly reflects state

    # Define process noise covariance matrix
    Q = np.eye(4) * 0.1  # Process noise covariance

    # Define measurement noise covariance matrix
    R = np.eye(4) * 0.01  # Measurement noise covariance, adjusted variance

    # Initialize Kalman filter for each target ID
    kalman_filters = {}
    for target_id in target_ids:
        kalman_filters[target_id] = KalmanFilter(F, H, Q, R)

    # Lists to store predicted values for all variables
    predicted_ranges = {}
    predicted_azimuths = {}
    predicted_elevations = {}
    predicted_times = {}

    # Predict and update for each measurement
    for _, row in data.iterrows():
        target_id = row[0]
        measurement = row[1:].values.astype(float)  # Convert values to float

        # Predict
        kalman_filters[target_id].predict()

        # Update with measurement
        kalman_filters[target_id].update(measurement[:4])  # Consider only the first four values (Range, Azimuth, Elevation, Time)

        # Get predicted state
        predicted_state = kalman_filters[target_id].x.squeeze()

        # Append predicted values for all variables
        if target_id not in predicted_ranges:
            predicted_ranges[target_id] = []
            predicted_azimuths[target_id] = []
            predicted_elevations[target_id] = []
            predicted_times[target_id] = []
        predicted_ranges[target_id].append(predicted_state[0])
        predicted_azimuths[target_id].append(predicted_state[1])
        predicted_elevations[target_id].append(predicted_state[2])
        predicted_times[target_id].append(measurement[3])  # Assuming Time is in the fourth column

    # Plotting
    for target_id in target_ids:
        plt.figure(figsize=(8, 6))

        # Plot measured and predicted range against time
        plt.plot(predicted_times[target_id], data[data[0] == target_id][1], label='Measured Range', marker='o')
        plt.plot(predicted_times[target_id], predicted_ranges[target_id], label='Predicted Range', linestyle='--', marker='o')
        plt.xlabel('Time')
        plt.ylabel('Range')
        plt.title(f'Range Measurement and Prediction vs. Time for Target ID: {target_id}')
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 6))

        # Plot measured and predicted azimuth against time
        plt.plot(predicted_times[target_id], data[data[0] == target_id][2], label='Measured Azimuth', marker='o')
        plt.plot(predicted_times[target_id], predicted_azimuths[target_id], label='Predicted Azimuth', linestyle='--', marker='o')
        plt.xlabel('Time')
        plt.ylabel('Azimuth')
        plt.title(f'Azimuth Measurement and Prediction vs. Time for Target ID: {target_id}')
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 6))

        # Plot measured and predicted elevation against time
        plt.plot(predicted_times[target_id], data[data[0] == target_id][3], label='Measured Elevation', marker='o')
        plt.plot(predicted_times[target_id], predicted_elevations[target_id], label='Predicted Elevation', linestyle='--', marker='o')
        plt.xlabel('Time')
        plt.ylabel('Elevation')
        plt.title(f'Elevation Measurement and Prediction vs. Time for Target ID: {target_id}')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
