import numpy as np

class KalmanFilter:
    def __init__(self, dt, process_var, measurement_var, gravity):
        # State: [x, y, z, vx, vy, vz]
        self.dt = dt
        self.gravity = gravity
        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i+3] = dt

        self.H = np.zeros((3,6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        self.Q = process_var * np.eye(6)        # Process noise
        self.R = measurement_var * np.eye(3)    # Measurement noise

        self.x = np.zeros((6,1))    # Initial state
        self.p = np.eye(6)          # Initial covariance

    def measurementUpdate(self, meas):
        meas = np.reshape(meas, (3,1))
        epsilon = meas - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ epsilon
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def timeUpdate(self):
        self.x = self.F @ self.x
        self.x[5, 0] -= self.gravity * self.dt # Include effect of gravity
        self.p = self.F @ self.P @ self.F.T + self.Q

    def predict(self, nr_predictions):
        x = np.array([self.x])
        for _ in range(nr_predictions):
            temp = self.F @ x[-1]
            temp[5, 0] -= self.gravity * self.dt
            x.append(x, temp)
        return x

    def getState(self):
        return self.x.flatten()