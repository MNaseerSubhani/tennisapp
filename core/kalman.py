

from filterpy.kalman import KalmanFilter
import numpy as np

class KalmanManager:
    def __init__(self):
        self.filters = {}
        self.velocities = {}  # To track velocities for movement detection

    def _create(self, x, y):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1,0,1,0],
                         [0,1,0,1],
                         [0,0,1,0],
                         [0,0,0,1]])
        kf.H = np.array([[1,0,0,0],
                         [0,1,0,0]])
        kf.P *= 500  # Reduced initial uncertainty
        kf.R *= 2    # Reduced measurement noise
        kf.Q *= 0.05 # Reduced process noise for smoother tracking
        kf.x = np.array([x, y, 0, 0])
        return kf

    def update(self, track_id, bbox):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        if track_id not in self.filters:
            self.filters[track_id] = self._create(cx, cy)
            self.velocities[track_id] = [0, 0]
        kf = self.filters[track_id]
        kf.predict()
        kf.update([cx, cy])
        self.velocities[track_id] = kf.x[2:].tolist()  # Update velocity
        return kf.x[:2].tolist()
    
    def predict(self, track_id):
        kf = self.filters[track_id]
        kf.predict()
        self.velocities[track_id] = kf.x[2:].tolist()
        return kf.x[:2].tolist()

    def get_velocity_magnitude(self, track_id):
        if track_id in self.velocities:
            vx, vy = self.velocities[track_id]
            return np.sqrt(vx**2 + vy**2)
        return 0

    def get_velocity(self, track_id):
        if track_id in self.velocities:
            return self.velocities[track_id]
        return [0, 0]

