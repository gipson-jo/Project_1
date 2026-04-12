"""
ekf.py
Extended Kalman Filter for GPS-IMU fusion.

State vector (9 elements):
    [x, y, z,           <- position (metres)
     vx, vy, vz,        <- velocity (m/s)
     roll, pitch, yaw]  <- orientation (radians)

Predict step: uses IMU (runs every timestep)
Update step:  uses GPS (skipped during outage experiments)
"""

import numpy as np
from scipy.spatial.transform import Rotation


class EKF:

    def __init__(self):

        # ── State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw] ──────────
        self.state = np.zeros(9)

        # ── P: state covariance — how uncertain are we right now ─────────────
        # Start small (we know where we are at t=0)
        self.P = np.eye(9) * 0.1

        # ── Q: process noise — how much we distrust the IMU per second ───────
        # Larger = assume IMU is noisier = P grows faster during outage
        # These values are tunable — start with reasonable defaults
        self.Q = np.diag([
            0.01, 0.01, 0.01,    # position noise
            0.1,  0.1,  0.1,     # velocity noise  (accel bias)
            0.01, 0.01, 0.01     # orientation noise (gyro bias)
        ])

        # ── R: measurement noise — how much we distrust the GPS ──────────────
        # KITTI OXTS is accurate to ~2cm but we use conservative values
        self.R = np.diag([
            1.0, 1.0, 1.0        # GPS x, y, z noise (metres^2)
        ])

        # ── H: measurement matrix — GPS only observes position (first 3) ─────
        self.H = np.zeros((3, 9))
        self.H[0, 0] = 1.0   # x
        self.H[1, 1] = 1.0   # y
        self.H[2, 2] = 1.0   # z

        # ── Gravity in ENU world frame ────────────────────────────────────────
        self.gravity = np.array([0.0, 0.0, -9.81])

        # ── Current orientation as Rotation object ────────────────────────────
        self.orientation = Rotation.identity()

        # ── History for plotting ──────────────────────────────────────────────
        self.position_history    = []
        self.orientation_history = []
        self.covariance_history  = []   # track P trace over time (uncertainty)

    # ────────────────────────────────────────────────────────────────────────
    def predict(self, accel, gyro, dt):
        """
        Predict step — integrates IMU forward one timestep.
        Identical math to dead reckoning but also propagates uncertainty P.

        Args:
            accel: (3,) accelerometer reading in body frame (m/s^2)
            gyro:  (3,) gyroscope reading in body frame (rad/s)
            dt:    time since last step (seconds)
        """

        # ── Unpack current state ─────────────────────────────────────────────
        pos = self.state[0:3]
        vel = self.state[3:6]

        # ── Step 1: Update orientation from gyro (same as dead reckoning) ────
        angle = np.linalg.norm(gyro) * dt
        if angle > 1e-10:
            axis = gyro / np.linalg.norm(gyro)
            delta_rot = Rotation.from_rotvec(axis * angle)
            self.orientation = self.orientation * delta_rot

        # ── Step 2: Rotate accel to world frame, remove gravity ───────────────
        accel_world = self.orientation.apply(accel) - self.gravity

        # ── Step 3: Integrate → velocity → position ───────────────────────────
        new_vel = vel + accel_world * dt
        new_pos = pos + vel * dt + 0.5 * accel_world * dt**2

        # ── Step 4: Update state vector ───────────────────────────────────────
        self.state[0:3] = new_pos
        self.state[3:6] = new_vel
        self.state[6:9] = self.orientation.as_euler('xyz')

        # ── Step 5: Propagate uncertainty P ──────────────────────────────────
        # Build state transition Jacobian F
        F = self._build_F(accel_world, dt)

        # P grows each predict step due to IMU noise Q
        self.P = F @ self.P @ F.T + self.Q * dt

        # ── Log state ────────────────────────────────────────────────────────
        self.position_history.append(self.state[0:3].copy())
        self.orientation_history.append(self.state[6:9].copy())
        self.covariance_history.append(np.trace(self.P))  # scalar uncertainty

    # ────────────────────────────────────────────────────────────────────────
    def update(self, gps_position):
        """
        Update step — corrects state using GPS position fix.
        Call this every timestep when GPS is available.
        Skip it entirely during GPS outage experiments.

        Args:
            gps_position: (3,) GPS position [x, y, z] in metres
        """

        # ── Innovation: difference between GPS and our prediction ─────────────
        z_pred = self.H @ self.state          # what we predicted GPS would see
        z_meas = gps_position                 # what GPS actually sees
        innovation = z_meas - z_pred          # the error we need to correct

        # ── Kalman gain K ─────────────────────────────────────────────────────
        # K decides how much to trust GPS vs our IMU prediction
        # K = P H^T (H P H^T + R)^-1
        S = self.H @ self.P @ self.H.T + self.R   # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain (9x3)

        # ── Correct the state ─────────────────────────────────────────────────
        self.state = self.state + K @ innovation

        # ── Shrink P — we just got new information so we're more certain ──────
        I = np.eye(9)
        self.P = (I - K @ self.H) @ self.P

        # Update orientation object to match corrected state
        self.orientation = Rotation.from_euler('xyz', self.state[6:9])

    # ────────────────────────────────────────────────────────────────────────
    def _build_F(self, accel_world, dt):
        """
        Build the state transition Jacobian F.
        Describes how small errors in state propagate forward one timestep.

        F is a 9x9 matrix:
            position    depends on velocity      → F[0:3, 3:6] = I*dt
            velocity    depends on acceleration  → F[3:6, 6:9] = accel terms
            orientation depends on itself        → F[6:9, 6:9] = I
        """
        F = np.eye(9)

        # Position changes with velocity
        F[0:3, 3:6] = np.eye(3) * dt

        # Velocity changes with orientation errors (simplified)
        # Full derivation would use skew-symmetric matrix of accel
        # This linearization is sufficient for our outage durations
        ax, ay, az = accel_world
        F[3, 7] =  az * dt   # vx affected by pitch error
        F[3, 8] = -ay * dt   # vx affected by yaw error
        F[4, 6] = -az * dt   # vy affected by roll error
        F[4, 8] =  ax * dt   # vy affected by yaw error
        F[5, 6] =  ay * dt   # vz affected by roll error
        F[5, 7] = -ax * dt   # vz affected by pitch error

        return F

    # ────────────────────────────────────────────────────────────────────────
    def get_position(self):
        return self.state[0:3].copy()

    def get_position_history(self):
        return np.array(self.position_history)

    def get_covariance_history(self):
        return np.array(self.covariance_history)
