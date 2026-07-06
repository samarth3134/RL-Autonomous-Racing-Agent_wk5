import gymnasium as gym
import numpy as np


class CurvatureAwareReward(gym.Wrapper):
    def __init__(self, env, lookahead=5, friction_coef=1.8, gravity=9.8,
                 braking_power=1.0, speed_error_penalty=0.1, max_steer_delta=0.15,
                 wrong_way_penalty=0.5, wrong_way_limit=30):
        super().__init__(env)
        self.lookahead = lookahead
        self.friction_coef = friction_coef
        self.gravity = gravity
        self.braking_power = braking_power  # how effective braking is (tune this by feel)
        self.speed_error_penalty = speed_error_penalty
        self.max_steer_delta = max_steer_delta
        self.wrong_way_penalty = wrong_way_penalty
        self.wrong_way_limit = wrong_way_limit
        self.prev_action = np.zeros(3)
        self.wrong_way_counter = 0

    def _get_speed_and_velocity(self):
        car = self.env.unwrapped.car
        vel = car.hull.linearVelocity
        speed = np.sqrt(vel[0]**2 + vel[1]**2)
        return speed, np.array([vel[0], vel[1]])

    def _nearest_track_index(self, car_pos, track):
        positions = np.array([[t[2], t[3]] for t in track])
        dists = np.linalg.norm(positions - car_pos, axis=1)
        return int(np.argmin(dists)), positions

    def _curvature_at(self, positions, idx, n):
        p1 = positions[idx % n]
        p2 = positions[(idx + self.lookahead // 2) % n]
        p3 = positions[(idx + self.lookahead) % n]

        area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
        ab = np.linalg.norm(p2 - p1)
        bc = np.linalg.norm(p3 - p2)
        ca = np.linalg.norm(p1 - p3)
        denom = ab * bc * ca
        if denom < 1e-6:
            return 0.0
        return 4 * area / denom

    def _safe_target_speed(self, curvature):
        if curvature < 1e-6:
            return 100.0
        radius = 1.0 / curvature
        return np.sqrt(self.friction_coef * self.gravity * radius)

    def _track_direction(self, positions, idx, n):
        tangent = positions[(idx + 1) % n] - positions[idx]
        norm = np.linalg.norm(tangent)
        return tangent / norm if norm > 1e-6 else None

    def _required_brake_fraction(self, current_speed, target_speed, distance_to_corner):
        """
        Uses v^2 = v0^2 - 2*a*d to find the deceleration needed to hit
        target_speed exactly at the corner, then expresses that as a
        fraction of max available braking deceleration.
        """
        if current_speed <= target_speed or distance_to_corner < 1e-3:
            return 0.0

        required_decel = (current_speed**2 - target_speed**2) / (2 * distance_to_corner)
        max_decel = self.friction_coef * self.gravity * self.braking_power

        return float(np.clip(required_decel / max_decel, 0.0, 1.0))

    def step(self, action):
        raw_steer = action[0]
        clipped_steer = np.clip(
            raw_steer,
            self.prev_action[0] - self.max_steer_delta,
            self.prev_action[0] + self.max_steer_delta,
        )

        env = self.env.unwrapped
        car = env.car
        car_pos = np.array([car.hull.position[0], car.hull.position[1]])
        vel = car.hull.linearVelocity
        current_speed = np.sqrt(vel[0]**2 + vel[1]**2)

        track = env.track
        n = len(track)
        idx, positions = self._nearest_track_index(car_pos, track)
        curvature = self._curvature_at(positions, idx, n)
        target_speed = self._safe_target_speed(curvature)

        # Distance from car to the lookahead point (i.e. distance to the corner we're evaluating)
        corner_point = positions[(idx + self.lookahead) % n]
        distance_to_corner = np.linalg.norm(corner_point - car_pos)

        gas, brake = action[1], action[2]

        # Graduated braking: blend policy's own brake with physically-required brake
        required_brake = self._required_brake_fraction(current_speed, target_speed, distance_to_corner)
        brake = max(brake, required_brake)
        # Reduce throttle proportionally to how hard we need to brake, rather than cutting it fully
        gas = gas * (1.0 - required_brake)

        action = np.array([clipped_steer, gas, brake], dtype=np.float32)

        obs, reward, terminated, truncated, info = self.env.step(action)

        speed, velocity_vec = self._get_speed_and_velocity()
        overspeed = max(0.0, speed - target_speed)
        reward -= self.speed_error_penalty * min(overspeed ** 2, 50.0)

        steer_delta = abs(clipped_steer - self.prev_action[0])
        reward -= 0.05 * steer_delta

        track_dir = self._track_direction(positions, idx, n)
        if track_dir is not None and speed > 1.0:
            vel_dir = velocity_vec / (speed + 1e-6)
            alignment = np.dot(vel_dir, track_dir)
            if alignment < -0.3:
                reward -= self.wrong_way_penalty
                self.wrong_way_counter += 1
            else:
                self.wrong_way_counter = 0
            if self.wrong_way_counter > self.wrong_way_limit:
                terminated = True

        self.prev_action = action
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.prev_action = np.zeros(3)
        self.wrong_way_counter = 0
        return super().reset(**kwargs)


class RacingEnv(gym.Wrapper):
    def __init__(self, render_mode=None):
        super().__init__(gym.make("CarRacing-v3", render_mode=render_mode))
        self.env = CurvatureAwareReward(self.env)