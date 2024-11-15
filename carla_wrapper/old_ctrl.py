import math
import time
from collections import deque

import numpy as np

import cvxpy
from cvxpy.expressions import constants

from control import control_utils
from control.mpc_utils import Trajectory, Vehicle, compute_curvature
from control.mpc_utils import CubicSpline2D, global_config, zero_to_2_pi

from utils.logging import get_timestamp_logger, get_module_logger, ModuleCompletionLogger

from objects.objects import Location, Rotation, Transform, Vector3D, Waypoints

import params

class Controller():
    def __init__(self):
        self._config_name = "Controller"
        # Dump logs for Controller
        self._module_logger = get_module_logger(self._config_name)
        self._csv_logger = ModuleCompletionLogger()
        self._timestamp_logger = get_timestamp_logger()
        
        print("\nInitializing Controller...")
        self._mpc_config = global_config
        self._pid = PIDLongitudinalController(1.0, 0.0, 0.05, dt = 1.0 / params.simulator_fps)
        self._mps = None
        self.last_timestamp = -1

    def get_control_instructions(self, timestamp, pose, waypoints):
        self.last_timestamp = timestamp
        start_time = time.time()
        ego = pose.transform
        speed = pose.forward_speed

        if waypoints == None or len(waypoints.waypoints) < 3:
            print('Braking! No more waypoints to follow.')
            return 0, 0, 0.5, 0

        try:
            if params.controller_type == "pid":
                steer, throttle, brake = self.get_pid_control_instructions(ego, speed, waypoints)
            else:
                steer, throttle, brake = self.get_mpc_control_instructions(ego, speed, waypoints)
        except ValueError:
            print('Braking! No more waypoints to follow.')
            throttle, brake = 0.0, 0.5
            steer = 0.0
       
        return steer, throttle, brake, 1000 * (time.time() - start_time)

    def get_pid_control_instructions(self, vehicle, speed, waypoints):

        try:
            angle_steer = waypoints.get_angle(vehicle, params.min_pid_steer_waypoint_distance)
            target_speed = waypoints.get_target_speed(vehicle, params.min_pid_speed_waypoint_distance)
            throttle, brake = self.compute_throttle_and_brake(self._pid, speed, target_speed)
            steer = control_utils.radians_to_steer(angle_steer, params.steer_gain)
        except (ValueError, AttributeError):
            print('Braking! No more waypoints to follow.')
            throttle, brake = 0.0, 0.5
            steer = 0.0
        
        return steer, throttle, brake

    def get_mpc_control_instructions(self, vehicle, speed, waypoints):
        target_speeds = waypoints.target_speeds

        self.setup_mpc(waypoints, target_speeds)
        self._mpc.vehicle.x = vehicle.location.x
        self._mpc.vehicle.y = vehicle.location.y
        self._mpc.vehicle.yaw = np.deg2rad(
            zero_to_2_pi(vehicle.rotation.yaw))

        try:
            self._mpc.step()
        except Exception as e:
            return 0, 0, 1

        # Compute pid controls.
        target_speed = self._mpc.solution.vel_list[-1]
        target_steer_rad = self._mpc.horizon_steer[0]  # in rad
        steer = control_utils.radians_to_steer(target_steer_rad, params.steer_gain)
        throttle, brake = self.compute_throttle_and_brake(
            self._pid, speed, target_speed)
        return steer, throttle, brake
    
    def setup_mpc(self, waypoints, target_speeds):
        path = waypoints.as_numpy_array_2D()
        # convert target waypoints into spline
        spline = CubicSpline2D(path[0, :], path[1, :])
        ss = []
        vs = []
        xs = []
        ys = []
        yaws = []
        ks = []
        for i, s in enumerate(spline.s[:-1]):
            x, y = spline.calc_position(s)
            yaw = np.abs(spline.calc_yaw(s))
            k = spline.calc_curvature(s)
            xs.append(x)
            ys.append(y)
            yaws.append(yaw)
            ks.append(k)
            ss.append(s)
            vs.append(target_speeds[i])
        self._mpc_config["reference"] = {
            't_list': [],  # Time [s]
            's_list': ss,  # Arc distance [m]
            'x_list': xs,  # Desired X coordinates [m]
            'y_list': ys,  # Desired Y coordinates [m]
            'k_list': ks,  # Curvatures [1/m]
            'vel_list': vs,  # Desired tangential velocities [m/s]
            'yaw_list': yaws,  # Yaws [rad]
        }
        # initialize mpc controller
        self._mpc = ModelPredictiveController(config=self._mpc_config)

    def compute_throttle_and_brake(self, pid, current_speed: float, target_speed: float):
        """Computes the throttle/brake required to reach the target speed.

        It uses the longitudinal controller to derive the required information.

        Args:
            pid: The pid controller.
            current_speed (:obj:`float`): The current speed of the ego vehicle
                (in m/s).
            target_speed (:obj:`float`): The target speed to reach (in m/s).
            flags (absl.flags): The flags object.

        Returns:
            Throttle and brake values.
        """
        if current_speed < 0:
            non_negative_speed = 0
        else:
            non_negative_speed = current_speed
        acceleration = pid.run_step(target_speed, non_negative_speed)
        self._timestamp_logger.write("{} {} {}\n".format(self.last_timestamp, 'acceleration', acceleration))
        if acceleration >= 0.0:
            throttle = min(acceleration, params.throttle_max)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(acceleration), params.brake_max)
        # Keep the brake pressed when stopped or when sliding back on a hill.
        if (current_speed < 1 and target_speed == 0) or current_speed < -0.3:
            brake = 1.0
        return throttle, brake

class PIDLongitudinalController(object):
    """Implements longitudinal control using a PID.

    Args:
       K_P (:obj:`float`): Proportional term.
       K_D (:obj:`float`): Differential term.
       K_I (:obj:`float`): Integral term.
       dt (:obj:`float`): time differential in seconds.
    """
    def __init__(self,
                 K_P: float = 1.0,
                 K_D: float = 0.0,
                 K_I: float = 0.0,
                 dt: float = 0.03):
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._last_time = time.time()
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed: float, current_speed: float):
        """Computes the throttle/brake based on the PID equations.

        Args:
            target_speed (:obj:`float`): Target speed in m/s.
            current_speed (:obj:`float`): Current speed in m/s.

        Returns:
            Throttle and brake values.
        """
        # Transform to km/h
        error = (target_speed - current_speed) * 3.6
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return (self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie)


class PIDLateralController(object):
    """Implements lateral control using a PID.

    Args:
       K_P (:obj:`float`): Proportional term.
       K_D (:obj:`float`): Differential term.
       K_I (:obj:`float`): Integral term.
       dt (:obj:`float`): time differential in seconds.
    """
    def __init__(self,
                 K_P: float = 1.0,
                 K_D: float = 0.0,
                 K_I: float = 0.0,
                 dt: float = 0.03,
                 use_real_time: bool = False):
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._use_real_time = use_real_time
        self._last_time = time.time()
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint, vehicle_transform):
        v_begin = vehicle_transform.location
        v_end = v_begin + Location(
            x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
            y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([
            waypoint.location.x - v_begin.x, waypoint.location.y - v_begin.y,
            0.0
        ])
        _dot = math.acos(
            np.clip(
                np.dot(w_vec, v_vec) /
                (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)

        if _cross[2] < 0:
            _dot *= -1.0

        if self._use_real_time:
            time_now = time.time()
            dt = time_now - self._last_time
            self._last_time = time_now
        else:
            dt = self._dt

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / dt
            _ie = sum(self._e_buffer) * dt
        else:
            _de = 0.0
            _ie = 0.0

        return (self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie)


class ModelPredictiveController:
    def __init__(self, config):
        self.reference = Trajectory(**config['reference'])
        self.vehicle = Vehicle(config['vehicle'])
        self.path_length = len(self.reference.s_list)
        self.path_index = 0
        self.t = 0.0  # [s]
        initial_condition = {
            't_list': [self.t],  # Initial time [s]
            's_list': self.reference.s_list[0:1],  # Initial arc distance [m]
            'x_list': self.reference.x_list[0:1],  # Initial X coordinate [m]
            'y_list': self.reference.y_list[0:1],  # Initial Y coordinate [m]
            'k_list': self.reference.k_list[0:1],  # Initial curvature [1/m]
            'vel_list': self.reference.vel_list[0:1],  # Initial velocity [m/s]
            'yaw_list': self.reference.yaw_list[0:1],  # Initial yaw [rad]
            'accel_list': np.asarray([]),  # Initial acceleration [m/s2]
            'steer_list': np.asarray([]),  # Initial steering [rad]
        }
        self.solution = Trajectory(**initial_condition)
        self.vehicle.update(
            self.t,
            self.solution.s_list[-1],
            self.solution.x_list[-1],
            self.solution.y_list[-1],
            self.solution.k_list[-1],
            self.solution.vel_list[-1],
            self.solution.yaw_list[-1],
            None,
            None,
        )
        self.config = config['controller']
        self.num_state = 4  # Number of states: X = [x, y, vel, yaw]
        self.num_input = 2  # Number of inputs: U = [accel, steer]
        self.position = self.vehicle.get_position()
        self.goal = np.asarray(
            [self.reference.x_list[-1], self.reference.y_list[-1]])
        self.delta_t = 0.1  # [s]
        self.delta_s = self.reference.s_list[1] - self.reference.s_list[0]
        self.horizon_accel = np.zeros((self.config['horizon'], 1))  # [m/s2]
        self.horizon_steer = np.zeros((self.config['horizon'], 1))  # [rad]

    def step(self):
        """
        Solve the mpc problem and step the vehicle forward in time with solved
        controls.

        :return: None
        """
        self.t = self.t + self.delta_t
        # Solve control for the next step
        self._update_path_index()
        reference_state, reference_steer = \
            self._retrieve_imminent_reference()

        for _ in range(self.config['max_iteration']):
            is_converged = self._iterative_control(reference_state,
                                                   reference_steer)
            if is_converged:
                break

        # Update vehicle
        t = self.t
        s = self.path_index
        accel = self.horizon_accel[0]
        steer = self.horizon_steer[0]
        state = self._step_state(self.vehicle.get_state(), accel, steer)
        x = state[0]
        y = state[1]
        vel = state[2]
        yaw = state[3]
        k = compute_curvature(vel, accel, yaw)
        self.vehicle.update(t, s, x, y, k, vel, yaw, accel, steer)

        # Update solution
        self.solution.append(t, s, x, y, k, vel, yaw, accel, steer)

    def _update_path_index(self):
        """
        Update the index along the path where the vehicle lies.

        :return: None
        """
        dx = [
            self.vehicle.x - x
            for x in self.reference.x_list[self.path_index:self.path_index +
                                           self.config['index_horizon']]
        ]
        dy = [
            self.vehicle.y - y
            for y in self.reference.y_list[self.path_index:self.path_index +
                                           self.config['index_horizon']]
        ]
        dxy = [np.sqrt(x**2 + y**2) for x, y in zip(dx, dy)]
        self.path_index = np.argmin(dxy) + self.path_index

    def _retrieve_imminent_reference(self):
        """
        Retrieve the reference state and reference steer in the imminent
        horizon.

        :return: reference state and reference steer
        """
        reference_state = np.zeros(
            (self.num_state, self.config['horizon'] + 1))
        reference_steer = np.zeros((1, self.config['horizon'] + 1))

        arc_displacement = 0.0
        for t in range(self.config['horizon'] + 1):
            offset = int(round(arc_displacement / self.delta_s))
            if (self.path_index + offset) < self.path_length:
                reference_state[0, t] = \
                    self.reference.x_list[self.path_index + offset]
                reference_state[1, t] = \
                    self.reference.y_list[self.path_index + offset]
                reference_state[2, t] = \
                    self.reference.vel_list[self.path_index + offset]
                reference_state[3, t] = \
                    self.reference.yaw_list[self.path_index + offset]
            else:
                reference_state[0, t] = \
                    self.reference.x_list[self.path_length - 1]
                reference_state[1, t] = \
                    self.reference.y_list[self.path_length - 1]
                reference_state[2, t] = \
                    self.reference.vel_list[self.path_length - 1]
                reference_state[3, t] = \
                    self.reference.yaw_list[self.path_length - 1]
            arc_displacement = \
                arc_displacement + abs(self.vehicle.vel) * self.delta_t
        return reference_state, reference_steer

    def _iterative_control(self, reference_state, reference_steer):
        """
        Helper function to iteratively optimize mpc output.

        :param reference_state: np.array of reference states
        :param reference_steer: np.array of reference steers
        :return: convergence and optimality as boolean
        """
        is_converged = False
        predicted_state = self._predict_state()
        horizon_x, horizon_y, horizon_vel, horizon_yaw, horizon_accel, \
            horizon_steer, solved = \
            self._control(reference_state, predicted_state,
                          reference_steer)
        iteration_difference = \
            np.linalg.norm(self.horizon_accel - horizon_accel, ord=1) + \
            np.linalg.norm(self.horizon_steer - horizon_steer, ord=1)
        self.horizon_accel = horizon_accel
        self.horizon_steer = horizon_steer
        if iteration_difference < self.config['convergence_threshold']:
            is_converged = True
        return is_converged and solved

    def _predict_state(self):
        """
        Propogate controls through the horizon to predict the next state.

        :return: predicted state at end of horizon
        """
        predicted_state = np.zeros(
            (self.num_state, self.config['horizon'] + 1))
        predicted_state[:, 0] = self.vehicle.get_state()
        state = predicted_state[:, 0]
        for accel, steer, t in zip(self.horizon_accel, self.horizon_steer,
                                   range(1, self.config['horizon'] + 1)):
            state = self._step_state(state, accel, steer)
            predicted_state[:, t] = state
        return predicted_state

    def _step_state(self, state, accel, steer):
        """
        Calculate the next state given current state, accel, and steer input.

        :param state: current state as np.array
        :param accel: accel in m/s^2
        :param steer: steer in radians
        :return: next state
        """
        steer = np.clip(steer, self.vehicle.config['min_steer'],
                        self.vehicle.config['max_steer'])
        state[0] = state[0] + state[2] * np.cos(state[3]) * self.delta_t
        state[1] = state[1] + state[2] * np.sin(state[3]) * self.delta_t
        state[2] = state[2] + accel * self.delta_t
        state[3] = state[3] + state[2] / self.vehicle.config['wheelbase'] * \
            np.tan(steer) * self.delta_t
        state[2] = np.clip(state[2], self.vehicle.config['min_vel'],
                           self.vehicle.config['max_vel'])
        return state

    def _control(self, reference_state, predicted_state, reference_steer):
        """
        Solve the MPC control problem.

        :param reference_state: np.array of reference states
        :param predicted_state: np.array of predicted states obtained using
            propogated controls
        :param reference_steer: np.array of reference steering
        :return:
        """
        # intialize problem
        x = cvxpy.Variable((self.num_state, self.config['horizon'] + 1))
        u = cvxpy.Variable((self.num_input, self.config['horizon']))
        cost = constants.Constant(0.0)
        constraints = []

        # iterate over the horizon
        for t in range(self.config['horizon']):
            cost += cvxpy.quad_form(u[:, t], self.config['R'])

            if t != 0:
                cost += cvxpy.quad_form(reference_state[:, t] - x[:, t],
                                        self.config['Q'])

            matrix_a, matrix_b, matrix_c = self._linearized_model_matrix(
                predicted_state[2, t], predicted_state[3, t],
                reference_steer[0, t])
            constraints += [
                x[:,
                  t + 1] == matrix_a @ x[:, t] + matrix_b @ u[:, t] + matrix_c
            ]

            if t < (self.config['horizon'] - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t],
                                        self.config['Rd'])
                constraints += [
                    cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                    self.vehicle.config['max_steer_speed'] * self.delta_t
                ]

        # set the cost
        cost += cvxpy.quad_form(
            reference_state[:, self.config['horizon']] -
            x[:, self.config['horizon']], self.config['Qf'])

        # set the constraints
        constraints += [x[:, 0] == self.vehicle.get_state()]
        constraints += [x[2, :] <= self.vehicle.config['max_vel']]
        constraints += [x[2, :] >= self.vehicle.config['min_vel']]
        constraints += [u[0, :] <= self.vehicle.config['max_accel']]
        constraints += [u[0, :] >= self.vehicle.config['min_accel']]
        constraints += [u[1, :] <= self.vehicle.config['max_steer']]
        constraints += [u[1, :] >= self.vehicle.config['min_steer']]

        # solve the problem
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        # keep track of optimality
        solved = False
        if prob.status == cvxpy.OPTIMAL or \
                prob.status == cvxpy.OPTIMAL_INACCURATE:
            solved = True

        # return solution
        horizon_x = np.array(x.value[0, :]).flatten()
        horizon_y = np.array(x.value[1, :]).flatten()
        horizon_vel = np.array(x.value[2, :]).flatten()
        horizon_yaw = np.array(x.value[3, :]).flatten()
        horizon_accel = np.array(u.value[0, :]).flatten()
        horizon_steer = np.array(u.value[1, :]).flatten()

        return horizon_x, horizon_y, horizon_vel, horizon_yaw, horizon_accel, \
            horizon_steer, solved

    def _linearized_model_matrix(self, vel, yaw, steer):
        """
        Return the transition matrices linearized around vel, yaw, steer.
        Transition matrices A, B, C are of the form:
            Ax_t + Bu_t + C = x_t+1

        :param vel: reference velocity in m/s
        :param yaw: reference yaw in radians
        :param steer: reference steer in radians
        :return: transition matrices
        """
        # state matrix
        matrix_a = np.zeros((self.num_state, self.num_state))
        matrix_a[0, 0] = 1.0
        matrix_a[1, 1] = 1.0
        matrix_a[2, 2] = 1.0
        matrix_a[3, 3] = 1.0
        matrix_a[0, 2] = self.delta_t * np.cos(yaw)
        matrix_a[0, 3] = -self.delta_t * vel * np.sin(yaw)
        matrix_a[1, 2] = self.delta_t * np.sin(yaw)
        matrix_a[1, 3] = self.delta_t * vel * np.cos(yaw)
        matrix_a[3, 2] = \
            self.delta_t * np.tan(steer) / self.vehicle.config['wheelbase']

        # input matrix
        matrix_b = np.zeros((self.num_state, self.num_input))
        matrix_b[2, 0] = self.delta_t
        matrix_b[3, 1] = self.delta_t * vel / \
            (self.vehicle.config['wheelbase'] * np.cos(steer)**2)

        # constant matrix
        matrix_c = np.zeros(self.num_state)
        matrix_c[0] = self.delta_t * vel * np.sin(yaw) * yaw
        matrix_c[1] = -self.delta_t * vel * np.cos(yaw) * yaw
        matrix_c[3] = - self.delta_t * vel * steer / \
            (self.vehicle.config['wheelbase'] * np.cos(steer)**2)

        return matrix_a, matrix_b, matrix_c


def main():
    from objects.objects import Pose, Transform, Location, Rotation, Waypoints, Vector3D
    pose = Pose(
            Transform(Location(x=396.0610046386719, y=268.4299621582031, z=0.033055685460567474), Rotation(pitch=0.10232988744974136, yaw=-89.22174072265625, roll=0.0017161744181066751)), 
            7.523096330217679, 
            Vector3D(x=0.10038097947835922, y=-7.522426605224609, z=-0.00014731811825186014)
            )
    
    pose2 = Pose(Transform(Location(x=396.00225830078125, y=274.99029541015625, z=0.03324241563677788), Rotation(pitch=0.020121736451983452, yaw=-89.90167999267578, roll=-0.02630615420639515)), 0.3966910557184042, Vector3D(x=0.11879204213619232, y=-0.3784867823123932, z=5.187123861105647e-06))

    waypoints2 = Waypoints(waypoints=deque([
        Transform(Location(x=396.4999300817895, y=247.38240568835948, z=1.9513485193682882), Rotation(pitch=0.020121736451983452, yaw=89.90167999267578, roll=-0.02630615420639515)),
        Transform(Location(x=396.4999300817895, y=247.38240568835948, z=1.9513485193682882), Rotation(pitch=0.020121736451983452, yaw=89.90167999267578, roll=-0.02630615420639515)),
        Transform(Location(x=396.25003054756905, y=247.381993623125, z=1.9990853476798827), Rotation(pitch=0.020121736451983452, yaw=89.90167999267578, roll=-0.02630615420639515)),
        Transform(Location(x=396.25003054756905, y=247.381993623125, z=1.9990853476798827), Rotation(pitch=0.020121736451983452, yaw=89.90167999267578, roll=-0.02630615420639515)),
        Transform(Location(x=396.4999040423407, y=247.38242553535468, z=2.007989134905502), Rotation(pitch=0.020121736451983452, yaw=89.90167999267578, roll=-0.02630615420639515))
    ]))

    waypoints = Waypoints(waypoints=deque([
        Transform(Location(x=396.50000488532976, y=246.99854892839795, z=1.2285631883801267), Rotation(pitch=0.10232988744974136, yaw=89.22174072265625, roll=0.0017161744181066751)),
        Transform(Location(x=396.4999920567817, y=246.87295831560212, z=0.904568255870138), Rotation(pitch=0.10232988744974136, yaw=89.22174072265625, roll=0.0017161744181066751)),
        Transform(Location(x=396.4999900156008, y=246.87231769582212, z=0.5459262499000094), Rotation(pitch=0.10232988744974136, yaw=89.22174072265625, roll=0.0017161744181066751)),
        Transform(Location(x=396.4999898766505, y=246.8722740865791, z=0.5215122263486052), Rotation(pitch=0.10232988744974136, yaw=89.22174072265625, roll=0.0017161744181066751)),
        Transform(Location(x=396.2500643873447, y=246.7443077719056, z=0.768080510618347), Rotation(pitch=0.10232988744974136, yaw=89.22174072265625, roll=0.0017161744181066751)),
        Transform(Location(x=396.4999794366592, y=246.74743311667083, z=0.6171943586872557), Rotation(pitch=0.10232988744974136, yaw=89.22174072265625, roll=0.0017161744181066751)),
        Transform(Location(x=396.4999805621572, y=246.7477863515393, z=0.8149479494536302), Rotation(pitch=0.10232988744974136, yaw=89.22174072265625, roll=0.0017161744181066751)),
        Transform(Location(x=396.5000915517723, y=246.6227494898065, z=0.8000345514480661), Rotation(pitch=0.10232988744974136, yaw=89.22174072265625, roll=0.0017161744181066751)),
        Transform(Location(x=396.50008049901, y=246.4977162031365, z=0.7880508399250239), Rotation(pitch=0.10232988744974136, yaw=89.22174072265625, roll=0.0017161744181066751)),
        Transform(Location(x=396.5000692406011, y=246.37261837478684, z=0.7399343735459034), Rotation(pitch=0.10232988744974136, yaw=89.22174072265625, roll=0.0017161744181066751))
        ]))

    controller = Controller()
    steer, throttle, brake, duration = controller.get_control_instructions(0, pose2, waypoints2)

    print("{} {} {} {}".format(throttle, steer, brake, duration))
    
if __name__ == '__main__':
    main()
