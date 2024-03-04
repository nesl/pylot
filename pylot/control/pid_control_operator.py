from collections import deque

import erdos
from erdos import Message, ReadStream, Timestamp, WriteStream

import socket, pickle, time

import pylot.control.utils
import pylot.planning.utils
from pylot.control.messages import ControlMessage
from pylot.control.pid import PIDLongitudinalController

import service
import convert

class PIDControlOperator(erdos.Operator):
    """Operator that uses PID to follow a list of waypoints.

    The operator waits for the pose and waypoint streams to receive a watermark
    message for timestamp t, and then it computes and sends a control command.

    Args:
        pose_stream (:py:class:`erdos.ReadStream`): Stream on which pose
            info is received.
        waypoints_stream (:py:class:`erdos.ReadStream`): Stream on which
            :py:class:`~pylot.planning.messages.WaypointMessage` messages are
            received. The operator receives waypoints from the planning
            operator, and must follow these waypoints.
        control_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends :py:class:`~pylot.control.messages.ControlMessage`
            messages.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, pose_stream: ReadStream, waypoints_stream: ReadStream,
                 control_stream: WriteStream, flags):
        self._file = open("/home/erdos/workspace/pylot/temp_log.txt", "a")
        pose_stream.add_callback(self.on_pose_update)
        waypoints_stream.add_callback(self.on_waypoints_update)
        erdos.add_watermark_callback([pose_stream, waypoints_stream],
                                     [control_stream], self.on_watermark)
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        pid_use_real_time = False
        if self._flags.execution_mode == 'real-world':
            # The PID is executing on a real car. Use the real time delta
            # between two control commands.
            pid_use_real_time = True
        if self._flags.simulator_control_frequency == -1:
            dt = 1.0 / self._flags.simulator_fps
        else:
            dt = 1.0 / self._flags.simulator_control_frequency
        self._pid = PIDLongitudinalController(flags.pid_p, flags.pid_d,
                                              flags.pid_i, dt,
                                              pid_use_real_time)
        # Queues in which received messages are stored.
        self._waypoint_msgs = deque()
        self._pose_msgs = deque()
        self._server=None

    @staticmethod
    def connect(pose_stream: ReadStream, waypoints_stream: ReadStream):
        control_stream = erdos.WriteStream()
        return [control_stream]

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))

    def connect_to_server(self):
        if self._flags.use_remote_pid_server:
            host = self._flags.remote_control_server_local
            port = self._flags.remote_control_port_local
        else:
            print("Remote PID Server not enabled!")
            return None
        
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.connect((host, port))

        return self._server

    def fetch_from_server(self, pose_msg, waypoints):
        if self._server == None:
            self.connect_to_server()
        
        controller_input = service.ControllerInput(
            pose_msg=convert.from_pylot_pose(pose_msg.data), 
            waypoints_msg=convert.from_pylot_waypoint(waypoints), 
            type="pid"
            )
        
        print("Sent controller input ", controller_input)
        input_string = pickle.dumps(controller_input)

        service.send_msg(self._server, input_string)
        output_string = self._server.recv(4096)

        control_output = pickle.loads(output_string)
        print("Received control message ", control_output)
        return control_output

    @erdos.profile_method()
    def on_watermark(self, timestamp: Timestamp, control_stream: WriteStream):
        """Computes and sends the control command on the control stream.

        Invoked when all input streams have received a watermark.

        Args:
            timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
                the watermark.
        """
        self._logger.debug('@{}: received watermark'.format(timestamp))
        if timestamp.is_top:
            return
        pose_msg = self._pose_msgs.popleft()
        ego_transform = pose_msg.data.transform
        # Vehicle speed in m/s.
        current_speed = pose_msg.data.forward_speed
        waypoints = self._waypoint_msgs.popleft().waypoints

        if self._flags.use_remote_pid_server:
            control_start_time = time.time()
            control_output = self.fetch_from_server(pose_msg=pose_msg, waypoints=waypoints)
            total_cotnrol_time = 1000*(time.time() - control_start_time)
            print("Total PID Control Time: ", total_cotnrol_time)
            control_message = convert.to_pylot_control_message(control_output, timestamp)
            control_stream.send(control_message)
        elif len(waypoints.waypoints) == 1:
            self._file.write('\nBraking! Only one waypoints to follow.')
            throttle, brake = 0.0, 0.5
            steer = 0.0
            control_stream.send(
                ControlMessage(steer, throttle, brake, False, False, timestamp))
        else:
            try:
                angle_steer = waypoints.get_angle(
                    ego_transform, self._flags.min_pid_steer_waypoint_distance)
                target_speed = waypoints.get_target_speed(
                    ego_transform, self._flags.min_pid_speed_waypoint_distance)
                throttle, brake = pylot.control.utils.compute_throttle_and_brake(
                    self._pid, current_speed, target_speed, self._flags,
                    self._logger)
                steer = pylot.control.utils.radians_to_steer(
                    angle_steer, self._flags.steer_gain)
                wp_index = waypoints._get_index(ego_transform, self._flags.min_pid_steer_waypoint_distance)
                self._file.write("\nego transform, waypoints : " + str(ego_transform) +" " + str(len(waypoints.waypoints)) + " " + str(waypoints.waypoints[wp_index]))
                self._file.write("\ncontrol input and output - speed, throttle, brake, steer "+str(timestamp)+" "+ str(target_speed) + " " + str(throttle)+" "+ str(brake)+" "+ str(steer))
            except ValueError:
                self._file.write('\nBraking! No more waypoints to follow.')
                throttle, brake = 0.0, 0.5
                steer = 0.0
            self._logger.debug(
                '@{}: speed {}, location {}, steer {}, throttle {}, brake {}'.
                format(timestamp, current_speed, ego_transform, steer, throttle,
                    brake))
            control_stream.send(
                ControlMessage(steer, throttle, brake, False, False, timestamp))

    def on_waypoints_update(self, msg: Message):
        self._logger.debug('@{}: waypoints update'.format(msg.timestamp))
        self._waypoint_msgs.append(msg)

    def on_pose_update(self, msg: Message):
        self._logger.debug('@{}: pose update'.format(msg.timestamp))
        self._pose_msgs.append(msg)
