import json
import rclpy
from rclpy.node import Node
from atr_interfaces.srv import UpdateATRTrajectoryStamped
from atr_interfaces.msg import ATRStateStamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
import sys
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import quat2euler, euler2quat
import time
import math
import os
from pathlib import Path
from .atr_model import ATRModel
from .mpc_builder import MPC
from numpy import matlib
from pyqtgraph.Qt import QtGui

import numpy as np
from .plotter import Plotter

class RobustControl(Node):

    def __init__(self):
        super().__init__('robust_control')

        # Setup ATR model
        self.atr = ATRModel()
        # Enable / disable printing of logging messages
        self.log_details = False
        # Init parameters
        self.init_params()
        self.traj_processed = False
        self.init = True
        self.solver_iteration = 0
        self.new_traj_idx = []
        self.solve_times = []
        self.states = [0,0,0]
        self.u_star = [0,0]

        # For plotting
        # Save references that were actually sent to controller
        self.saved_traj_refs = []
        self.saved_vel_refs = []
        # Save original (nonsplined) trajectory for plotting
        self.saved_trajectories = []
        # Save simulated states
        self.saved_states = []
        # Save control signals
        self.saved_u = []

        # Number of trajectories recieved
        self.num_traj_recieved = 0
        # Import solver
        self.solver = self.import_solver()

        # Setup service to recieve trajectories
        self.update_atr_traj_srv = self.create_service(
            UpdateATRTrajectoryStamped,
            f'update_atr_{self.atr_id}_path',
            self.new_trajectory_callback)
        
        self.get_logger().info("Trajectory service set up")

        self.atr_vel_pub = self.create_publisher(
            Twist, '/cmd_vel_atr_{}'.format(self.atr_id), 1)

        self.atr_state_pub = self.create_publisher(ATRStateStamped, f'atr_{self.atr.id}_state', 1)
        # Publish initial state to indicate that the controller is ready for a new trajectory
        self.publish_state()
        self.get_logger().info("Waiting for trajectory...")

    def new_trajectory_callback(self, req, res):
        self.num_traj_recieved += 1
        self.new_traj_idx += [self.solver_iteration]
        self.get_logger().info("Average solve time: " + str(round(np.average(self.solve_times),1)) + " ms")
        if self.num_traj_recieved > self.trajs_to_sim:
            self.plot()
        self.traj_processed = False
        self.update_trajectory(req)
        self.traj_processed = True
        self.recieved_path = True
        res.success = True
        if self.init:
            self.get_logger().info("Starting solver.")
            self.run_solver_timer = self.create_timer(self.atr.ts, self.run_solver)
            self.init = False
        return res

    def update_trajectory(self, msg):
        trajectory = msg.trajectory
        
        self.get_logger().info("New trajectory recieved")
        self.atr_traj_dt = []
        traj = []
        vel_ref = []

        for pose in trajectory.poses:
            position = pose.pose.position
            orientation = pose.pose.orientation
            _, _, theta = quat2euler(
                [orientation.w, orientation.x, orientation.y, orientation.z])
            traj += [position.x, position.y, theta]
            self.atr_traj_dt += [pose.delta_time]
        self.atr_traj_dt = np.array(self.atr_traj_dt)
        for vel in trajectory.vel:
            vel_ref += [vel.linear.x, vel.angular.z]

        self.atr_traj = np.array(traj)
        # Save for plotting
        self.saved_trajectories = np.append(self.saved_trajectories, self.atr_traj)
        self.atr_vel_ref = np.array(vel_ref)

        # # Length of trajectory
        self.traj_len = int(len(self.atr_traj) / self.NX)
        # # Extract timestamps for current trajectory
        self.atr_traj_t0 = msg.header.stamp.sec + msg.header.stamp.nanosec/1e9
        # Spline trajectory
        self.linear_spline()
        # Delete poses that are irrelevant to reach due to delay
        self.delete_past_poses()

        # self.simulate_trajectory()

    def linear_spline(self):
        _traj = self.atr_traj.copy()
        tt = time.time()
        # Number of points needed in-between each node
        N = int(self.traj_ts / self.atr.ts)
        # Save data for plotting reasons..
        self.N_new_poses = N
        # Perform linear spline between all nodes
        x_new = self.ranges_based_v2(_traj[0::3], N + 1)
        y_new = self.ranges_based_v2(_traj[1::3], N + 1)
        # Add N-1 copies of first pose before new trajectory
        x_new = np.append([[x_new[0]] * (N-1)], x_new)
        y_new = np.append([[y_new[0]] * (N-1)], y_new)
        # Since the spline is linear just add N copies of each theta
        theta = np.array([[_theta] * N for _theta in _traj[2::3]]).flatten()

        # Update size of trajectory
        self.N_splined_traj = int(len(x_new))
        # Add new x, y and theta to the splined trajectory
        splined_traj = np.zeros(self.NX * self.N_splined_traj)

        splined_traj[0::3] = x_new
        splined_traj[1::3] = y_new
        splined_traj[2::3] = theta
        self.splined_atr_traj = splined_traj

        # Update dt to reflect new nodes / sampling time
        # Add a zero in front to spline from 0 to the first dt
        temp = np.append(0, self.atr_traj_dt)
        splined_traj_dt = self.ranges_based_v2(temp, N + 1)
        # Remove the zero
        self.splined_traj_dt = splined_traj_dt[1:]
        # Splined trajectory sampling time
        self.splined_traj_ts = self.atr.ts
        # When using linear_spline u_ref is piece-wise constant       
        u_ref = np.array(
            [[_u_ref] * self.N_new_poses for _u_ref in np.reshape(self.atr_vel_ref, [self.traj_len, self.NU])]).flatten()
        self.splined_atr_vel_ref = u_ref
        # Splined trajectory sampling time
        self.splined_traj_ts = self.atr.ts
        if self.log_details:
            self.get_logger().info(
                f"Total spline time: {round(1e3*(time.time() - tt), 4)} ms.")

    def ranges_based_v2(self, a, N):
        # https://stackoverflow.com/a/53961706 @Divakar
        """
        Fast way to create linspace between points in vector
        """
        start = a
        stop = np.concatenate((a[1:], [0]))
        return self.create_ranges(start, stop, N-1, endpoint=False).ravel()[:-N+2]

    def create_ranges(self, start, stop, N, endpoint=True):
        # https://stackoverflow.com/a/40624614/ @Divakar
        if endpoint == 1:
            divisor = N - 1
        else:
            divisor = N
        steps = (1.0 / divisor) * (stop - start)
        return steps[:, None]*np.arange(N) + start[:, None]

    def delete_past_poses(self):
        """
        Based on the delay between when the trajectory was sent and the current time,
        discard poses that were already supposed to be reached.
        """
        self.tt = time.time()
        _traj = self.splined_atr_traj
        _traj_dt = self.splined_traj_dt
        _u_ref = self.splined_atr_vel_ref
        # Current time (in s) compared to when trajectory was sent
        cur_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec/1e9
        delay = cur_time - self.atr_traj_t0
        # Account for additional delay after this by
        # manually increasing the delay a little bit to make sure
        # controller is not commanded to go to poses behind it
        extra_delay = 0.0
        delay = delay + extra_delay
        
        # Get indicies of unreachable poses due to delay
        idxs = np.where(_traj_dt <= delay)[0]
        # Number of unreachable poses 
        count = len(idxs)
        # count = 0
        # if self.log_details:
        self.get_logger().info(f"Delay since trajectory was sent {round(1e3 * (delay-extra_delay))} ms")
        self.get_logger().info(f"→ deleting {count} irrelevant poses")
        # Save amount of deleted poses for plotting:
        self.del_poses = count
        # If there are unreachable poses (and if not all of them are unreachable)
        if count > 0 and count < self.N_splined_traj - 1:
            # Delete unreachable poses
            _traj = np.delete(_traj, range(0, idxs[-1] * self.NX + self.NX))
            # Delete corresponding reference control signals
            _u_ref = np.delete(_u_ref, range(0, idxs[-1] * self.NU + self.NU))

            # Extend trajectory with copies of last pose
            _traj = np.append(_traj, np.matlib.repmat(
                _traj[-3:], 1, count)[0])
            # Same for u_ref
            _u_ref = np.append(_u_ref, np.array([_u_ref[-2:]] * count ).flatten())

        elif count >= self.N_splined_traj - 1:
            self.get_logger().error(f"Delay is too large: {delay} seconds.")
            self.destroy_node()
            exit(-1)

        assert len(_traj) == self.NX * \
            self.N_splined_traj, f"new traj len {len(_traj)}, should be {self.NX * self.N_splined_traj}"
        if self.log_details:
            self.get_logger().info(
                f"Deletion took: {round(1e3*(time.time() - self.tt), 4)} ms")
        self.splined_atr_traj = _traj
        self.splined_atr_vel_ref = _u_ref

    def publish_control_msg(self):
        msg = Twist()
        msg.linear.x = self.u_star[0]
        msg.angular.z = self.u_star[1]
        self.atr_vel_pub.publish(msg)
        self.control_msg_published = self.get_clock().now().to_msg().nanosec

    def eul2quat(self,theta):
        orientation = PoseStamped().pose.orientation
        orientation.w, orientation.x, orientation.y, orientation.z  = euler2quat(0, 0, theta)
        return orientation

    def publish_state(self):
        msg = ATRStateStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        msg.state.pose.fused_odom.position.x = float(self.states[0])
        msg.state.pose.fused_odom.position.y = float(self.states[1])
        msg.state.pose.fused_odom.position.z = 0.0
        msg.state.pose.fused_odom.orientation = self.eul2quat(float(self.states[2]))
        
        msg.state.atr_id = self.atr.id

        self.atr_state_pub.publish(msg)

    def simulate_control_msg(self):
        self.states = np.array(self.atr.dynamics_dt_rk4(self.states, self.u_star))
        self.publish_state()
        self.saved_states = np.append(self.saved_states, self.states)
        self.saved_u = np.append(self.saved_u, self.u_star)        

    def run_solver(self):
        # self.get_logger().info(
        #         f"Time since deletion: {round(1e3*(time.time() - self.tt), 4)} ms")
        if self.traj_processed:
            # Create parameter vector
            self.solver_iteration += 1
            p = self.states
            p = np.append(p, self.u_star)
            # Send the first N (MPC_Horizon) nodes and control references
            p = np.append(p, self.splined_atr_traj[:self.N * self.NX])
            p = np.append(p, self.splined_atr_vel_ref[:self.N * self.NU])
            
            # Run solver
            result = self.solver.run(p)
            self.u_star = result.solution[:2]
            self.solve_times += [result.solve_time_ms]
            # Save for plotting reasons
            self.saved_traj_refs = np.append(self.saved_traj_refs, self.splined_atr_traj[:3])
            self.saved_vel_refs = np.append(self.saved_vel_refs, self.splined_atr_vel_ref[:2])
            # Remove previous node and pad with last node if needed
            self.splined_atr_traj = np.delete(self.splined_atr_traj, [0, 1, 2])
            if len(self.splined_atr_traj) < self.N * self.NX:
                self.splined_atr_traj = np.append(self.splined_atr_traj, self.splined_atr_traj[-self.NX:])
            
            # Remove previous u_ref and pad with last u_ref if needed
            self.splined_atr_vel_ref = np.delete(self.splined_atr_vel_ref, [0, 1])
            if len(self.splined_atr_vel_ref) < self.N * self.NX:
                self.splined_atr_vel_ref = np.append(self.splined_atr_vel_ref, self.splined_atr_vel_ref[-self.NU:])
            self.simulate_control_msg()
        
            
    def plot(self):        
        self.get_logger().info(f"Simulated {self.trajs_to_sim} trajectories, plotting results...")
        app = QtGui.QApplication(sys.argv)
        thisapp = Plotter(self.saved_trajectories, self.saved_traj_refs,
                    self.saved_states, self.saved_u, self.saved_vel_refs, self.new_traj_idx, self.traj_len, self.get_logger(), details = True)
        thisapp.show()
        sys.exit(app.exec_())

    def init_params(self):
        config_path = Path(__file__)
        config_path = os.path.join(
            str(config_path.parent.parent), 'config', 'config.json')
        config_path = config_path.replace('/build/', '/src/')
        with open(config_path) as f:
            config = json.load(f)
        simulation = config['simulation']
        self.trajs_to_sim = simulation["number_of_traj_to_sim"]

        base = config['base']
        # Base config
        self.N = base['N']  # The MPC horizon length
        self.NX = base['NX']  # The number of elements in the state vector
        self.NU = base['NU']  # The number of elements in the control vector
        self.ts = base['atr_ts']  # Controller sampling time
        self.traj_ts = base['traj_ts'] 
        self.atr_id = base['atr_id']
        self.build_directory = base['build_directory']
        self.optimizer_name = base['optimizer_name']

    def import_solver(self):
        fpath = Path(__file__)
        fpath = os.path.join(
            str(fpath.parent), self.build_directory, self.optimizer_name)
        sys.path.insert(1, fpath)
        
        self.get_logger().info("Importing solver...")
        try:
            from .mpc_build.controller import controller
            self.get_logger().info("Solver imported")
            # Commented code does not work when run from a ros2 launch script
            # inp = input(
            #     "Would you like to rebuild solver? (y/n) (or just press ENTER to continue without rebuilding)\n")
            # if inp == 'y':
            #     MPC().build()
            #     from .mpc_build.controller import controller
        except ModuleNotFoundError or ImportError:            
            self.get_logger().info("Solver not found")        
            self.get_logger().info("Building solver...")
            # inp = input("Press ENTER to build solver.")
            # if inp == '':
            MPC().build()
            from .mpc_build.controller import controller
        return controller.solver()

def main(args=None):
    rclpy.init(args=args)

    robust_control = RobustControl()

    rclpy.spin(robust_control)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    robust_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
