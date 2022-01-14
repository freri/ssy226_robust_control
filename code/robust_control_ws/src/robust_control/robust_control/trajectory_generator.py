import numpy as np
import casadi as cs
import random as rnd
from transforms3d.euler import euler2quat, quat2euler
from rclpy.node import Node
import rclpy
from atr_interfaces.srv import UpdateATRTrajectoryStamped
from atr_interfaces.msg import PoseWithDTime
from atr_interfaces.msg import TrajectoryWithVel
from geometry_msgs.msg import PoseStamped
from atr_interfaces.msg import ATRStateStamped
from geometry_msgs.msg import Twist
from pathlib import Path
import os, json
import time


class TrajGen(Node):
    """Sloppy implementation but does what it should, namely generates 
    2 second trajectories from the robots current position.
    The trajectories are random, and no thourough check is done on if the generated
    trajectories are feasible, so take that into consideration when testing using this.

    The update rate (how often to publish trajectories) can be modified in config.json (only tested 1 and 2).

    There is also a simulated delay in the send_trajectory method
    """
    def __init__(self):
        super().__init__('traj_gen')
        config_path = Path(__file__)
        config_path = os.path.join(
            str(config_path.parent.parent), 'config', 'config.json')
        config_path = config_path.replace('/build/', '/src/')
        with open(config_path) as f:
            config = json.load(f)
        simulation = config['simulation']
        base = config['base']
        self.traj_update_rate = simulation["traj_update_rate"] #in seconds
        self.N = base['N']  # The MPC horizon length
        self.NX = base['NX']  # The number of elements in the state vector
        self.NU = base['NU']  # The number of elements in the control vector
        self.ts = base['atr_ts']  # Controller sampling time
        self.traj_ts = base['traj_ts'] 
        self.atr_id = base['atr_id']
        # Used to simulate a delay in 'send_trajectory'        
        self.delay = 0.0
        self.traj_len = 20 # how many sampling points to generate (20 = 2s traj)
        self.counter = 0
        self.log = False
        # Wheel velocities
        self.v = np.vstack([0.5, 0.5])
        self.init_state = False
        
        # Setup client to simulate trajectory generation
        self.client = self.create_client(UpdateATRTrajectoryStamped,f'update_atr_{self.atr_id}_path')
        self.publisher = self.create_publisher(TrajectoryWithVel, f'atr_{self.atr_id}_traj', 1)
        self.atr_state_sub = self.create_subscription(ATRStateStamped,f'atr_{self.atr_id}_state', self.atr_state_callback, 1)
        self.timer = self.create_timer(self.traj_update_rate, self.generate_trajectory)

    def atr_state_callback(self, msg):
        # NOTE: multiplied by hardcoded factor to change orientation of next trajectory
        theta = self.quat2eul(msg.state.pose.fused_odom.orientation) * 1.1
        self.state = [msg.state.pose.fused_odom.position.x, msg.state.pose.fused_odom.position.y, theta]
        self.init_state = True

    def quat2eul(self, orientation):
            _,_,theta = quat2euler([orientation.w, orientation.x, orientation.y, orientation.z])
            return theta #Conversion to radians

    def generate_trajectory(self):
        if self.init_state:
            if self.log:
                self.get_logger().info("Generating Trajectory")
                self.get_logger().info(f"Simulating an additional delay of {self.delay} seconds.")
            
            u = np.zeros(self.traj_len * self.NU)
            L = 0.4645 # distance between wheels on atr

            # Matrix to convert from wheel speeds to velocity and angular velocity
            A = np.matrix([[1/2, 1/2], [1/L, -1/L]])

            # Base value for wheel speeds
            
            # Maximum / minimum speed for individual wheels
            v_wheel_max = 1
            v_wheel_min = -0.5

            # Minimum speed for the robot
            v_robot_min = 0.5

            # Wheel velocities
            v = self.v
            # Atr states
            x = self.state

            # Control signals, velocity and ang. vel
            u = np.array([0.0, 0.0])
            
            v_temp = v
            traj = np.array([])
            for i in range(0, self.traj_len):

                v_temp[0] = np.fmax(v_wheel_min, np.fmin(v_wheel_max, v[0] + [-1,1][rnd.randrange(2)] * rnd.randrange(0,10)/10))
                v_temp[1] = np.fmax(v_wheel_min, np.fmin(v_wheel_max, v[1] + [-1,1][rnd.randrange(2)] * rnd.randrange(0,10)/10))
                u_temp = np.matmul(A, v_temp)
                
                while (u_temp[0] < v_robot_min):
                    v_temp[0] = np.fmax(v_wheel_min, np.fmin(v_wheel_max, v[0] + [-1,1][rnd.randrange(2)] * rnd.randrange(0,10)/10))
                    v_temp[1] = np.fmax(v_wheel_min, np.fmin(v_wheel_max, v[1] + [-1,1][rnd.randrange(2)] * rnd.randrange(0,10)/10))
                    u_temp = A * v_temp
                v = v_temp
                u = u_temp
                x = self.dynamics_dt_rk4(x, u)
                traj = np.append(traj, np.append(x, u))
            
            traj = np.reshape(traj,[self.traj_len, self.NX + self.NU])
            
            self.send_trajectory(traj)
            self.v = v
    def dynamics_ct(self, _x, _u):
        return cs.vcat([_u[0] * cs.cos(_x[2]),
                        _u[0] * cs.sin(_x[2]),
                        _u[1]])

    def dynamics_dt(self, x, u):
        dx = self.dynamics_ct(x, u)
        return cs.vcat([x[i] + self.traj_ts * dx[i] for i in range(self.NX)])

    def dynamics_dt_rk4(self, _x, _u):
        
        f = self.dynamics_ct(_x, _u)
        k1 = f * self.traj_ts
        f = self.dynamics_ct(_x + 0.5*k1, _u)
        k2 = f * self.traj_ts
        f = self.dynamics_ct(_x + 0.5*k2, _u)
        k3 = f * self.traj_ts
        f = self.dynamics_ct(_x + k3, _u)
        k4 = f * self.traj_ts         
        return cs.vcat([_x[i] + (1/6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(self.NX)])

    def send_trajectory(self, traj):
        """Taken from gpss-atr-trajgen, but added timestamp"""
        req = UpdateATRTrajectoryStamped.Request()

        i = 1 # Used to set the delta_time for each pose
        for elem in traj: 
            pwdt = PoseWithDTime()
            pwdt.delta_time = self.traj_ts * i
            pwdt.pose.position.x = elem[0]
            pwdt.pose.position.y = elem[1]
            pwdt.pose.position.z = 0.0 
            pwdt.pose.orientation = self.eul2quat(elem[2])
            req.trajectory.poses.append(pwdt)            
            
            velcmd = Twist()
            velcmd.linear.x = elem[3]
            velcmd.angular.z = elem[4]
            req.trajectory.vel.append(velcmd)

            i += 1
        
        req.header.stamp = self.get_clock().now().to_msg()
        time_stamp_sec = req.header.stamp.sec + req.header.stamp.nanosec/1e9
        now_sec = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec/1e9
        
        while (now_sec - time_stamp_sec) < self.delay:
            now_sec = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec/1e9
        self.client.call_async(req)
        # msg = TrajectoryWithVel()
        # msg.poses = req.trajectory.poses
        # msg.vel = req.trajectory.vel
       
        # self.publisher.publish(msg)
       



    def eul2quat(self,theta):
        """Taken from gpss-atr-trajgen"""
        orientation = PoseStamped().pose.orientation
        orientation.w, orientation.x, orientation.y, orientation.z  = euler2quat(0, 0, theta)
        return orientation

def main(args=None):

    rclpy.init(args=args)

    traj_gen = TrajGen()

    rclpy.spin(traj_gen)

    traj_gen.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()