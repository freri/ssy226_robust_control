# Author :
# Modified by August Mikaelsson, Sophie Vännman, 2021-07-14

import rclpy
from rclpy.node import Node
import os
from ament_index_python.packages import get_package_share_directory
import utils_config
from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import PoseStamped
from atr_interfaces.msg import ATRStateStamped
from atr_interfaces.msg import ATRStateListStamped
from atr_interfaces.msg import ATRPoseListStamped
#from tf_conversions.transformations import quaternion_from_euler as quat2eul
from transforms3d.euler import quat2euler
from transforms3d.euler import euler2quat
import numpy as np
from .kalman_filter import KalmanFilter
from atr_interfaces.msg import ATRState
import numpy as np

from atr_interfaces.msg import Pose2DStamped

class Observer(Node):
    def __init__(self, config, kalman):
        """ A node that updates the current position with the help of a kalman filter,
        using odometry and photogrammetry data.

        :param kalman: a kalman filter
        """
        super().__init__('observer')


        self.kalman = kalman
        self.config_s = config.observer
        self.id = config.robot_id

        #create publishers and subscribers
        self.pub_ = self.create_publisher(ATRStateStamped, 'atr_{}_state'.format(self.id), 1)
        self.pub_rviz = self.create_publisher(PoseStamped, '/rviz_observer_pose_atr_{}'.format(self.id), 1)
        self.pub_rviz_opt = self.create_publisher(PoseStamped, '/rviz_optom_atr_{}'.format(self.id), 1)
        self.sub_photo = self.create_subscription(ATRPoseListStamped, '/atr_state_list', self.photogrammetry_callback, 1)
        self.sub_odom = self.create_subscription(Pose2DStamped, '/odom_atr_{}'.format(self.id), self.odometry_callback, 1)

        #kalman matrices
        self.kalman.mat_process_cov =  self.config_s.weight_odom*np.diag([1, 1, 1])
        self.kalman.mat_measure_cov =  np.diag([self.config_s.weight_photo_pos, self.config_s.weight_photo_pos, self.config_s.weight_photo_ang])

        #variables for photogrametry callback
        self.startup = True
        self.photo_x = 0
        self.photo_y = 0
        self.photo_theta = 0
        self.odom_angle = 0.0

        #variables for odom callback
        self.startup_odom = True
        self.temp_theta = 0.0

        #iniate kalman filter variables
        self.kalman.vec_xest = [self.config_s.start_x,self.config_s.start_y,self.config_s.start_theta]

        # Buffer vectors
        self.saveN = self.config_s.saveN
        self.vec_odom = np.zeros(self.saveN*3).reshape(self.saveN,3)
        self.vec_pos_odom = np.zeros(self.saveN*3).reshape(self.saveN,3)
        self.timestamp_vec = np.zeros(self.saveN)       
        self.cov_matrix_vec = np.zeros((self.saveN, 3,3))

        self.diff_limit = 50   

        self.get_logger().info("Observer initiated")

        
    
    def quat2eul(self, orientation):
        _,_,theta = quat2euler([orientation.w, orientation.x, orientation.y, orientation.z])
        return theta #Convertion to radians 
    
    def odometry_callback(self, msg_odom):
        """ odometry_callback is called each time a Pose2DStamped has been published to odom_atr_ID. This
         function updates the fused position using only odometry data."""


        current_timestamp = self.get_clock().now().to_msg()

        #self.get_logger().info("odom_callbacktime: {}".format(self.get_clock().now().to_msg()))
        #self.get_logger().info("msgtime: {}".format(msg_odom.timestamp))
        
        # Update temp_theta until first photogrammetry has arrived.
        if self.startup_odom:
            self.temp_theta = self.kalman.vec_xest[2] - self.odom_angle
            self.odom_angle += msg_odom.pose.theta
        if not self.startup and self.startup_odom:
            self.startup_odom = False

        # Calculate change in pose in world frame
        cos_t = np.cos(self.temp_theta)
        sin_t = np.sin(self.temp_theta)
        delta_x = msg_odom.pose.x*cos_t - msg_odom.pose.y*sin_t
        delta_y = msg_odom.pose.y*cos_t + msg_odom.pose.x*sin_t
        delta_theta = msg_odom.pose.theta

        # Add new odom reading to vec_odom
        self.vec_odom[0:self.saveN-1] = self.vec_odom[1:self.saveN]
        self.vec_odom[self.saveN-1] = np.array([delta_x, delta_y, delta_theta])
        
        # Add new timestamp to timestamp_vec
        self.timestamp_vec[0:self.saveN-1] = self.timestamp_vec[1:self.saveN]
        self.timestamp_vec[self.saveN-1] = np.array([msg_odom.timestamp])

        # Do prediction step and save absolute position
        self.vec_pos_odom[0:self.saveN-1] = self.vec_pos_odom[1:self.saveN]
        self.vec_pos_odom[self.saveN-1] = self.kalman.predict(np.array([delta_x, delta_y, delta_theta]))
        
        # Update covariance matrix vector
        self.cov_matrix_vec[0:self.saveN-1,:,:] = self.cov_matrix_vec[1:self.saveN,:,:]
        self.cov_matrix_vec[self.saveN-1,:,:] = self.kalman.mat_state_cov

        # Create message and populate with corresponding values, publish both msg_fused and msg_rviz
        msg_fused = ATRStateStamped()
        msg_fused.state.pose.fused_odom.position.z = 0.0
        # msg_fused.state.pose.fused_odom.position.x = self.kalman.vec_xest[0]
        # msg_fused.state.pose.fused_odom.position.y = self.kalman.vec_xest[1]
        # msg_fused.state.pose.fused_odom.orientation = self.eul2quat(self.kalman.vec_xest[2])
        msg_fused.state.pose.fused_odom.position.x = self.vec_pos_odom[-1,0]
        msg_fused.state.pose.fused_odom.position.y = self.vec_pos_odom[-1,1]
        msg_fused.state.pose.fused_odom.orientation = self.eul2quat(self.vec_pos_odom[-1,2])
        msg_fused.header.frame_id = "/base_link"
        msg_fused.header.stamp = current_timestamp
        msg_fused.state.pose_source = ATRState.FUSED_ODOM
        msg_fused.state.atr_id = self.id

        msg_fused.state.pose.odom.position.z = 0.0
        msg_fused.state.pose.odom.position.x = self.vec_pos_odom[-1,0]
        msg_fused.state.pose.odom.position.y = self.vec_pos_odom[-1,1]
        msg_fused.state.pose.odom.orientation = self.eul2quat(self.vec_pos_odom[-1,2])

        self.pub_.publish(msg_fused)

        msg_rviz = PoseStamped()
        msg_rviz.header = msg_fused.header

        msg_rviz.pose = msg_fused.state.pose.fused_odom

        self.pub_rviz.publish(msg_rviz)

        # self.get_logger().info("Odom pos: ({},{})".format(msg_fused.state.pose.fused_odom.position.x, msg_fused.state.pose.fused_odom.position.y))

        
    def photogrammetry_callback(self, msg):
        """ photogrammetry_callback is called each time a ATRPoseListStamped has been published to /atr_state_list. This
         function fuses the photo data with odometry data, using timestamps to ensure the fusing is done between 
         corresponding values. The fusing is done using a Kalman filter."""

        # self.get_logger().info("msgtime: {}".format(msg.timestamp))
        # self.get_logger().info("callbacktime: {}".format(self.get_clock().now().to_msg()))
        # self.get_logger().info("Photo Timestamp: {}".format(msg.timestamp))

        for m in msg.list.atr_states:
            if m.atr_id == self.id and m.pose_source == ATRState.OPTOM:

                # Update photo data based on received messages
                self.photo_x = m.pose.optom.position.x
                self.photo_y = m.pose.optom.position.y
                self.photo_theta = (self.quat2eul(m.pose.optom.orientation) + np.pi) % (2 * np.pi) - np.pi
                #self.flag_photo = True


                # Remap april tag to center of robot
                angle = self.photo_theta
                matrix = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                self.photo_x, self.photo_y = np.matmul(matrix, [self.config_s.shift_tag_x, self.config_s.shift_tag_y]) + [self.photo_x, self.photo_y]


                # Initiate kalman internal variable at startup
                if self.startup and self.config_s.photo == 1.0:
                    self.kalman.vec_xest = self.photo_x, self.photo_y, self.photo_theta
                    self.startup = False

                # Search for odometry timestamp closest to the photo timestamp
                idx = (np.abs(self.timestamp_vec-msg.timestamp)).argmin()
                #self.get_logger().info("IDX: {}".format(idx))
                #self.get_logger().info("Diff: {}".format(abs(self.timestamp_vec-msg.timestamp)))

                # Check that found odometry-data has a timestamp within limit from the photo timestamp
                diff = abs(self.timestamp_vec[idx]-msg.timestamp)
                # self.get_logger().info(str(diff))
                

                if not self.startup and diff<self.diff_limit: 
                    # Prediction step from odometry data
                    prediction = self.vec_pos_odom[idx]

                    # Get cov matrix to corresponding timestamp:
                    cov_mat = self.cov_matrix_vec[idx,:,:]

                    # Do outlier detection
                    curr_x = prediction[0]
                    curr_y = prediction[1]

                    # self.get_logger().info("Robots position: ({}, {})   Photo position: ({}, {})".format(curr_x, curr_y, self.photo_x, self.photo_y))

                    dist = np.sqrt( (self.photo_x-curr_x)**2 + (self.photo_y-curr_y)**2)
                    self.get_logger().info("Dist {}".format(dist)) 
                    if dist > 0.2 : #and not self.startingup:
                        self.get_logger().info("Outlier, skipping photo data") 
                        break
                    # else:
                    #     self.startingup = False

                    
                    # Calculate fused position, first update, then add prediction steps again
                    # with remapping
                    fused_pos = self.kalman.update(np.array([self.photo_x, self.photo_y, self.photo_theta]), prediction, cov_mat)   
                    # without remapping  
                    #fused_pos = self.kalman.update(np.array([m.pose.optom.position.x, m.pose.optom.position.y, (self.quat2eul(m.pose.optom.orientation) + np.pi) % (2 * np.pi) - np.pi]), prediction, cov_mat)
                    fused_pos = fused_pos + sum(self.vec_odom[idx:])

                    # self.get_logger().info("Fused pos: ({},{})".format(fused_pos[0], fused_pos[1]))

                    # self.get_logger().info("Photo pos: ({},{})".format(self.photo_x, self.photo_y))

                    # Update kalman internal variable
                    self.kalman.vec_xest = fused_pos
                    self.kalman.mat_state_cov = self.cov_matrix_vec[-1]

                    # self.get_logger().info("x: {}, y: {}, theta: {}".format(self.photo_x, self.photo_y, self.photo_theta))

                    # Create and populate messages and then publish
                    msg_fused = ATRStateStamped()
                    msg_fused.state.pose.fused_odom.position.z = 0.0
                    msg_fused.state.pose.fused_odom.position.x = fused_pos[0]
                    msg_fused.state.pose.fused_odom.position.y = fused_pos[1]
                    msg_fused.state.pose.fused_odom.orientation = self.eul2quat(fused_pos[2])
                    msg_fused.header.frame_id = "/base_link"
                    msg_fused.header.stamp = self.get_clock().now().to_msg() #self.timestamp_vec[-1:]    #self.get_clock().now().to_msg()
                    msg_fused.state.pose_source = ATRState.FUSED_ODOM
                    msg_fused.state.atr_id = self.id

                    msg_fused.state.pose.optom.position.z = 0.0
                    msg_fused.state.pose.optom.position.x = self.photo_x
                    msg_fused.state.pose.optom.position.y = self.photo_y
                    msg_fused.state.pose.optom.orientation = self.eul2quat(self.photo_theta)
                    self.pub_.publish(msg_fused)

                    msg_rviz = PoseStamped()
                    msg_rviz.header = msg_fused.header
                    msg_rviz.pose = msg_fused.state.pose.fused_odom
                    self.pub_rviz.publish(msg_rviz)

                break
        
        
    def eul2quat(self,theta):
        orientation = PoseStamped().pose.orientation
        orientation.w, orientation.x, orientation.y, orientation.z  = euler2quat(0, 0, theta)
        return orientation

    def quat2eul(self, orientation):
        _,_,theta = quat2euler([orientation.w, orientation.x, orientation.y, orientation.z])
        return theta #Convertion to radians

def main(args=None):
    # Setting up some ROS stuff
    rclpy.init(args=args)

    path = get_package_share_directory('observer')
    #config_file='../../../../config.json'
    #config = utils_config.load_config(os.path.join(path,config_file))
    config = utils_config.load_config('./config.json')
    
    
    observer_node = Observer(config,KalmanFilter())
    observer_node.get_logger().info("start spin")
    rclpy.spin(observer_node)    
    observer_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
