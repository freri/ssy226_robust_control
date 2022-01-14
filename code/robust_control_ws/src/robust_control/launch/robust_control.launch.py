from launch import LaunchDescription
from launch_ros.actions import Node
import utils_config
import sys, os
from pathlib import Path
def generate_launch_description():
    
    node_list = []
    fpath = Path(__file__)
    fpath = os.path.join(str(fpath.parent.parent), "config", "config.json")
    config = utils_config.load_config(fpath)
    atr_id = config.base.atr_id

    atr_name = '_atr_' + str(atr_id)

    node_list.append(Node(
            package='robust_control',
            executable='robust_control',
            name='robust_control' + atr_name))

    node_list.append(Node(
        package='robust_control',
        executable='trajectory_generator',
        name='trajectory_generator'+atr_name
    ))

    return LaunchDescription(node_list)
