from setuptools import setup
import os
from glob import glob
package_name = 'robust_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    py_modules=[
        'robust_control.atr_model',
        'robust_control.mpc_builder',
        'robust_control.trajectory_generator',
        'robust_control.plotter'
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.json'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='freri',
    maintainer_email='frcar@student.chalmers.se',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robust_control = robust_control.robust_control:main',
            'trajectory_generator = robust_control.trajectory_generator:main'
        ],
    },
)
