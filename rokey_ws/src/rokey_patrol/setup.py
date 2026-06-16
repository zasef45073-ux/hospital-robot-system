import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'rokey_patrol'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'assets'),
         glob('assets/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='rokey@example.com',
    description='Patrol planner + person pose detection with audio feedback.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'amr_detect            = rokey_patrol.amr_detect:main',
            'patrol_node_april     = rokey_patrol.patrol_node_april:main',
            'enji_auto_localization = rokey_patrol.enji_auto_localization:main',
        ],
    },
)
