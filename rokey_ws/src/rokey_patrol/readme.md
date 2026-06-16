**# what to run for each terminal (for robot1)**



**Terminal 1**

ros2 launch turtlebot4\_navigation localization.launch.py namespace:=/robot1 map =/home/rokey/rokey\_ws/maps/hospital\_map.yaml

**Terminal 2**

ros2 launch turtlebot4\_viz view\_robot.launch.py namespace:=/robot1

**Terminal 3**

ros2 launch turtlebot4\_navigation nav2.launch.py namespace:=/robot1

**Terminal 4**

ros2 run rokey\_patrol amr\_detect --ros-args -r \_\_ns:=/robot1

**Terminal 5**

ros2 run rokey\_patrol enji\_auto\_localization --ros-args -r \_\_ns:=/robot1

**Terminal 6**

ros2 run rokey\_patrol patrol\_node\_april --ros-args -r \_\_ns:=/robot1

**Terminal 7(optional)**

ros2 topic echo /patrol\_status

**Terminal 8(optional)**

ros2 run teleop\_twist\_keyboard teleop\_twist\_keyboard --ros-args -r /cmd\_vel:=/robot1/cmd\_vel

