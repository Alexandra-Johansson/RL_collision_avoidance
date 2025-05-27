# ROS2 Humble cheet sheet

## Create a package with a node from src folder
```
ros2 pkg create --build-type ament_python --license Apache-2.0 --node-name my_node my_package
```

## Build and source packages using from ros2_ws folder:
```
rm -rf build/ install/ log/
colcon build
source install/local_setup.bash
```

## Only build specific package
```
colcon build --packages-select my_package
```

## Run a node by using:
```
ros2 run my_package my_node
```