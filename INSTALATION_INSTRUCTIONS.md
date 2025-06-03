# Instalation instructions
1. Install Ubuntu 22.
2. Install ROS2 Jazzy according to [instructions](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html).
3. Add:
```
source /opt/ros/jazzy/setup.bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
```
to the end of your startup script:
```
code ~/.bashrc
```
Then apply changes using:
```
source ~/.bashrc
```
4. Source custom WS: source ~/your_ros2_ws/install/setup.bash
5. Test by running Rviz with: 
```
rviz2
```
6. Install and launch rqt using:
```
sudo apt update
sudo apt install '~nros-jazzy-rqt*'
rqt
```
7. Install and launch Gazebo for jazzy:
```
sudo apt-get install ros-${ROS_DISTRO}-ros-gz
ros2 launch ros_gz_sim gz_sim.launch.py gz_args:=empty.sdf
```