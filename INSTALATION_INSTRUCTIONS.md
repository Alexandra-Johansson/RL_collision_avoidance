# Instalation instructions
1. Install Ubuntu 24.
2. Create a virtual environment for python
```
python -m venv env_name
source env_name/bin/activate
```
To deactivate the environment use:
```
deactivate
```
3. Install ROS2 Jazzy according to [instructions](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html).
4. Add:
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
5. Source custom WS: source ~/your_ros2_ws/install/setup.bash
6. Test by running Rviz with: 
```
rviz2
```
7. Install and launch rqt using:
```
sudo apt update
sudo apt install '~nros-jazzy-rqt*'
rqt
```
8. Install and launch Gazebo for jazzy:
```
sudo apt-get install ros-${ROS_DISTRO}-ros-gz
ros2 launch ros_gz_sim gz_sim.launch.py gz_args:=empty.sdf
```
9. Install gym-pybullet-drones:
Move into the gym-pybullet-drones directory
```
pip3 install -e .
```
10. Install all dependencies
```
pip install -r requirements.txt
```