cmake_minimum_required(VERSION 3.0.2)
project(unidepth_ros)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  sensor_msgs
  std_msgs
  tf2_ros
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS cv_bridge rospy sensor_msgs std_msgs tf2_ros
)

catkin_install_python(PROGRAMS
  scripts/unidepth_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
