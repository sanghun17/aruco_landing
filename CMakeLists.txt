cmake_minimum_required(VERSION 3.0.2)
project(aruco_landing)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

if(EXISTS "/opt/opencv-aruco/lib/cmake/opencv4/OpenCVConfig.cmake")
  set(OpenCV_DIR "/opt/opencv-aruco/lib/cmake/opencv4")
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d aruco)
find_package(yaml-cpp REQUIRED)

catkin_python_setup()
catkin_package(CATKIN_DEPENDS geometry_msgs roscpp sensor_msgs std_msgs)

add_executable(paper_pad_estimator_node src/paper_pad_estimator_node.cpp)
target_compile_features(paper_pad_estimator_node PRIVATE cxx_std_14)
target_compile_options(paper_pad_estimator_node PRIVATE -O3 -Wall -Wextra -Wpedantic)
target_include_directories(paper_pad_estimator_node PRIVATE
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(paper_pad_estimator_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  yaml-cpp
)
if(EXISTS "/opt/opencv-aruco/lib")
  set_target_properties(paper_pad_estimator_node PROPERTIES
    BUILD_RPATH "/opt/opencv-aruco/lib"
    INSTALL_RPATH "/opt/opencv-aruco/lib"
  )
endif()

catkin_install_python(PROGRAMS
  scripts/aruco_detector_node.py
  scripts/generate_paper_pad.py
  scripts/landing_controller_node.py
  scripts/pad_relative_vehicle_state.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(TARGETS paper_pad_estimator_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
