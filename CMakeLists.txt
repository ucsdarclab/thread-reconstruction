cmake_minimum_required(VERSION 3.0.2)
project("thread_reconstruction")

# Get the information about this package's buildtime dependencies
find_package(catkin REQUIRED COMPONENTS
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    message_generation
)

# # Declare the message files to be built
# add_message_files(FILES
# MyMessage1.msg
# MyMessage2.msg
# )

# Declare the service files to be built
add_service_files(FILES
    GetGraspPoint.srv
)

# Actually generate the language-specific message and service files
generate_messages(DEPENDENCIES std_msgs geometry_msgs)

# Declare that this catkin package's runtime dependencies
catkin_package(CATKIN_DEPENDS message_runtime)

include_directories(
#   include
  ${catkin_INCLUDE_DIRS}
)