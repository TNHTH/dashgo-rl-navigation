# DashGo ROS2 实车迁移代码与测试汇总

> 创建时间: 2026-03-20
> 汇总范围: 本次实车 ROS2 原生迁移直接相关的参数基线、权威 ROS1 驱动、ROS2 底盘驱动、ROS2 单雷达驱动、实车导航接入层、测试文件和上车部署说明。
> 文件数量: 41

## 说明

- 本文件用于一次性查看本次实车迁移涉及的关键源码和测试。
- 为避免遗漏，保留了 ROS1 权威基线文件与 ROS2 实现文件的并列展示。
- 如果后续继续修改源码，应以原始文件为准，这个汇总文件是快照，不应替代源文件本身。

## 文件清单

- `drivers/EAI_DRIVER/README.md`
- `drivers/EAI_DRIVER/src/CMakeLists.txt`
- `drivers/EAI_DRIVER/src/package.xml`
- `drivers/EAI_DRIVER/src/config/my_dashgo_params.yaml`
- `drivers/EAI_DRIVER/src/nodes/dashgo_driver.py`
- `drivers/lakibeam_driver/src/launch/lakibeam1_scan.launch`
- `drivers/lakibeam_driver/src/src/lakibeam1_scan.cpp`
- `drivers/lakibeam_driver/src/src/remote.cpp`
- `workspaces/ros2_ws/src/dashgo_driver_ros2/package.xml`
- `workspaces/ros2_ws/src/dashgo_driver_ros2/setup.py`
- `workspaces/ros2_ws/src/dashgo_driver_ros2/setup.cfg`
- `workspaces/ros2_ws/src/dashgo_driver_ros2/config/dashgo_driver.yaml`
- `workspaces/ros2_ws/src/dashgo_driver_ros2/config/laser_static_tf.yaml`
- `workspaces/ros2_ws/src/dashgo_driver_ros2/dashgo_driver_ros2/__init__.py`
- `workspaces/ros2_ws/src/dashgo_driver_ros2/dashgo_driver_ros2/driver_core.py`
- `workspaces/ros2_ws/src/dashgo_driver_ros2/dashgo_driver_ros2/driver_node.py`
- `workspaces/ros2_ws/src/dashgo_driver_ros2/dashgo_driver_ros2/static_tf_node.py`
- `workspaces/ros2_ws/src/dashgo_driver_ros2/launch/base_only.launch.py`
- `workspaces/ros2_ws/src/dashgo_driver_ros2/tests/test_driver_core.py`
- `workspaces/ros2_ws/src/dashgo_driver_ros2/tests/test_parameter_alignment.py`
- `workspaces/ros2_ws/src/lakibeam_driver_ros2/package.xml`
- `workspaces/ros2_ws/src/lakibeam_driver_ros2/CMakeLists.txt`
- `workspaces/ros2_ws/src/lakibeam_driver_ros2/config/lakibeam_driver.yaml`
- `workspaces/ros2_ws/src/lakibeam_driver_ros2/include/lakibeam_driver_ros2/data_type.h`
- `workspaces/ros2_ws/src/lakibeam_driver_ros2/include/lakibeam_driver_ros2/remote.h`
- `workspaces/ros2_ws/src/lakibeam_driver_ros2/src/remote.cpp`
- `workspaces/ros2_ws/src/lakibeam_driver_ros2/src/lakibeam_scan_node.cpp`
- `workspaces/ros2_ws/src/lakibeam_driver_ros2/launch/lidar_only.launch.py`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/package.xml`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/setup.py`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/config/dashgo_rl.yaml`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/config/nav2_planning.yaml`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/controller_core.py`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/safety_filter.py`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/goal_plan_bridge.py`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/launch/real_model_nav.launch.py`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/launch/real_robot_nav.launch.py`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_controller_core.py`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_safety_filter.py`
- `docs/07-ros2-migration/dashgo-real-robot-ros2-deployment_2026-03-20.md`

## `drivers/EAI_DRIVER/README.md`

```markdown
驱动流程：

解压zip，创建工作空间为EAI，将解压后的src文件放入EAI文件夹中

1. cd EAI
2. catkin_make
3. source ./devel/setup.bash
4. sudo chmod 666 dev/ttyUSB0 (如有报错请上网搜索一下自己的USB设备是哪个，再加串口权限)
5. roslaunch dashgo_bringup minimal.launch

检查是否连接：

rosrun teleop_twist_keyboard teleop_twist_keyboard.py 

键盘启动节点，没装就装一下。

说明：

1. 当前实际启用的 ROS1 主节点是 `src/nodes/dashgo_driver.py`，它已经是 Python 3 版本。
2. `dashgo_driver_fl.py`、`dashgo_driver_sm.py`、`dashgo_driver_sm_ob.py` 仍是旧的 Python 2 历史变体，仅保留参考，不建议继续接入新链路。
3. 若实机长期使用，优先通过 `src/startup/create_dashgo_udev.sh` 生成稳定串口名 `/dev/dashgo`，不要长期依赖 `/dev/ttyUSB0`。

```

## `drivers/EAI_DRIVER/src/CMakeLists.txt`

```text
cmake_minimum_required(VERSION 2.8.3)
project(dashgo_bringup)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  roslib
  rospy
  std_msgs
  tf
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp roslib rospy std_msgs tf
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(dashgo_bringup
#   src/${PROJECT_NAME}/dashgo_bringup.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(dashgo_bringup ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
# add_executable(dashgo_bringup_node src/dashgo_bringup_node.cpp)

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(dashgo_bringup_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(dashgo_bringup_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
  nodes/dashgo_driver.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark executables and/or libraries for installation
# install(TARGETS dashgo_bringup dashgo_bringup_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
install(DIRECTORY config launch scripts startup
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_dashgo_bringup.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)

```

## `drivers/EAI_DRIVER/src/package.xml`

```xml
<?xml version="1.0"?>
<package format="2">
  <name>dashgo_bringup</name>
  <version>0.0.0</version>
  <description>The dashgo_bringup package</description>

  <!-- One maintainer tag required, multiple allowed, one person per tag --> 
  <!-- Example:  -->
  <!-- <maintainer email="jane.doe@example.com">Jane Doe</maintainer> -->
  <maintainer email="harney@todo.todo">harney</maintainer>


  <!-- One license tag required, multiple allowed, one license per tag -->
  <!-- Commonly used license strings: -->
  <!--   BSD, MIT, Boost Software License, GPLv2, GPLv3, LGPLv2.1, LGPLv3 -->
  <license>TODO</license>


  <!-- Url tags are optional, but mutiple are allowed, one per tag -->
  <!-- Optional attribute type can be: website, bugtracker, or repository -->
  <!-- Example: -->
  <!-- <url type="website">http://wiki.ros.org/dashgo_bringup</url> -->


  <!-- Author tags are optional, mutiple are allowed, one per tag -->
  <!-- Authors do not have to be maintianers, but could be -->
  <!-- Example: -->
  <!-- <author email="jane.doe@example.com">Jane Doe</author> -->


  <!-- The *_depend tags are used to specify dependencies -->
  <!-- Dependencies can be catkin packages or system dependencies -->
  <!-- Examples: -->
  <!-- Use build_depend for packages you need at compile time: -->
  <!--   <build_depend>message_generation</build_depend> -->
  <!-- Use buildtool_depend for build tool packages: -->
  <!--   <buildtool_depend>catkin</buildtool_depend> -->
  <!-- Use run_depend for packages you need at runtime: -->
  <!--   <run_depend>message_runtime</run_depend> -->
  <!-- Use test_depend for packages you need only for testing: -->
  <!--   <test_depend>gtest</test_depend> -->
  <buildtool_depend>catkin</buildtool_depend>

  <depend>geometry_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>roscpp</depend>
  <depend>roslib</depend>
  <depend>rospy</depend>
  <depend>std_msgs</depend>
  <depend>tf</depend>

  <exec_depend>python3-serial</exec_depend>


  <!-- The export tag contains other, unspecified, tags -->
  <export>
    <!-- Other tools can request additional information be placed here -->

  </export>
</package>

```

## `drivers/EAI_DRIVER/src/config/my_dashgo_params.yaml`

```yaml
port: /dev/ttyUSB0
baud: 115200
timeout: 0.1
rate: 50
sensorstate_rate: 10

use_base_controller: True
base_controller_rate: 10

# For a robot that uses base_footprint, change base_frame to base_footprint
base_frame: base_link
#base_frame: base_footprint

# === Robot drivetrain parameters
wheel_diameter: 0.1264
#wheel_track: 0.3550
wheel_track: 0.3420
encoder_resolution: 1200 # from Pololu for 13*34*4 motors
gear_reduction: 1.0
motors_reversed: False

# === PID parameters
Kp: 50
Kd: 20
Ki: 0
Ko: 50
accel_limit: 1.0

```

## `drivers/EAI_DRIVER/src/nodes/dashgo_driver.py`

```python
#!/usr/bin/env python3

"""
    A ROS Node for the Arduino microcontroller
    
    Created for the Pi Robot Project: http://www.pirobot.org
    Copyright (c) 2012 Patrick Goebel.  All rights reserved.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details at:
    
    http://www.gnu.org/licenses/gpl.html
"""

import rospy
from geometry_msgs.msg import Twist
import os, time
import _thread

from math import pi as PI, degrees, radians, sin, cos
import os
import time
import sys, traceback
from serial.serialutil import SerialException
from serial import Serial

import roslib

from geometry_msgs.msg import Quaternion, Twist, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import Int16
from tf.broadcaster import TransformBroadcaster
 
ODOM_POSE_COVARIANCE = [1e-3, 0, 0, 0, 0, 0, 
                        0, 1e-3, 0, 0, 0, 0,
                        0, 0, 1e6, 0, 0, 0,
                        0, 0, 0, 1e6, 0, 0,
                        0, 0, 0, 0, 1e6, 0,
                        0, 0, 0, 0, 0, 1e3]
ODOM_POSE_COVARIANCE2 = [1e-9, 0, 0, 0, 0, 0, 
                         0, 1e-3, 1e-9, 0, 0, 0,
                         0, 0, 1e6, 0, 0, 0,
                         0, 0, 0, 1e6, 0, 0,
                         0, 0, 0, 0, 1e6, 0,
                         0, 0, 0, 0, 0, 1e-9]

ODOM_TWIST_COVARIANCE = [1e-3, 0, 0, 0, 0, 0, 
                         0, 1e-3, 0, 0, 0, 0,
                         0, 0, 1e6, 0, 0, 0,
                         0, 0, 0, 1e6, 0, 0,
                         0, 0, 0, 0, 1e6, 0,
                         0, 0, 0, 0, 0, 1e3]
ODOM_TWIST_COVARIANCE2 = [1e-9, 0, 0, 0, 0, 0, 
                          0, 1e-3, 1e-9, 0, 0, 0,
                          0, 0, 1e6, 0, 0, 0,
                          0, 0, 0, 1e6, 0, 0,
                          0, 0, 0, 0, 1e6, 0,
                          0, 0, 0, 0, 0, 1e-9]


SERVO_MAX = 180
SERVO_MIN = 0

class Arduino:
    ''' Configuration Parameters
    '''    
    N_ANALOG_PORTS = 6
    N_DIGITAL_PORTS = 12
    
    def __init__(self, port="/dev/ttyUSB0", baudrate=57600, timeout=0.5):
        
        self.PID_RATE = 30 # Do not change this!  It is a fixed property of the Arduino PID controller.
        self.PID_INTERVAL = 1000 / 30
        
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.encoder_count = 0
        self.writeTimeout = timeout
        self.interCharTimeout = timeout / 30.
    
        # Keep things thread safe
        self.mutex = _thread.allocate_lock()
            
        # An array to cache analog sensor readings
        self.analog_sensor_cache = [None] * self.N_ANALOG_PORTS
        
        # An array to cache digital sensor readings
        self.digital_sensor_cache = [None] * self.N_DIGITAL_PORTS
    
    def connect(self):
        try:
            print("Connecting to Arduino on port", self.port, "...")
            self.port = Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout, writeTimeout=self.writeTimeout)
            # The next line is necessary to give the firmware time to wake up.
            time.sleep(1)
            test = self.get_baud()
            if test != self.baudrate:
                time.sleep(1)
                test = self.get_baud()   
                if test != self.baudrate:
                    raise SerialException
            print("Connected at", self.baudrate)
            print("Arduino is ready.")

        except SerialException:
            print("Serial Exception:")
            print(sys.exc_info())
            print("Traceback follows:")
            traceback.print_exc(file=sys.stdout)
            print("Cannot connect to Arduino!")
            os._exit(1)

    def open(self): 
        ''' Open the serial port.
        '''
        self.port.open()

    def close(self): 
        ''' Close the serial port.
        '''
        self.port.close() 
    
    def send(self, cmd):
        ''' This command should not be used on its own: it is called by the execute commands
            below in a thread safe manner.
        '''
        self.port.write((cmd + '\r').encode('UTF-8'))

    def recv(self, timeout=0.5):
        timeout = min(timeout, self.timeout)
        ''' This command should not be used on its own: it is called by the execute commands   
            below in a thread safe manner.  Note: we use read() instead of readline() since
            readline() tends to return garbage characters from the Arduino
        '''
        c = ''
        value = ''
        attempts = 0
        while c != '\r':
            c = self.port.read(1).decode('UTF-8')
            value += c
            attempts += 1
            if attempts * self.interCharTimeout > timeout:
                return None

        value = value.strip('\r')

        return value
            
    def recv_ack(self):
        ''' This command should not be used on its own: it is called by the execute commands
            below in a thread safe manner.
        '''
        ack = self.recv(self.timeout)
        return ack == 'OK'

    def recv_int(self):
        ''' This command should not be used on its own: it is called by the execute commands
            below in a thread safe manner.
        '''
        value = self.recv(self.timeout)
        try:
            return int(value)
        except:
            return None

    def recv_array(self):
        ''' This command should not be used on its own: it is called by the execute commands
            below in a thread safe manner.
        '''
        try:
            values = self.recv(self.timeout * self.N_ANALOG_PORTS).split()
            return list(map(int, values))
        except:
            return []

    def execute(self, cmd):
        ''' Thread safe execution of "cmd" on the Arduino returning a single integer value.
        '''
        self.mutex.acquire()
        
        try:
            self.port.flushInput()
        except:
            pass
        
        ntries = 1
        attempts = 0
        
        try:
            self.port.write((cmd + '\r').encode('UTF-8'))
            value = self.recv(self.timeout)
            while attempts < ntries and (value == '' or value == 'Invalid Command' or value == None):
                try:
                    self.port.flushInput()
                    self.port.write((cmd + '\r').encode('UTF-8'))
                    value = self.recv(self.timeout)
                except:
                    print("Exception executing command: " + cmd)
                attempts += 1
        except:
            self.mutex.release()
            print("Exception executing command: " + cmd)
            value = None
        
        self.mutex.release()
        return int(value)

    def execute_array(self, cmd):
        ''' Thread safe execution of "cmd" on the Arduino returning an array.
        '''
        self.mutex.acquire()
        
        try:
            self.port.flushInput()
        except:
            pass
        
        ntries = 1
        attempts = 0
        
        try:
            self.port.write((cmd + '\r').encode('UTF-8'))
            values = self.recv_array()
            while attempts < ntries and (values == '' or values == 'Invalid Command' or values == [] or values == None):
                try:
                    self.port.flushInput()
                    self.port.write((cmd + '\r').encode('UTF-8'))
                    values = self.recv_array()
                except:
                    print(("Exception executing command: " + cmd))
                attempts += 1
        except:
            self.mutex.release()
            print("Exception executing command: " + cmd)
            raise SerialException
            return []
        
        try:
            values = list(map(int, values))
        except:
            values = []

        self.mutex.release()
        return values
        
    def execute_ack(self, cmd):
        ''' Thread safe execution of "cmd" on the Arduino returning True if response is ACK.
        '''
        self.mutex.acquire()
        
        try:
            self.port.flushInput()
        except:
            pass
        
        ntries = 1
        attempts = 0
        
        try:
            self.port.write((cmd + '\r').encode('UTF-8'))
            ack = self.recv(self.timeout)
            while attempts < ntries and (ack == '' or ack == 'Invalid Command' or ack == None):
                try:
                    self.port.flushInput()
                    self.port.write((cmd + '\r').encode('UTF-8'))
                    ack = self.recv(self.timeout)
                except:
                    print("Exception executing command: " + cmd)
            attempts += 1
        except:
            self.mutex.release()
            print("execute_ack exception when executing", cmd)
            print(sys.exc_info())
            return 0
        
        self.mutex.release()
        return ack == 'OK'   
    
    def update_pid(self, Kp, Kd, Ki, Ko):
        ''' Set the PID parameters on the Arduino
        '''
        print("Updating PID parameters")
        cmd = 'u ' + str(Kp) + ':' + str(Kd) + ':' + str(Ki) + ':' + str(Ko)
        self.execute_ack(cmd)                          

    def get_baud(self):
        ''' Get the current baud rate on the serial port.
        '''
        return int(self.execute('b'));

    def get_encoder_counts(self):
        values = self.execute_array('e')
        if len(values) != 2:
            print("Encoder count was not 2")
            raise SerialException
            return None
        else:
            return values

    def reset_encoders(self):
        ''' Reset the encoder counts to 0
        '''
        return self.execute_ack('r')
    
    def drive(self, right, left):
        ''' Speeds are given in encoder ticks per PID interval
        '''
        return self.execute_ack('m %d %d' %(right, left))
    
    def drive_m_per_s(self, right, left):
        ''' Set the motor speeds in meters per second.
        '''
        left_revs_per_second = float(left) / (self.wheel_diameter * PI)
        right_revs_per_second = float(right) / (self.wheel_diameter * PI)

        left_ticks_per_loop = int(left_revs_per_second * self.encoder_resolution * self.PID_INTERVAL * self.gear_reduction)
        right_ticks_per_loop  = int(right_revs_per_second * self.encoder_resolution * self.PID_INTERVAL * self.gear_reduction)

        self.drive(right_ticks_per_loop , left_ticks_per_loop )
        
    def stop(self):
        ''' Stop both motors.
        '''
        self.drive(0, 0)

    def ping(self, pin):
        ''' The srf05/Ping command queries an SRF05/Ping sonar sensor
            connected to the General Purpose I/O line pinId for a distance,
            and returns the range in cm.  Sonar distance resolution is integer based.
        '''
        return self.execute('p %d' %pin);

    def get_pidin(self):
        values = self.execute_array('i')
        if len(values) != 2:
            print("get_pidin count was not 2")
            raise SerialException
            return None
        else:
            return values

    def get_pidout(self):
        values = self.execute_array('f')
        if len(values) != 2:
            print("get_pidout count was not 2")
            raise SerialException
            return None
        else:
            return values


""" Class to receive Twist commands and publish Odometry data """
class BaseController:
    def __init__(self, arduino, base_frame):
        self.arduino = arduino
        self.base_frame = base_frame
        self.rate = float(rospy.get_param("~base_controller_rate", 10))
        self.timeout = rospy.get_param("~base_controller_timeout", 1.0)
        self.stopped = False
                 
        pid_params = dict()
        pid_params['wheel_diameter'] = rospy.get_param("~wheel_diameter", "") 
        pid_params['wheel_track'] = rospy.get_param("~wheel_track", "")
        pid_params['encoder_resolution'] = rospy.get_param("~encoder_resolution", "") 
        pid_params['gear_reduction'] = rospy.get_param("~gear_reduction", 1.0)
        pid_params['Kp'] = rospy.get_param("~Kp", 20)
        pid_params['Kd'] = rospy.get_param("~Kd", 12)
        pid_params['Ki'] = rospy.get_param("~Ki", 0)
        pid_params['Ko'] = rospy.get_param("~Ko", 50)
        
        self.accel_limit = rospy.get_param('~accel_limit', 0.1)
        self.motors_reversed = rospy.get_param("~motors_reversed", False)
        
        # Set up PID parameters and check for missing values
        self.setup_pid(pid_params)
            
        # How many encoder ticks are there per meter?
        self.ticks_per_meter = self.encoder_resolution * self.gear_reduction  / (self.wheel_diameter * PI)
        
        # What is the maximum acceleration we will tolerate when changing wheel speeds?
        self.max_accel = self.accel_limit * self.ticks_per_meter / self.rate
                
        # Track how often we get a bad encoder count (if any)
        self.bad_encoder_count = 0

        self.encoder_min = rospy.get_param('encoder_min', -32768)
        self.encoder_max = rospy.get_param('encoder_max', 32768)
        self.encoder_low_wrap = rospy.get_param('wheel_low_wrap', (self.encoder_max - self.encoder_min) * 0.3 + self.encoder_min )
        self.encoder_high_wrap = rospy.get_param('wheel_high_wrap', (self.encoder_max - self.encoder_min) * 0.7 + self.encoder_min )
        self.l_wheel_mult = 0
        self.r_wheel_mult = 0
                        
        now = rospy.Time.now()    
        self.then = now # time for determining dx/dy
        self.t_delta = rospy.Duration(1.0 / self.rate)
        self.t_next = now + self.t_delta

        # Internal data        
        self.enc_left = None            # encoder readings
        self.enc_right = None
        self.x = 0                      # position in xy plane
        self.y = 0
        self.th = 0                     # rotation in radians
        self.v_left = 0
        self.v_right = 0
        self.v_des_left = 0             # cmd_vel setpoint
        self.v_des_right = 0
        self.last_cmd_vel = now

        # Subscriptions
        rospy.Subscriber("cmd_vel", Twist, self.cmdVelCallback)
        
        # Clear any old odometry info
        self.arduino.reset_encoders()
        
        # Set up the odometry broadcaster
        self.odomPub = rospy.Publisher('odom', Odometry, queue_size=5)
        self.odomBroadcaster = TransformBroadcaster()
        
        rospy.loginfo("Started base controller for a base of " + str(self.wheel_track) + "m wide with " + str(self.encoder_resolution) + " ticks per rev")
        rospy.loginfo("Publishing odometry data at: " + str(self.rate) + " Hz using " + str(self.base_frame) + " as base frame")

        self.lEncoderPub = rospy.Publisher('Lencoder', Int16)
        self.rEncoderPub = rospy.Publisher('Rencoder', Int16)
        self.lPidoutPub = rospy.Publisher('Lpidout', Int16)
        self.rPidoutPub = rospy.Publisher('Rpidout', Int16)
        self.lVelPub = rospy.Publisher('Lvel', Int16)
        self.rVelPub = rospy.Publisher('Rvel', Int16)
        
    def setup_pid(self, pid_params):
        # Check to see if any PID parameters are missing
        missing_params = False
        for param in pid_params:
            if pid_params[param] == "":
                print(("*** PID Parameter " + param + " is missing. ***"))
                missing_params = True
        
        if missing_params:
            os._exit(1)
                
        self.wheel_diameter = pid_params['wheel_diameter']
        self.wheel_track = pid_params['wheel_track']
        self.encoder_resolution = pid_params['encoder_resolution']
        self.gear_reduction = pid_params['gear_reduction']
        
        self.Kp = pid_params['Kp']
        self.Kd = pid_params['Kd']
        self.Ki = pid_params['Ki']
        self.Ko = pid_params['Ko']
        
        self.arduino.update_pid(self.Kp, self.Kd, self.Ki, self.Ko)

    def poll(self):
        now = rospy.Time.now()
        if now > self.t_next:
            try:
                left_pidin, right_pidin = self.arduino.get_pidin()
            except:
                rospy.logerr("getpidout exception count: ")
                return

            self.lEncoderPub.publish(left_pidin)
            self.rEncoderPub.publish(right_pidin)
            try:
                left_pidout, right_pidout = self.arduino.get_pidout()
            except:
                rospy.logerr("getpidout exception count: ")
                return
            self.lPidoutPub.publish(left_pidout)
            self.rPidoutPub.publish(right_pidout)
            # Read the encoders
            try:
                left_enc, right_enc = self.arduino.get_encoder_counts()
                #rospy.loginfo("left_enc: " + str(left_enc)+"right_enc: " + str(right_enc))
            except:
                self.bad_encoder_count += 1
                rospy.logerr("Encoder exception count: " + str(self.bad_encoder_count))
                return
                            
            dt = now - self.then
            self.then = now
            dt = dt.to_sec()
            
            # Calculate odometry
            if self.enc_left == None:
                dright = 0
                dleft = 0
            else:
                if (left_enc < self.encoder_low_wrap and self.enc_left > self.encoder_high_wrap) :
                    self.l_wheel_mult = self.l_wheel_mult + 1     
                elif (left_enc > self.encoder_high_wrap and self.enc_left < self.encoder_low_wrap) :
                    self.l_wheel_mult = self.l_wheel_mult - 1
                else:
                     self.l_wheel_mult = 0
                if (right_enc < self.encoder_low_wrap and self.enc_right > self.encoder_high_wrap) :
                    self.r_wheel_mult = self.r_wheel_mult + 1     
                elif (right_enc > self.encoder_high_wrap and self.enc_right < self.encoder_low_wrap) :
                    self.r_wheel_mult = self.r_wheel_mult - 1
                else:
                     self.r_wheel_mult = 0
                #dright = (right_enc - self.enc_right) / self.ticks_per_meter
                #dleft = (left_enc - self.enc_left) / self.ticks_per_meter
                dleft = 1.0 * (left_enc + self.l_wheel_mult * (self.encoder_max - self.encoder_min)-self.enc_left) / self.ticks_per_meter 
                dright = 1.0 * (right_enc + self.r_wheel_mult * (self.encoder_max - self.encoder_min)-self.enc_right) / self.ticks_per_meter 

            self.enc_right = right_enc
            self.enc_left = left_enc
            
            dxy_ave = (dright + dleft) / 2.0
            dth = (dright - dleft) / self.wheel_track
            vxy = dxy_ave / dt
            vth = dth / dt
                
            if (dxy_ave != 0):
                dx = cos(dth) * dxy_ave
                dy = -sin(dth) * dxy_ave
                self.x += (cos(self.th) * dx - sin(self.th) * dy)
                self.y += (sin(self.th) * dx + cos(self.th) * dy)
    
            if (dth != 0):
                self.th += dth 
    
            quaternion = Quaternion()
            quaternion.x = 0.0 
            quaternion.y = 0.0
            quaternion.z = sin(self.th / 2.0)
            quaternion.w = cos(self.th / 2.0)
    
            # Create the odometry transform frame broadcaster.
            self.odomBroadcaster.sendTransform(
                (self.x, self.y, 0), 
                (quaternion.x, quaternion.y, quaternion.z, quaternion.w),
                rospy.Time.now(),
                self.base_frame,
                "odom"
                )
    
            odom = Odometry()
            odom.header.frame_id = "odom"
            odom.child_frame_id = self.base_frame
            odom.header.stamp = now
            odom.pose.pose.position.x = self.x
            odom.pose.pose.position.y = self.y
            odom.pose.pose.position.z = 0
            odom.pose.pose.orientation = quaternion
            odom.twist.twist.linear.x = vxy
            odom.twist.twist.linear.y = 0
            odom.twist.twist.angular.z = vth

            # todo sensor_state.distance == 0
            if self.v_des_left == 0 and self.v_des_right == 0:
                odom.pose.covariance = ODOM_POSE_COVARIANCE2
                odom.twist.covariance = ODOM_TWIST_COVARIANCE2
            else:
                odom.pose.covariance = ODOM_POSE_COVARIANCE
                odom.twist.covariance = ODOM_TWIST_COVARIANCE

            self.odomPub.publish(odom)
            
            if now > (self.last_cmd_vel + rospy.Duration(self.timeout)):
                self.v_des_left = 0
                self.v_des_right = 0
                
            if self.v_left < self.v_des_left:
                self.v_left += self.max_accel
                if self.v_left > self.v_des_left:
                    self.v_left = self.v_des_left
            else:
                self.v_left -= self.max_accel
                if self.v_left < self.v_des_left:
                    self.v_left = self.v_des_left
            
            if self.v_right < self.v_des_right:
                self.v_right += self.max_accel
                if self.v_right > self.v_des_right:
                    self.v_right = self.v_des_right
            else:
                self.v_right -= self.max_accel
                if self.v_right < self.v_des_right:
                    self.v_right = self.v_des_right
            self.lVelPub.publish(self.v_left)
            self.rVelPub.publish(self.v_right)            

            # Set motor speeds in encoder ticks per PID loop
            if not self.stopped:
                self.arduino.drive(self.v_left, self.v_right)
                
            self.t_next = now + self.t_delta
            
    def stop(self):
        self.stopped = True
        self.arduino.drive(0, 0)
            
    def cmdVelCallback(self, req):
        # Handle velocity-based movement requests
        self.last_cmd_vel = rospy.Time.now()
        
        x = req.linear.x         # m/s
        th = req.angular.z       # rad/s

        if x == 0:
            # Turn in place
            right = th * self.wheel_track  * self.gear_reduction / 2.0
            left = -right
        elif th == 0:
            # Pure forward/backward motion
            left = right = x
        else:
            # Rotation about a point in space
            left = x - th * self.wheel_track  * self.gear_reduction / 2.0
            right = x + th * self.wheel_track  * self.gear_reduction / 2.0
            
        self.v_des_left = int(left * self.ticks_per_meter / self.arduino.PID_RATE)
        self.v_des_right = int(right * self.ticks_per_meter / self.arduino.PID_RATE)

class ArduinoROS():
    def __init__(self):
        rospy.init_node('Arduino', log_level=rospy.DEBUG)
                
        # Cleanup when termniating the node
        rospy.on_shutdown(self.shutdown)
        
        self.port = rospy.get_param("~port", "/dev/ttyACM0")
        self.baud = int(rospy.get_param("~baud", 57600))
        self.timeout = rospy.get_param("~timeout", 0.5)
        self.base_frame = rospy.get_param("~base_frame", 'base_link')

        # Overall loop rate: should be faster than fastest sensor rate
        self.rate = int(rospy.get_param("~rate", 50))
        r = rospy.Rate(self.rate)

        # Rate at which summary SensorState message is published. Individual sensors publish
        # at their own rates.        
        self.sensorstate_rate = int(rospy.get_param("~sensorstate_rate", 10))
        
        self.use_base_controller = rospy.get_param("~use_base_controller", False)
        
        # Set up the time for publishing the next SensorState message
        now = rospy.Time.now()
        self.t_delta_sensors = rospy.Duration(1.0 / self.sensorstate_rate)
        self.t_next_sensors = now + self.t_delta_sensors
        
        # Initialize a Twist message
        self.cmd_vel = Twist()
  
        # A cmd_vel publisher so we can stop the robot when shutting down
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        
        # Initialize the controlller
        self.controller = Arduino(self.port, self.baud, self.timeout)
        
        # Make the connection
        self.controller.connect()
        
        rospy.loginfo("Connected to Arduino on port " + self.port + " at " + str(self.baud) + " baud")
     
        # Reserve a thread lock
        mutex = _thread.allocate_lock()
              
        # Initialize the base controller if used
        if self.use_base_controller:
            self.myBaseController = BaseController(self.controller, self.base_frame)
    
        # Start polling the sensors and base controller
        while not rospy.is_shutdown():
                    
            if self.use_base_controller:
                mutex.acquire()
                self.myBaseController.poll()
                mutex.release()
            r.sleep()
    
    def shutdown(self):
        # Stop the robot
        try:
            rospy.loginfo("Stopping the robot...")
            self.cmd_vel_pub.publish(Twist())
            rospy.sleep(2)
        except:
            pass
        rospy.loginfo("Shutting down Arduino Node...")
        
if __name__ == '__main__':
    myArduino = ArduinoROS()

```

## `drivers/lakibeam_driver/src/launch/lakibeam1_scan.launch`

```xml
<?xml version="1.0"?>

<launch>
    <node name="richbeam_lidar" pkg="lakibeam1" type="lakibeam1_scan_node" output="screen">
        <param name="frame_id" type="string" value="laser"/><!--frame_id设置-->
        <param name="output_topic" type="string" value="scan" /><!--topic设置-->
        <param name="inverted" type="bool" value="false"/><!--配置是否倒装,true倒装-->
        <param name="hostip" type="string" value="0.0.0.0"/><!--配置本机监听地址，0.0.0.0表示监听全部-->
        <param name="sensorip" type="string" value="192.168.8.2"/><!--配置sensor地址-->
        <param name="port" type="string" value="2368"/><!--配置本机监听端口-->
        <param name="angle_offset" type="int" value="0"/><!--配置点云旋转角度，可以是负数-->

        <param name="scanfreq" type="string" value="30" /><!--配置扫描频率，范围：10、20、25、30-->
        <param name="filter" type="string" value="3" /><!--配置滤波选项，范围：3、2、1、0 -->
        <param name="laser_enable" type="string" value="true" /><!--雷达扫描使能，范围：true、false-->
        <param name="scan_range_start" type="string" value="90" /><!--雷达扫描起始角度，范围：45~315-->
        <param name="scan_range_stop" type="string" value="270" /><!--雷达扫描结束角度，范围：45~315，结束角度必须大于起始角度-->
        <remap from="/richbeam_lidar/scan" to="/scan" />
    </node>

    <node pkg="tf" type="static_transform_publisher" name="base_link_to_laser0"
    args="0.0 0.0 0 0.0 0.0 0.0 /base_link /laser 40" />
</launch>

```

## `drivers/lakibeam_driver/src/src/lakibeam1_scan.cpp`

```cpp
#include <ros/ros.h> 
#include <sensor_msgs/LaserScan.h>

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <sched.h>

#include <sys/select.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <math.h>
#include "../include/data_type.h"
#include "../include/remote.h"

#define DEG2RAD(x) ((x)*M_PI / 180.f)

int main(int argc, char **argv)
{
	struct sockaddr_in ser_addr, clent_addr;
	std::string hostip, sensorip, port, frame_id, output_topic, scanfreq, filter, laser_enable, scan_range_start, scan_range_stop;
	std::vector <bm_response_scan_t> scan_vec;
	ros::Time scan_begin, scan_end;
	bool inverted;
	int i = 0, j = 12;
	int angle_offset = 0;
	int resolution = 25;
	int scan_vec_ready = 0;
	int sockfd;

	ros::init(argc, argv, "laser_scan_publisher");
  	ros::NodeHandle nh("~");
	ros::Rate rate(30); 
	
	nh.getParam("frame_id", frame_id);
	nh.getParam("output_topic", output_topic);
	nh.getParam("inverted", inverted);
	nh.getParam("hostip", hostip);
	nh.getParam("sensorip", sensorip);
	nh.getParam("port", port);
	nh.getParam("angle_offset", angle_offset);

	nh.getParam("scanfreq", scanfreq);
	nh.getParam("filter", filter);
	nh.getParam("laser_enable", laser_enable);
	nh.getParam("scan_range_start", scan_range_start);
	nh.getParam("scan_range_stop", scan_range_stop);
	std::cout<<output_topic<<std::endl;
	ros::Publisher scan_pub = nh.advertise<sensor_msgs::LaserScan> (output_topic, 1000);

	ROS_INFO("frame_id:%s", frame_id.c_str());
	ROS_INFO("output_topic:%s", output_topic.c_str());
	ROS_INFO("inverted:%s", (inverted ? "True" : "False"));
	ROS_INFO("hostip:%s", hostip.c_str());
	ROS_INFO("sensorip:%s", sensorip.c_str());
	ROS_INFO("port:%s", port.c_str());
	ROS_INFO("scanfreq:%s", scanfreq.c_str());
	ROS_INFO("filter:%s", filter.c_str());
	ROS_INFO("laser_enable:%s", laser_enable.c_str());
	ROS_INFO("scan_range_start:%s", scan_range_start.c_str());
	ROS_INFO("scan_range_stop:%s", scan_range_stop.c_str());

	sensor_config(sensorip, "/api/v1/sensor/scanfreq", scanfreq);
	//sensor_config(sensorip, "/api/v1/sensor/filter", filter);
	sensor_config(sensorip, "/api/v1/sensor/laser_enable", laser_enable);
	sensor_config(sensorip, "/api/v1/sensor/scan_range/start", scan_range_start);
	sensor_config(sensorip, "/api/v1/sensor/scan_range/stop", scan_range_stop);

	ros::Duration(2.0).sleep();
	get_telemetry_data(sensorip);

	sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if(sockfd == -1)
	{
		ROS_INFO("Failed to create socket");
		return -1;
    }

	memset(&ser_addr, 0, sizeof(ser_addr));
    ser_addr.sin_family = AF_INET;
	ser_addr.sin_addr.s_addr = inet_addr(hostip.c_str());
    ser_addr.sin_port = htons(atoi(port.c_str()));

    if(bind(sockfd, (struct sockaddr*)&ser_addr, sizeof(ser_addr)) < 0)
	{
		ROS_INFO("Socket bind error!");
		return -1;
	}

	while (ros::ok())
	{
		if(scan_vec_ready == 0)
		{
			while(1)
			{
				if(j == 12)
				{
					unsigned int len = sizeof(clent_addr);
					recvfrom(sockfd, &MSOP_Data, sizeof(MSOP_Data), 0, (struct sockaddr*)&clent_addr, &len);
					if(MSOP_Data.BlockID[0].Azimuth == 0)
					{
						scan_end = scan_begin;
						scan_begin = ros::Time::now();
					}			
					if((MSOP_Data.BlockID[1].Azimuth - MSOP_Data.BlockID[0].Azimuth) > 0)
					{
						resolution = (MSOP_Data.BlockID[1].Azimuth - MSOP_Data.BlockID[0].Azimuth) / 16;
					}
					j = 0;
				}

				for(;j < 12; j++)
				{
					for(i = 0; i < 16; i++)
					{
						bm_response_scan_t response_ptr;
						response_ptr.angle = (MSOP_Data.BlockID[j].Azimuth + (resolution * i));
						if(MSOP_Data.BlockID[j].DataFlag == 0xEEFF)
						{
							if(response_ptr.angle == 0)
							{
								if(!scan_vec.empty() & (scan_vec_ready == 0))
								{
									scan_vec_ready = 1;
									if(scan_vec.size() < 1200)
									{
										j = 12;
									}
									break;
								}
							}
							response_ptr.dist = MSOP_Data.BlockID[j].Result[i].Dist_1;
							response_ptr.rssi = MSOP_Data.BlockID[j].Result[i].RSSI_1;
							scan_vec.push_back(response_ptr);
						}
					}
					if(scan_vec_ready == 1)
					{
						break;
					}
				}
				if(scan_vec_ready == 1)
				{
					break;
				}
			}
		}

		if(scan_vec_ready == 1)
		{
			sensor_msgs::LaserScan scan;
			uint16_t num_readings;
			float duration = (scan_begin - scan_end).toSec();

			num_readings = scan_vec.size();
			scan.header.stamp = scan_begin;
			scan.header.frame_id = frame_id;
			scan.angle_min = DEG2RAD(-180 + angle_offset);
			scan.angle_max = DEG2RAD(180 + angle_offset);
			scan.angle_increment = 2.0 * M_PI / num_readings;
			scan.scan_time = duration;
			scan.time_increment = duration/(float)num_readings;
			scan.range_min = 0.0;
			scan.range_max = 100.0;
			scan.ranges.resize(num_readings);
			scan.intensities.resize(num_readings);

			for(int i = 0;i < num_readings; i++)
			{
				if (!inverted)
    			{
					scan.ranges[i] = (float)scan_vec[i].dist / 1000;
					scan.intensities[i] = scan_vec[i].rssi;
				}
				else
				{
					scan.ranges[num_readings - i - 1] = (float)scan_vec[i].dist / 1000;
					scan.intensities[num_readings - i - 1] = scan_vec[i].rssi;
				}
			}

			scan_pub.publish(scan);
			ROS_INFO("New topic %s published, total data points: %d", output_topic.c_str(), num_readings);
			scan_vec.clear();
			scan_vec_ready = 0;
		}
	}
	close(sockfd);

	return 0;
}

```

## `drivers/lakibeam_driver/src/src/remote.cpp`

```cpp
#include <stdio.h>
#include <ros/ros.h> 
#include <curl/curl.h>
#include "../thirdparty/rapidjson/document.h"
#include "../thirdparty/rapidjson/prettywriter.h"
#include "../include/remote.h"

using namespace rapidjson;

static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

static size_t dummy_callback(void *buffer, size_t size, size_t nmemb, void *userp)
{
   return size * nmemb;
}

int sensor_config(std::string sensor_ipaddr, std::string parameter, std::string value)
{
	long http_code;
	CURL *curl = curl_easy_init();
	std::string URL_RESTFUL_API = "http://" + sensor_ipaddr + parameter;

	curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3);
	if(curl) {
		curl_easy_setopt(curl, CURLOPT_URL, URL_RESTFUL_API.c_str());
    	curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT");
		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, value.c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, dummy_callback);
		if (curl_easy_perform(curl) == CURLE_OK){
			curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
			if(http_code == 200)
			{
				ROS_INFO("Set %s, Value: %s ... done", URL_RESTFUL_API.c_str(), value.c_str());
			}
			else
			{
				ROS_INFO("Set %s, Value: %s ... failed!", URL_RESTFUL_API.c_str(), value.c_str());
			}
		}
		else
		{
			ROS_INFO("http put error! please check lidar connection!");
		}
	}
	curl_easy_cleanup(curl);
    curl_global_cleanup();

	return 0;
}

int get_telemetry_data(std::string sensor_ipaddr)
{
	CURL *curl;
	CURLcode res;
	std::string readBuffer;
	std::string URL_API_FIRMWARE = "http://" + sensor_ipaddr + "/api/v1/system/firmware";
	std::string URL_API_MONITOR = "http://" + sensor_ipaddr + "/api/v1/system/monitor";
	std::string URL_API_OVERVIEW = "http://" + sensor_ipaddr + "/api/v1/sensor/overview";

	curl = curl_easy_init();
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3);
	if(curl) {
		readBuffer = "";
		curl_easy_setopt(curl, CURLOPT_URL, URL_API_FIRMWARE.c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
		res = curl_easy_perform(curl);
		curl_easy_cleanup(curl);
		const char* json = const_cast<char*>(readBuffer.c_str());
		Document jsondoc;
		jsondoc.Parse(json);
		assert(jsondoc.IsObject());
		ROS_INFO("-------------------------------------------------");
		ROS_INFO("model:		%s", jsondoc["model"].GetString());
		ROS_INFO("sn:		%s", jsondoc["sn"].GetString());
		ROS_INFO("ver hw:		%s", jsondoc["hw"].GetString());
		ROS_INFO("ver fpga:	%s", jsondoc["fpga"].GetString());
		ROS_INFO("ver core:	%s", jsondoc["core"].GetString());
		ROS_INFO("ver aux:	%s", jsondoc["aux"].GetString());
	}

	curl = curl_easy_init();
	if(curl) {
		readBuffer = "";
		curl_easy_setopt(curl, CURLOPT_URL, URL_API_MONITOR.c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
		res = curl_easy_perform(curl);
		curl_easy_cleanup(curl);

		const char* json = const_cast<char*>(readBuffer.c_str());
		Document jsondoc;
		jsondoc.Parse(json);
		assert(jsondoc.IsObject());
		ROS_INFO("load average:	%.2f", jsondoc["load_average"].GetDouble());
		ROS_INFO("men useage:	%.2f", jsondoc["mem_useage"].GetDouble());
		ROS_INFO("uptime:		%.2f sec", jsondoc["uptime"].GetDouble());
	}

	curl = curl_easy_init();
	if(curl) {
		readBuffer = "";
		curl_easy_setopt(curl, CURLOPT_URL, URL_API_OVERVIEW.c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
		res = curl_easy_perform(curl);
		curl_easy_cleanup(curl);

		const char* json = const_cast<char*>(readBuffer.c_str());
		Document jsondoc;
		jsondoc.Parse(json);
		assert(jsondoc.IsObject());
		ROS_INFO("scanfreq:	%d hz", jsondoc["scanfreq"].GetInt());
		ROS_INFO("motor rpm:	%d (%.2fhz)", jsondoc["motor_rpm"].GetInt(), (jsondoc["motor_rpm"].GetInt() / 60.f));
		ROS_INFO("laser enable:	%d", jsondoc["laser_enable"].GetBool());
		ROS_INFO("scan start:	%d deg", jsondoc["scan_range"]["start"].GetInt());
		ROS_INFO("scan stop:	%d deg", jsondoc["scan_range"]["stop"].GetInt());
		ROS_INFO("flt level:	%d", jsondoc["filter"]["level"].GetInt());
		ROS_INFO("flt min_angle:	%d", jsondoc["filter"]["min_angle"].GetInt());
		ROS_INFO("flt max_angle:	%d", jsondoc["filter"]["max_angle"].GetInt());
		ROS_INFO("flt window:	%d", jsondoc["filter"]["window"].GetInt());
		ROS_INFO("flt neighbors:	%d", jsondoc["filter"]["neighbors"].GetInt());
		ROS_INFO("host ip:	%s", jsondoc["host"]["ip"].GetString());
		ROS_INFO("host port:	%d", jsondoc["host"]["port"].GetInt());
		ROS_INFO("-------------------------------------------------");
	}

	return 0;
}
```

## `workspaces/ros2_ws/src/dashgo_driver_ros2/package.xml`

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>dashgo_driver_ros2</name>
  <version>0.1.0</version>
  <description>DashGo 底盘 ROS2 原生串口驱动。</description>

  <maintainer email="gwh@example.com">gwh</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_python</buildtool_depend>

  <depend>ament_index_python</depend>
  <depend>geometry_msgs</depend>
  <depend>launch</depend>
  <depend>launch_ros</depend>
  <depend>nav_msgs</depend>
  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>tf2_ros</depend>

  <exec_depend>python3-serial</exec_depend>

  <test_depend>python3-numpy</test_depend>
  <test_depend>python3-pytest</test_depend>
  <test_depend>python3-yaml</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>

```

## `workspaces/ros2_ws/src/dashgo_driver_ros2/setup.py`

```python
from glob import glob
import os

from setuptools import setup


package_name = "dashgo_driver_ros2"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="gwh",
    maintainer_email="gwh@example.com",
    description="DashGo 底盘 ROS2 原生串口驱动。",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "dashgo_driver_node = dashgo_driver_ros2.driver_node:main",
            "static_tf_node = dashgo_driver_ros2.static_tf_node:main",
        ]
    },
)

```

## `workspaces/ros2_ws/src/dashgo_driver_ros2/setup.cfg`

```ini
[develop]
script_dir=$base/lib/dashgo_driver_ros2
[install]
install_scripts=$base/lib/dashgo_driver_ros2

[tool:pytest]
testpaths = tests

```

## `workspaces/ros2_ws/src/dashgo_driver_ros2/config/dashgo_driver.yaml`

```yaml
dashgo_driver_node:
  ros__parameters:
    serial_port: /dev/dashgo
    baudrate: 115200
    serial_timeout_sec: 0.1
    loop_rate: 50.0
    sensorstate_rate: 10.0
    use_base_controller: true
    base_controller_rate: 10.0
    base_controller_timeout_sec: 1.0
    base_frame: base_link
    odom_frame: odom
    wheel_diameter: 0.1264
    wheel_track: 0.342
    encoder_resolution: 1200
    gear_reduction: 1.0
    Kp: 50.0
    Kd: 20.0
    Ki: 0.0
    Ko: 50.0
    accel_limit: 1.0
    motors_reversed: false
    encoder_min: -32768
    encoder_max: 32768
    cmd_vel_topic: /cmd_vel
    odom_topic: /odom

```

## `workspaces/ros2_ws/src/dashgo_driver_ros2/config/laser_static_tf.yaml`

```yaml
static_tf_node:
  ros__parameters:
    parent_frame: base_link
    child_frame: laser
    translation: [0.0, 0.0, 0.0]
    rotation_rpy: [0.0, 0.0, 0.0]

```

## `workspaces/ros2_ws/src/dashgo_driver_ros2/dashgo_driver_ros2/__init__.py`

```python
"""DashGo ROS2 底盘驱动包。"""

__all__ = [
    "driver_core",
    "driver_node",
    "static_tf_node",
]

```

## `workspaces/ros2_ws/src/dashgo_driver_ros2/dashgo_driver_ros2/driver_core.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi as PI, sin
from typing import Optional, Tuple

ODOM_POSE_COVARIANCE = [
    1e-3, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1e-3, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1e6, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1e6, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1e6, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1e3,
]
ODOM_POSE_COVARIANCE_STOPPED = [
    1e-9, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1e-3, 1e-9, 0.0, 0.0, 0.0,
    0.0, 0.0, 1e6, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1e6, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1e6, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1e-9,
]
ODOM_TWIST_COVARIANCE = [
    1e-3, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1e-3, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1e6, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1e6, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1e6, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1e3,
]
ODOM_TWIST_COVARIANCE_STOPPED = [
    1e-9, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1e-3, 1e-9, 0.0, 0.0, 0.0,
    0.0, 0.0, 1e6, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1e6, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1e6, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1e-9,
]


@dataclass(frozen=True)
class DriverParameters:
    wheel_diameter: float
    wheel_track: float
    encoder_resolution: int
    gear_reduction: float
    accel_limit: float
    base_controller_rate: float
    pid_rate: float = 30.0
    encoder_min: int = -32768
    encoder_max: int = 32768

    @property
    def ticks_per_meter(self) -> float:
        return self.encoder_resolution * self.gear_reduction / (self.wheel_diameter * PI)

    @property
    def max_accel_ticks(self) -> float:
        return self.accel_limit * self.ticks_per_meter / self.base_controller_rate

    @property
    def encoder_span(self) -> int:
        return self.encoder_max - self.encoder_min

    @property
    def encoder_low_wrap(self) -> float:
        return self.encoder_span * 0.3 + self.encoder_min

    @property
    def encoder_high_wrap(self) -> float:
        return self.encoder_span * 0.7 + self.encoder_min


@dataclass
class OdometryMeasurement:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0


class DifferentialDriveOdometry:
    """保留 ROS1 权威驱动的编码器包络与里程计积分逻辑。"""

    def __init__(self, params: DriverParameters) -> None:
        self.params = params
        self.measurement = OdometryMeasurement()
        self.enc_left: Optional[int] = None
        self.enc_right: Optional[int] = None
        self.left_wrap_mult = 0
        self.right_wrap_mult = 0

    def update(self, left_encoder: int, right_encoder: int, dt: float) -> OdometryMeasurement:
        if dt <= 0.0:
            raise ValueError("dt 必须大于 0")

        if self.enc_left is None or self.enc_right is None:
            dleft = 0.0
            dright = 0.0
        else:
            if left_encoder < self.params.encoder_low_wrap and self.enc_left > self.params.encoder_high_wrap:
                self.left_wrap_mult = self.left_wrap_mult + 1
            elif left_encoder > self.params.encoder_high_wrap and self.enc_left < self.params.encoder_low_wrap:
                self.left_wrap_mult = self.left_wrap_mult - 1
            else:
                self.left_wrap_mult = 0

            if right_encoder < self.params.encoder_low_wrap and self.enc_right > self.params.encoder_high_wrap:
                self.right_wrap_mult = self.right_wrap_mult + 1
            elif right_encoder > self.params.encoder_high_wrap and self.enc_right < self.params.encoder_low_wrap:
                self.right_wrap_mult = self.right_wrap_mult - 1
            else:
                self.right_wrap_mult = 0

            dleft = (
                left_encoder + self.left_wrap_mult * self.params.encoder_span - self.enc_left
            ) / self.params.ticks_per_meter
            dright = (
                right_encoder + self.right_wrap_mult * self.params.encoder_span - self.enc_right
            ) / self.params.ticks_per_meter

        self.enc_left = left_encoder
        self.enc_right = right_encoder

        dxy_ave = (dright + dleft) / 2.0
        dtheta = (dright - dleft) / self.params.wheel_track
        linear_velocity = dxy_ave / dt
        angular_velocity = dtheta / dt

        if dxy_ave != 0.0:
            dx = cos(dtheta) * dxy_ave
            dy = -sin(dtheta) * dxy_ave
            self.measurement.x += cos(self.measurement.theta) * dx - sin(self.measurement.theta) * dy
            self.measurement.y += sin(self.measurement.theta) * dx + cos(self.measurement.theta) * dy

        if dtheta != 0.0:
            self.measurement.theta += dtheta

        self.measurement.linear_velocity = linear_velocity
        self.measurement.angular_velocity = angular_velocity
        return self.measurement


def twist_to_target_ticks(
    linear_x: float,
    angular_z: float,
    params: DriverParameters,
) -> Tuple[int, int]:
    """将 Twist 命令转换为左右轮目标 ticks/PID-loop。"""
    if linear_x == 0.0:
        right = angular_z * params.wheel_track * params.gear_reduction / 2.0
        left = -right
    elif angular_z == 0.0:
        left = right = linear_x
    else:
        left = linear_x - angular_z * params.wheel_track * params.gear_reduction / 2.0
        right = linear_x + angular_z * params.wheel_track * params.gear_reduction / 2.0

    left_ticks = int(left * params.ticks_per_meter / params.pid_rate)
    right_ticks = int(right * params.ticks_per_meter / params.pid_rate)
    return left_ticks, right_ticks


def ramp_tick_velocity(current_ticks: float, desired_ticks: float, max_accel_ticks: float) -> float:
    if current_ticks < desired_ticks:
        return min(current_ticks + max_accel_ticks, desired_ticks)
    return max(current_ticks - max_accel_ticks, desired_ticks)


def yaw_to_quaternion(theta: float) -> Tuple[float, float, float, float]:
    return 0.0, 0.0, sin(theta / 2.0), cos(theta / 2.0)

```

## `workspaces/ros2_ws/src/dashgo_driver_ros2/dashgo_driver_ros2/driver_node.py`

```python
from __future__ import annotations

import threading
import time
import traceback
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import Quaternion, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from serial import Serial
from serial.serialutil import SerialException
from std_msgs.msg import Int16
from tf2_ros import TransformBroadcaster

from .driver_core import (
    DriverParameters,
    DifferentialDriveOdometry,
    ODOM_POSE_COVARIANCE,
    ODOM_POSE_COVARIANCE_STOPPED,
    ODOM_TWIST_COVARIANCE,
    ODOM_TWIST_COVARIANCE_STOPPED,
    ramp_tick_velocity,
    twist_to_target_ticks,
    yaw_to_quaternion,
)


class DashgoSerialInterface:
    """严格保留 ROS1 权威驱动的串口命令格式。"""

    ANALOG_PORTS = 6
    PID_RATE = 30

    def __init__(self, port: str, baudrate: int, timeout_sec: float) -> None:
        self.port_name = port
        self.baudrate = int(baudrate)
        self.timeout = float(timeout_sec)
        self.write_timeout = self.timeout
        self.inter_char_timeout = self.timeout / 30.0
        self.port: Optional[Serial] = None
        self.mutex = threading.Lock()

    def connect(self) -> None:
        self.port = Serial(
            port=self.port_name,
            baudrate=self.baudrate,
            timeout=self.timeout,
            write_timeout=self.write_timeout,
        )
        time.sleep(1.0)
        baud = self.get_baud()
        if baud != self.baudrate:
            time.sleep(1.0)
            baud = self.get_baud()
            if baud != self.baudrate:
                raise SerialException(f"串口握手失败: expected={self.baudrate}, got={baud}")

    def close(self) -> None:
        if self.port is not None and self.port.is_open:
            self.port.close()

    def _reset_input(self) -> None:
        if self.port is None:
            return
        try:
            self.port.reset_input_buffer()
        except AttributeError:
            self.port.flushInput()

    def recv(self, timeout_sec: Optional[float] = None) -> Optional[str]:
        if self.port is None:
            raise SerialException("串口未连接")

        timeout = min(timeout_sec if timeout_sec is not None else self.timeout, self.timeout)
        attempts = 0
        value = ""
        while True:
            chunk = self.port.read(1)
            if not chunk:
                attempts += 1
                if attempts * self.inter_char_timeout > timeout:
                    return None
                continue

            char = chunk.decode("utf-8", errors="ignore")
            value += char
            if char == "\r":
                return value.strip("\r")

    def recv_array(self) -> list[int]:
        payload = self.recv(self.timeout * self.ANALOG_PORTS)
        if not payload:
            return []
        try:
            return [int(item) for item in payload.split()]
        except ValueError:
            return []

    def execute(self, command: str) -> int:
        if self.port is None:
            raise SerialException("串口未连接")

        with self.mutex:
            self._reset_input()
            attempts = 0
            value: Optional[str] = None
            while attempts < 2:
                self.port.write((command + "\r").encode("utf-8"))
                value = self.recv(self.timeout)
                if value not in {"", "Invalid Command", None}:
                    break
                attempts += 1
                self._reset_input()

        if value in {None, "", "Invalid Command"}:
            raise SerialException(f"执行命令失败: {command}")
        return int(value)

    def execute_array(self, command: str) -> list[int]:
        if self.port is None:
            raise SerialException("串口未连接")

        with self.mutex:
            self._reset_input()
            attempts = 0
            values: list[int] = []
            while attempts < 2:
                self.port.write((command + "\r").encode("utf-8"))
                values = self.recv_array()
                if values:
                    break
                attempts += 1
                self._reset_input()

        if not values:
            raise SerialException(f"执行数组命令失败: {command}")
        return values

    def execute_ack(self, command: str) -> bool:
        if self.port is None:
            raise SerialException("串口未连接")

        with self.mutex:
            self._reset_input()
            attempts = 0
            ack: Optional[str] = None
            while attempts < 2:
                self.port.write((command + "\r").encode("utf-8"))
                ack = self.recv(self.timeout)
                if ack not in {"", "Invalid Command", None}:
                    break
                attempts += 1
                self._reset_input()
        return ack == "OK"

    def get_baud(self) -> int:
        return self.execute("b")

    def update_pid(self, kp: float, kd: float, ki: float, ko: float) -> bool:
        return self.execute_ack(f"u {kp}:{kd}:{ki}:{ko}")

    def get_encoder_counts(self) -> Tuple[int, int]:
        values = self.execute_array("e")
        if len(values) != 2:
            raise SerialException("编码器返回值不是 2 个")
        return int(values[0]), int(values[1])

    def reset_encoders(self) -> bool:
        return self.execute_ack("r")

    def get_pidin(self) -> Tuple[int, int]:
        values = self.execute_array("i")
        if len(values) != 2:
            raise SerialException("PID 输入返回值不是 2 个")
        return int(values[0]), int(values[1])

    def get_pidout(self) -> Tuple[int, int]:
        values = self.execute_array("f")
        if len(values) != 2:
            raise SerialException("PID 输出返回值不是 2 个")
        return int(values[0]), int(values[1])

    def drive(self, left_ticks: float, right_ticks: float) -> bool:
        # 保留旧驱动的入参顺序与串口报文顺序，不在此处“修正”方向定义。
        return self.execute_ack("m %d %d" % (int(left_ticks), int(right_ticks)))

    def stop(self) -> bool:
        return self.drive(0, 0)


class DashgoDriverNode(Node):
    def __init__(self) -> None:
        super().__init__("dashgo_driver_node")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("serial_port", "/dev/dashgo"),
                ("baudrate", 115200),
                ("serial_timeout_sec", 0.1),
                ("loop_rate", 50.0),
                ("sensorstate_rate", 10.0),
                ("use_base_controller", True),
                ("base_controller_rate", 10.0),
                ("base_controller_timeout_sec", 1.0),
                ("base_frame", "base_link"),
                ("odom_frame", "odom"),
                ("wheel_diameter", 0.1264),
                ("wheel_track", 0.3420),
                ("encoder_resolution", 1200),
                ("gear_reduction", 1.0),
                ("Kp", 50.0),
                ("Kd", 20.0),
                ("Ki", 0.0),
                ("Ko", 50.0),
                ("accel_limit", 1.0),
                ("motors_reversed", False),
                ("encoder_min", -32768),
                ("encoder_max", 32768),
                ("cmd_vel_topic", "/cmd_vel"),
                ("odom_topic", "/odom"),
            ],
        )

        self.serial_port = str(self.get_parameter("serial_port").value)
        self.baudrate = int(self.get_parameter("baudrate").value)
        self.serial_timeout_sec = float(self.get_parameter("serial_timeout_sec").value)
        self.use_base_controller = bool(self.get_parameter("use_base_controller").value)
        self.base_controller_rate = float(self.get_parameter("base_controller_rate").value)
        self.base_controller_timeout_sec = float(self.get_parameter("base_controller_timeout_sec").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.motors_reversed = bool(self.get_parameter("motors_reversed").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)

        self.params = DriverParameters(
            wheel_diameter=float(self.get_parameter("wheel_diameter").value),
            wheel_track=float(self.get_parameter("wheel_track").value),
            encoder_resolution=int(self.get_parameter("encoder_resolution").value),
            gear_reduction=float(self.get_parameter("gear_reduction").value),
            accel_limit=float(self.get_parameter("accel_limit").value),
            base_controller_rate=self.base_controller_rate,
            pid_rate=DashgoSerialInterface.PID_RATE,
            encoder_min=int(self.get_parameter("encoder_min").value),
            encoder_max=int(self.get_parameter("encoder_max").value),
        )

        self.serial = DashgoSerialInterface(self.serial_port, self.baudrate, self.serial_timeout_sec)
        self.serial.connect()
        self.serial.update_pid(
            float(self.get_parameter("Kp").value),
            float(self.get_parameter("Kd").value),
            float(self.get_parameter("Ki").value),
            float(self.get_parameter("Ko").value),
        )
        self.serial.reset_encoders()

        self.odom = DifferentialDriveOdometry(self.params)
        self.current_left_ticks = 0.0
        self.current_right_ticks = 0.0
        self.target_left_ticks = 0.0
        self.target_right_ticks = 0.0
        self.last_cmd_time = self.get_clock().now()
        self.last_poll_time = self.get_clock().now()

        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.left_encoder_pub = self.create_publisher(Int16, "Lencoder", 10)
        self.right_encoder_pub = self.create_publisher(Int16, "Rencoder", 10)
        self.left_pidout_pub = self.create_publisher(Int16, "Lpidout", 10)
        self.right_pidout_pub = self.create_publisher(Int16, "Rpidout", 10)
        self.left_velocity_pub = self.create_publisher(Int16, "Lvel", 10)
        self.right_velocity_pub = self.create_publisher(Int16, "Rvel", 10)
        self.create_subscription(Twist, self.cmd_vel_topic, self.cmd_vel_callback, qos_profile_sensor_data)

        if self.use_base_controller:
            self.create_timer(1.0 / self.base_controller_rate, self.poll_base_controller)

        self.get_logger().info(
            "DashGo ROS2 底盘驱动已启动: "
            f"serial={self.serial_port}, baud={self.baudrate}, wheel_diameter={self.params.wheel_diameter}, "
            f"wheel_track={self.params.wheel_track}, encoder_resolution={self.params.encoder_resolution}"
        )

    def cmd_vel_callback(self, msg: Twist) -> None:
        left_ticks, right_ticks = twist_to_target_ticks(msg.linear.x, msg.angular.z, self.params)
        if self.motors_reversed:
            left_ticks = -left_ticks
            right_ticks = -right_ticks
        self.target_left_ticks = float(left_ticks)
        self.target_right_ticks = float(right_ticks)
        self.last_cmd_time = self.get_clock().now()

    def _publish_odometry(self, now_msg, measurement) -> None:
        qx, qy, qz, qw = yaw_to_quaternion(measurement.theta)

        transform = TransformStamped()
        transform.header.stamp = now_msg
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = measurement.x
        transform.transform.translation.y = measurement.y
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(transform)

        odom_msg = Odometry()
        odom_msg.header.stamp = now_msg
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame
        odom_msg.pose.pose.position.x = measurement.x
        odom_msg.pose.pose.position.y = measurement.y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        odom_msg.twist.twist.linear.x = measurement.linear_velocity
        odom_msg.twist.twist.angular.z = measurement.angular_velocity
        if self.target_left_ticks == 0.0 and self.target_right_ticks == 0.0:
            odom_msg.pose.covariance = ODOM_POSE_COVARIANCE_STOPPED
            odom_msg.twist.covariance = ODOM_TWIST_COVARIANCE_STOPPED
        else:
            odom_msg.pose.covariance = ODOM_POSE_COVARIANCE
            odom_msg.twist.covariance = ODOM_TWIST_COVARIANCE
        self.odom_pub.publish(odom_msg)

    def poll_base_controller(self) -> None:
        now = self.get_clock().now()
        dt = (now - self.last_poll_time).nanoseconds / 1e9
        if dt <= 0.0:
            return
        self.last_poll_time = now

        try:
            left_pidin, right_pidin = self.serial.get_pidin()
            left_pidout, right_pidout = self.serial.get_pidout()
            left_enc, right_enc = self.serial.get_encoder_counts()
        except SerialException as exc:
            self.get_logger().error(f"串口读取失败: {exc}")
            return
        except Exception as exc:  # pragma: no cover - 保护硬件交互
            self.get_logger().error(f"底盘轮询异常: {exc}\n{traceback.format_exc()}")
            return

        self.left_encoder_pub.publish(Int16(data=int(left_pidin)))
        self.right_encoder_pub.publish(Int16(data=int(right_pidin)))
        self.left_pidout_pub.publish(Int16(data=int(left_pidout)))
        self.right_pidout_pub.publish(Int16(data=int(right_pidout)))

        measurement = self.odom.update(left_enc, right_enc, dt)
        self._publish_odometry(now.to_msg(), measurement)

        if (now - self.last_cmd_time).nanoseconds / 1e9 > self.base_controller_timeout_sec:
            self.target_left_ticks = 0.0
            self.target_right_ticks = 0.0

        self.current_left_ticks = ramp_tick_velocity(
            self.current_left_ticks,
            self.target_left_ticks,
            self.params.max_accel_ticks,
        )
        self.current_right_ticks = ramp_tick_velocity(
            self.current_right_ticks,
            self.target_right_ticks,
            self.params.max_accel_ticks,
        )

        self.left_velocity_pub.publish(Int16(data=int(self.current_left_ticks)))
        self.right_velocity_pub.publish(Int16(data=int(self.current_right_ticks)))

        try:
            self.serial.drive(self.current_left_ticks, self.current_right_ticks)
        except SerialException as exc:
            self.get_logger().error(f"串口写入失败: {exc}")

    def stop_robot(self) -> None:
        try:
            self.target_left_ticks = 0.0
            self.target_right_ticks = 0.0
            self.current_left_ticks = 0.0
            self.current_right_ticks = 0.0
            self.serial.stop()
        except Exception:
            pass

    def destroy_node(self):  # type: ignore[override]
        self.stop_robot()
        self.serial.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DashgoDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

```

## `workspaces/ros2_ws/src/dashgo_driver_ros2/dashgo_driver_ros2/static_tf_node.py`

```python
from __future__ import annotations

from typing import Sequence

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster


class ConfigurableStaticTFNode(Node):
    def __init__(self) -> None:
        super().__init__("static_tf_node")
        self.declare_parameters(
            namespace="",
            parameters=[
                ("parent_frame", "base_link"),
                ("child_frame", "laser"),
                ("translation", [0.0, 0.0, 0.0]),
                ("rotation_rpy", [0.0, 0.0, 0.0]),
            ],
        )
        translation = self._as_triplet(self.get_parameter("translation").value)
        roll, pitch, yaw = self._as_triplet(self.get_parameter("rotation_rpy").value)
        parent_frame = str(self.get_parameter("parent_frame").value)
        child_frame = str(self.get_parameter("child_frame").value)

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        transform.transform.translation.x = translation[0]
        transform.transform.translation.y = translation[1]
        transform.transform.translation.z = translation[2]
        qx, qy, qz, qw = self._rpy_to_quaternion(roll, pitch, yaw)
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw

        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcaster.sendTransform(transform)
        self.get_logger().info(
            f"已发布静态 TF: {parent_frame} -> {child_frame}, xyz={translation}, rpy={[roll, pitch, yaw]}"
        )

    @staticmethod
    def _as_triplet(values: Sequence[float]) -> list[float]:
        data = [float(item) for item in values]
        if len(data) != 3:
            raise ValueError("静态 TF 参数必须是长度为 3 的数组")
        return data

    @staticmethod
    def _rpy_to_quaternion(roll: float, pitch: float, yaw: float):
        from math import cos, sin

        cy = cos(yaw * 0.5)
        sy = sin(yaw * 0.5)
        cp = cos(pitch * 0.5)
        sp = sin(pitch * 0.5)
        cr = cos(roll * 0.5)
        sr = sin(roll * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return qx, qy, qz, qw


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConfigurableStaticTFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

```

## `workspaces/ros2_ws/src/dashgo_driver_ros2/launch/base_only.launch.py`

```python
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("dashgo_driver_ros2")
    params_file = LaunchConfiguration("params_file")
    default_params = os.path.join(pkg_share, "config", "dashgo_driver.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            Node(
                package="dashgo_driver_ros2",
                executable="dashgo_driver_node",
                name="dashgo_driver_node",
                output="screen",
                parameters=[params_file],
            ),
        ]
    )

```

## `workspaces/ros2_ws/src/dashgo_driver_ros2/tests/test_driver_core.py`

```python
import numpy as np

from dashgo_driver_ros2.driver_core import (
    DifferentialDriveOdometry,
    DriverParameters,
    ramp_tick_velocity,
    twist_to_target_ticks,
    yaw_to_quaternion,
)


PARAMS = DriverParameters(
    wheel_diameter=0.1264,
    wheel_track=0.342,
    encoder_resolution=1200,
    gear_reduction=1.0,
    accel_limit=1.0,
    base_controller_rate=10.0,
)


def test_twist_to_target_ticks_straight_motion_is_symmetric():
    left, right = twist_to_target_ticks(0.2, 0.0, PARAMS)

    assert left == right
    assert left > 0


def test_twist_to_target_ticks_turn_in_place_is_opposite():
    left, right = twist_to_target_ticks(0.0, 1.0, PARAMS)

    assert left == -right
    assert left < 0 < right


def test_ramp_tick_velocity_limits_step_size():
    assert np.isclose(ramp_tick_velocity(0.0, 50.0, 7.5), 7.5)
    assert np.isclose(ramp_tick_velocity(20.0, 10.0, 7.5), 12.5)


def test_odometry_update_accumulates_forward_motion():
    odom = DifferentialDriveOdometry(PARAMS)
    odom.update(1000, 1000, 0.1)
    measurement = odom.update(1120, 1120, 0.1)

    assert measurement.x > 0.0
    assert np.isclose(measurement.y, 0.0)
    assert np.isclose(measurement.theta, 0.0)
    assert measurement.linear_velocity > 0.0


def test_yaw_to_quaternion_for_zero_yaw():
    qx, qy, qz, qw = yaw_to_quaternion(0.0)

    np.testing.assert_allclose([qx, qy, qz, qw], [0.0, 0.0, 0.0, 1.0])

```

## `workspaces/ros2_ws/src/dashgo_driver_ros2/tests/test_parameter_alignment.py`

```python
from pathlib import Path
from xml.etree import ElementTree

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def test_ros2_driver_defaults_match_ros1_authority_yaml():
    repo_root = _repo_root()
    ros1_yaml = repo_root / "drivers" / "EAI_DRIVER" / "src" / "config" / "my_dashgo_params.yaml"
    ros2_yaml = repo_root / "workspaces" / "ros2_ws" / "src" / "dashgo_driver_ros2" / "config" / "dashgo_driver.yaml"

    ros1_params = yaml.safe_load(ros1_yaml.read_text())
    ros2_params = yaml.safe_load(ros2_yaml.read_text())["dashgo_driver_node"]["ros__parameters"]

    assert ros1_params["baud"] == ros2_params["baudrate"]
    assert ros1_params["timeout"] == ros2_params["serial_timeout_sec"]
    assert ros1_params["wheel_diameter"] == ros2_params["wheel_diameter"]
    assert ros1_params["wheel_track"] == ros2_params["wheel_track"]
    assert ros1_params["encoder_resolution"] == ros2_params["encoder_resolution"]
    assert ros1_params["gear_reduction"] == ros2_params["gear_reduction"]
    assert ros1_params["motors_reversed"] == ros2_params["motors_reversed"]
    assert ros1_params["Kp"] == ros2_params["Kp"]
    assert ros1_params["Kd"] == ros2_params["Kd"]
    assert ros1_params["Ki"] == ros2_params["Ki"]
    assert ros1_params["Ko"] == ros2_params["Ko"]
    assert ros1_params["accel_limit"] == ros2_params["accel_limit"]
    assert ros1_params["sensorstate_rate"] == ros2_params["sensorstate_rate"]
    assert ros1_params["base_controller_rate"] == ros2_params["base_controller_rate"]
    assert ros1_params["base_frame"] == ros2_params["base_frame"]
    assert ros2_params["serial_port"] == "/dev/dashgo"


def test_lidar_defaults_match_selected_single_lidar_baseline():
    repo_root = _repo_root()
    legacy_launch = repo_root / "drivers" / "lakibeam_driver" / "src" / "launch" / "lakibeam1_scan.launch"
    lidar_yaml = repo_root / "workspaces" / "ros2_ws" / "src" / "lakibeam_driver_ros2" / "config" / "lakibeam_driver.yaml"
    static_tf_yaml = repo_root / "workspaces" / "ros2_ws" / "src" / "dashgo_driver_ros2" / "config" / "laser_static_tf.yaml"

    launch_root = ElementTree.fromstring(legacy_launch.read_text())
    legacy_lidar_node = launch_root.find("./node[@pkg='lakibeam1']")
    legacy_tf_node = launch_root.find("./node[@pkg='tf']")
    legacy_params = {
        element.attrib["name"]: element.attrib["value"]
        for element in legacy_lidar_node.findall("./param")
    }
    remap_target = legacy_lidar_node.find("./remap").attrib["to"]

    params = yaml.safe_load(lidar_yaml.read_text())["lakibeam_scan_node"]["ros__parameters"]
    static_tf = yaml.safe_load(static_tf_yaml.read_text())["static_tf_node"]["ros__parameters"]

    assert params["frame_id"] == legacy_params["frame_id"]
    assert params["hostip"] == legacy_params["hostip"]
    assert params["sensorip"] == legacy_params["sensorip"]
    assert params["port"] == int(legacy_params["port"])
    assert params["angle_offset"] == int(legacy_params["angle_offset"])
    assert params["scanfreq"] == int(legacy_params["scanfreq"])
    assert params["filter"] == int(legacy_params["filter"])
    assert params["laser_enable"] is True
    assert params["scan_range_start"] == int(legacy_params["scan_range_start"])
    assert params["scan_range_stop"] == int(legacy_params["scan_range_stop"])
    assert params["output_topic"] == remap_target
    assert static_tf["parent_frame"] == "base_link"
    assert static_tf["child_frame"] == "laser"
    assert static_tf["translation"] == [0.0, 0.0, 0.0]
    assert static_tf["rotation_rpy"] == [0.0, 0.0, 0.0]
    assert "/base_link /laser" in legacy_tf_node.attrib["args"]

```

## `workspaces/ros2_ws/src/lakibeam_driver_ros2/package.xml`

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>lakibeam_driver_ros2</name>
  <version>0.1.0</version>
  <description>Lakibeam 单雷达 ROS2 原生 LaserScan 驱动。</description>

  <maintainer email="gwh@example.com">gwh</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <exec_depend>ament_index_python</exec_depend>
  <depend>dashgo_driver_ros2</depend>
  <depend>launch</depend>
  <depend>launch_ros</depend>
  <depend>rclcpp</depend>
  <depend>sensor_msgs</depend>

  <build_depend>libcurl4-openssl-dev</build_depend>
  <exec_depend>libcurl4-openssl-dev</exec_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>

```

## `workspaces/ros2_ws/src/lakibeam_driver_ros2/CMakeLists.txt`

```text
cmake_minimum_required(VERSION 3.8)
project(lakibeam_driver_ros2)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(CURL REQUIRED)

add_executable(lakibeam_scan_node
  src/lakibeam_scan_node.cpp
  src/remote.cpp
)

target_include_directories(lakibeam_scan_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(lakibeam_scan_node rclcpp sensor_msgs)
target_link_libraries(lakibeam_scan_node CURL::libcurl)

target_compile_features(lakibeam_scan_node PUBLIC cxx_std_17)

install(TARGETS lakibeam_scan_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

```

## `workspaces/ros2_ws/src/lakibeam_driver_ros2/config/lakibeam_driver.yaml`

```yaml
lakibeam_scan_node:
  ros__parameters:
    frame_id: laser
    output_topic: /scan
    inverted: false
    hostip: 0.0.0.0
    sensorip: 192.168.8.2
    port: 2368
    angle_offset: 0
    scanfreq: 30
    filter: 3
    laser_enable: true
    scan_range_start: 90
    scan_range_stop: 270

```

## `workspaces/ros2_ws/src/lakibeam_driver_ros2/include/lakibeam_driver_ros2/data_type.h`

```cpp
#pragma once

#include <cstdint>

#define AUTO_ALIGN __attribute__((packed))

#pragma pack(push, 1)
struct MeasuringResult {
  std::uint16_t dist_1;
  std::uint8_t rssi_1;
  std::uint16_t dist_2;
  std::uint8_t rssi_2;
} AUTO_ALIGN;

struct DataBlock {
  std::uint16_t data_flag;
  std::uint16_t azimuth;
  MeasuringResult result[16];
} AUTO_ALIGN;

struct MsopData {
  DataBlock blocks[12];
  std::uint32_t timestamp;
  std::uint16_t factory;
} AUTO_ALIGN;
#pragma pack(pop)

struct ScanResponse {
  std::uint16_t angle;
  std::uint16_t dist;
  std::uint8_t rssi;
  std::uint32_t timestamp;
};

```

## `workspaces/ros2_ws/src/lakibeam_driver_ros2/include/lakibeam_driver_ros2/remote.h`

```cpp
#pragma once

#include <rclcpp/rclcpp.hpp>

#include <string>

bool sensor_config(
    const std::string &sensor_ipaddr,
    const std::string &parameter,
    const std::string &value,
    const rclcpp::Logger &logger);

bool get_telemetry_data(const std::string &sensor_ipaddr, const rclcpp::Logger &logger);

```

## `workspaces/ros2_ws/src/lakibeam_driver_ros2/src/remote.cpp`

```cpp
#include "lakibeam_driver_ros2/remote.h"

#include <curl/curl.h>

#include <string>

namespace {
size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp) {
  auto *buffer = static_cast<std::string *>(userp);
  buffer->append(static_cast<char *>(contents), size * nmemb);
  return size * nmemb;
}

bool perform_request(
    const std::string &url,
    const std::string &method,
    const std::string &payload,
    std::string *response,
    const rclcpp::Logger &logger) {
  CURL *curl = curl_easy_init();
  if (curl == nullptr) {
    RCLCPP_ERROR(logger, "curl 初始化失败");
    return false;
  }

  long http_code = 0;
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3L);
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, method.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, response);
  if (!payload.empty()) {
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
  }

  const auto result = curl_easy_perform(curl);
  if (result != CURLE_OK) {
    RCLCPP_ERROR(logger, "HTTP 请求失败: %s", curl_easy_strerror(result));
    curl_easy_cleanup(curl);
    return false;
  }

  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  curl_easy_cleanup(curl);
  if (http_code != 200) {
    RCLCPP_WARN(logger, "HTTP 返回码异常: %ld, url=%s", http_code, url.c_str());
    return false;
  }
  return true;
}
}  // namespace

bool sensor_config(
    const std::string &sensor_ipaddr,
    const std::string &parameter,
    const std::string &value,
    const rclcpp::Logger &logger) {
  const std::string url = "http://" + sensor_ipaddr + parameter;
  std::string response;
  const bool ok = perform_request(url, "PUT", value, &response, logger);
  if (ok) {
    RCLCPP_INFO(logger, "已下发雷达参数: %s = %s", url.c_str(), value.c_str());
  }
  return ok;
}

bool get_telemetry_data(const std::string &sensor_ipaddr, const rclcpp::Logger &logger) {
  const std::string firmware_url = "http://" + sensor_ipaddr + "/api/v1/system/firmware";
  const std::string monitor_url = "http://" + sensor_ipaddr + "/api/v1/system/monitor";
  const std::string overview_url = "http://" + sensor_ipaddr + "/api/v1/sensor/overview";

  std::string firmware_response;
  std::string monitor_response;
  std::string overview_response;

  const bool firmware_ok = perform_request(firmware_url, "GET", "", &firmware_response, logger);
  const bool monitor_ok = perform_request(monitor_url, "GET", "", &monitor_response, logger);
  const bool overview_ok = perform_request(overview_url, "GET", "", &overview_response, logger);

  if (firmware_ok) {
    RCLCPP_INFO(logger, "雷达 firmware: %s", firmware_response.c_str());
  }
  if (monitor_ok) {
    RCLCPP_INFO(logger, "雷达 monitor: %s", monitor_response.c_str());
  }
  if (overview_ok) {
    RCLCPP_INFO(logger, "雷达 overview: %s", overview_response.c_str());
  }

  return firmware_ok || monitor_ok || overview_ok;
}

```

## `workspaces/ros2_ws/src/lakibeam_driver_ros2/src/lakibeam_scan_node.cpp`

```cpp
#include "lakibeam_driver_ros2/data_type.h"
#include "lakibeam_driver_ros2/remote.h"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr std::uint16_t kDataFlag = 0xEEFF;
constexpr double kDegToRad = M_PI / 180.0;

std::string bool_to_rest(bool value) { return value ? "true" : "false"; }
}  // namespace

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("lakibeam_scan_node");

  node->declare_parameter<std::string>("frame_id", "laser");
  node->declare_parameter<std::string>("output_topic", "/scan");
  node->declare_parameter<bool>("inverted", false);
  node->declare_parameter<std::string>("hostip", "0.0.0.0");
  node->declare_parameter<std::string>("sensorip", "192.168.8.2");
  node->declare_parameter<int>("port", 2368);
  node->declare_parameter<int>("angle_offset", 0);
  node->declare_parameter<int>("scanfreq", 30);
  node->declare_parameter<int>("filter", 3);
  node->declare_parameter<bool>("laser_enable", true);
  node->declare_parameter<int>("scan_range_start", 90);
  node->declare_parameter<int>("scan_range_stop", 270);

  const auto frame_id = node->get_parameter("frame_id").as_string();
  const auto output_topic = node->get_parameter("output_topic").as_string();
  const auto inverted = node->get_parameter("inverted").as_bool();
  const auto host_ip = node->get_parameter("hostip").as_string();
  const auto sensor_ip = node->get_parameter("sensorip").as_string();
  const auto port = node->get_parameter("port").as_int();
  const auto angle_offset = node->get_parameter("angle_offset").as_int();
  const auto scanfreq = node->get_parameter("scanfreq").as_int();
  const auto filter = node->get_parameter("filter").as_int();
  const auto laser_enable = node->get_parameter("laser_enable").as_bool();
  const auto scan_range_start = node->get_parameter("scan_range_start").as_int();
  const auto scan_range_stop = node->get_parameter("scan_range_stop").as_int();
  const auto logger = node->get_logger();

  auto scan_pub = node->create_publisher<sensor_msgs::msg::LaserScan>(output_topic, rclcpp::SensorDataQoS());

  RCLCPP_INFO(logger, "Lakibeam ROS2 驱动启动: sensor=%s host=%s:%ld topic=%s frame=%s",
              sensor_ip.c_str(), host_ip.c_str(), port, output_topic.c_str(), frame_id.c_str());

  sensor_config(sensor_ip, "/api/v1/sensor/scanfreq", std::to_string(scanfreq), logger);
  sensor_config(sensor_ip, "/api/v1/sensor/laser_enable", bool_to_rest(laser_enable), logger);
  sensor_config(sensor_ip, "/api/v1/sensor/scan_range/start", std::to_string(scan_range_start), logger);
  sensor_config(sensor_ip, "/api/v1/sensor/scan_range/stop", std::to_string(scan_range_stop), logger);
  sensor_config(sensor_ip, "/api/v1/sensor/filter", std::to_string(filter), logger);
  rclcpp::sleep_for(std::chrono::seconds(2));
  get_telemetry_data(sensor_ip, logger);

  const int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd < 0) {
    RCLCPP_ERROR(logger, "创建 UDP socket 失败");
    rclcpp::shutdown();
    return 1;
  }

  int reuse = 1;
  setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  sockaddr_in server_addr {};
  server_addr.sin_family = AF_INET;
  server_addr.sin_addr.s_addr = inet_addr(host_ip.c_str());
  server_addr.sin_port = htons(static_cast<uint16_t>(port));

  if (bind(sockfd, reinterpret_cast<sockaddr *>(&server_addr), sizeof(server_addr)) < 0) {
    RCLCPP_ERROR(logger, "UDP bind 失败: host=%s port=%ld", host_ip.c_str(), port);
    close(sockfd);
    rclcpp::shutdown();
    return 1;
  }

  std::vector<ScanResponse> scan_vec;
  scan_vec.reserve(2048);
  MsopData msop_data {};
  rclcpp::Time scan_begin = node->get_clock()->now();
  rclcpp::Time scan_end = scan_begin;
  bool scan_vec_ready = false;
  int resolution = 25;
  int block_index = 12;
  std::size_t publish_count = 0;

  while (rclcpp::ok()) {
    if (!scan_vec_ready) {
      while (rclcpp::ok()) {
        if (block_index == 12) {
          sockaddr_in client_addr {};
          socklen_t client_len = sizeof(client_addr);
          const auto received = recvfrom(
              sockfd,
              &msop_data,
              sizeof(msop_data),
              0,
              reinterpret_cast<sockaddr *>(&client_addr),
              &client_len);
          if (received <= 0) {
            continue;
          }

          if (msop_data.blocks[0].azimuth == 0) {
            scan_end = scan_begin;
            scan_begin = node->get_clock()->now();
          }
          if (msop_data.blocks[1].azimuth > msop_data.blocks[0].azimuth) {
            resolution = std::max<int>((msop_data.blocks[1].azimuth - msop_data.blocks[0].azimuth) / 16, 1);
          }
          block_index = 0;
        }

        for (; block_index < 12; ++block_index) {
          for (int point_index = 0; point_index < 16; ++point_index) {
            ScanResponse response {};
            response.angle = msop_data.blocks[block_index].azimuth + resolution * point_index;
            if (msop_data.blocks[block_index].data_flag == kDataFlag) {
              if (response.angle == 0 && !scan_vec.empty() && !scan_vec_ready) {
                scan_vec_ready = true;
                if (scan_vec.size() < 1200) {
                  block_index = 12;
                }
                break;
              }
              response.dist = msop_data.blocks[block_index].result[point_index].dist_1;
              response.rssi = msop_data.blocks[block_index].result[point_index].rssi_1;
              scan_vec.push_back(response);
            }
          }
          if (scan_vec_ready) {
            break;
          }
        }
        if (scan_vec_ready) {
          break;
        }
      }
    }

    if (scan_vec_ready && !scan_vec.empty()) {
      sensor_msgs::msg::LaserScan scan_msg;
      const auto num_readings = static_cast<std::size_t>(scan_vec.size());
      double duration = (scan_begin - scan_end).seconds();
      if (duration <= 0.0) {
        duration = 1.0 / std::max<int>(scanfreq, 1);
      }

      scan_msg.header.stamp = scan_begin;
      scan_msg.header.frame_id = frame_id;
      scan_msg.angle_min = (-180.0 + static_cast<double>(angle_offset)) * kDegToRad;
      scan_msg.angle_max = (180.0 + static_cast<double>(angle_offset)) * kDegToRad;
      scan_msg.angle_increment = 2.0 * M_PI / static_cast<double>(num_readings);
      scan_msg.scan_time = duration;
      scan_msg.time_increment = duration / static_cast<double>(num_readings);
      scan_msg.range_min = 0.0;
      scan_msg.range_max = 100.0;
      scan_msg.ranges.resize(num_readings);
      scan_msg.intensities.resize(num_readings);

      for (std::size_t index = 0; index < num_readings; ++index) {
        const auto range_m = static_cast<float>(scan_vec[index].dist) / 1000.0f;
        const auto intensity = static_cast<float>(scan_vec[index].rssi);
        if (!inverted) {
          scan_msg.ranges[index] = range_m;
          scan_msg.intensities[index] = intensity;
        } else {
          const auto mirrored_index = num_readings - index - 1;
          scan_msg.ranges[mirrored_index] = range_m;
          scan_msg.intensities[mirrored_index] = intensity;
        }
      }

      scan_pub->publish(scan_msg);
      ++publish_count;
      if (publish_count == 1 || publish_count % 30 == 0) {
        RCLCPP_INFO(logger, "已发布 /scan: points=%zu, scan_time=%.4f", num_readings, duration);
      }
      scan_vec.clear();
      scan_vec_ready = false;
    }
  }

  close(sockfd);
  rclcpp::shutdown();
  return 0;
}

```

## `workspaces/ros2_ws/src/lakibeam_driver_ros2/launch/lidar_only.launch.py`

```python
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("lakibeam_driver_ros2")
    dashgo_driver_share = get_package_share_directory("dashgo_driver_ros2")
    params_file = LaunchConfiguration("params_file")
    static_tf_params = LaunchConfiguration("static_tf_params")
    default_params = os.path.join(pkg_share, "config", "lakibeam_driver.yaml")
    default_static_tf = os.path.join(dashgo_driver_share, "config", "laser_static_tf.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("static_tf_params", default_value=default_static_tf),
            Node(
                package="lakibeam_driver_ros2",
                executable="lakibeam_scan_node",
                name="lakibeam_scan_node",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="dashgo_driver_ros2",
                executable="static_tf_node",
                name="static_tf_node",
                output="screen",
                parameters=[static_tf_params],
            ),
        ]
    )

```

## `workspaces/ros2_ws/src/dashgo_rl_ros2/package.xml`

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>dashgo_rl_ros2</name>
  <version>0.1.0</version>
  <description>DashGo 深度强化学习项目的 ROS2 部署与 Gazebo Classic 验证包。</description>

  <maintainer email="gwh@example.com">gwh</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_python</buildtool_depend>

  <depend>ament_index_python</depend>
  <exec_depend>dashgo_driver_ros2</exec_depend>
  <depend>geometry_msgs</depend>
  <exec_depend>lakibeam_driver_ros2</exec_depend>
  <depend>launch</depend>
  <depend>launch_ros</depend>
  <depend>nav2_amcl</depend>
  <depend>nav2_lifecycle_manager</depend>
  <depend>nav2_map_server</depend>
  <depend>nav2_msgs</depend>
  <depend>nav2_planner</depend>
  <depend>nav_msgs</depend>
  <depend>python3-numpy</depend>
  <depend>rclpy</depend>
  <depend>robot_state_publisher</depend>
  <depend>sensor_msgs</depend>
  <depend>tf2_geometry_msgs</depend>
  <depend>tf2_ros</depend>
  <depend>xacro</depend>

  <exec_depend>gazebo_ros</exec_depend>
  <exec_depend>joint_state_publisher</exec_depend>
  <exec_depend>rviz2</exec_depend>

  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>

```

## `workspaces/ros2_ws/src/dashgo_rl_ros2/setup.py`

```python
from glob import glob
import os

from setuptools import setup


package_name = "dashgo_rl_ros2"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
        (f"share/{package_name}/maps", glob("maps/*")),
        (f"share/{package_name}/urdf", glob("urdf/*")),
        (f"share/{package_name}/worlds", glob("worlds/*")),
        (f"share/{package_name}/models", glob("models/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="gwh",
    maintainer_email="gwh@example.com",
    description="DashGo 深度强化学习项目的 ROS2 控制与验证包。",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "geo_nav_node = dashgo_rl_ros2.geo_nav_node:main",
            "goal_plan_bridge = dashgo_rl_ros2.goal_plan_bridge:main",
        ],
    },
)

```

## `workspaces/ros2_ws/src/dashgo_rl_ros2/config/dashgo_rl.yaml`

```yaml
geo_nav_node:
  ros__parameters:
    use_sim_time: false
    control_rate: 20.0
    max_lin_vel: 0.3
    max_ang_vel: 1.0
    max_lin_acc: 1.0
    max_ang_acc: 0.6
    max_reverse_speed: 0.15
    max_lidar_range: 12.0
    lidar_dim: 72
    single_obs_dim: 82
    history_len: 3
    waypoint_dist: 1.0
    goal_reached_dist: 0.25
    near_goal_dist: 0.35
    goal_reached_speed: 0.08
    near_goal_speed: 0.05
    goal_obs_max_dist: 8.0
    waypoint_obs_max_dist: 1.0
    heading_guard_enabled: true
    heading_guard_slowdown_angle_deg: 25.0
    heading_guard_turn_in_place_angle_deg: 65.0
    recovery_enabled: true
    recovery_front_blocked_dist: 0.30
    recovery_rear_safe_dist: 0.28
    recovery_stuck_speed: 0.03
    recovery_goal_min_dist: 0.40
    recovery_reverse_speed: 0.08
    recovery_turn_speed: 0.80
    recovery_duration_sec: 0.90
    recovery_cooldown_sec: 1.20
    recovery_side_sector_deg: 70.0
    safety_filter_enabled: true
    goal_topic: /goal_pose
    legacy_goal_topic: /move_base_simple/goal
    plan_topic: /dashgo/global_plan
    cmd_vel_topic: /cmd_vel
    scan_topic: /scan
    odom_topic: /odom
    base_frame: base_link
    model_path: ""

goal_plan_bridge:
  ros__parameters:
    use_sim_time: false
    goal_topic: /goal_pose
    legacy_goal_topic: /move_base_simple/goal
    plan_topic: /dashgo/global_plan
    planner_action_name: /compute_path_to_pose
    planner_id: GridBased
    action_wait_timeout_sec: 2.0

```

## `workspaces/ros2_ws/src/dashgo_rl_ros2/config/nav2_planning.yaml`

```yaml
amcl:
  ros__parameters:
    use_sim_time: false
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: base_link
    global_frame_id: map
    odom_frame_id: odom
    scan_topic: scan
    tf_broadcast: true
    transform_tolerance: 1.0
    min_particles: 200
    max_particles: 2000
    z_hit: 0.5
    z_rand: 0.5
    z_short: 0.05
    sigma_hit: 0.2
    lambda_short: 0.1
    laser_model_type: likelihood_field
    laser_max_range: 12.0
    laser_min_range: 0.0
    max_beams: 60
    update_min_d: 0.2
    update_min_a: 0.2
    set_initial_pose: true
    initial_pose:
      x: 0.0
      y: 0.0
      z: 0.0
      yaw: 0.0

amcl_map_client:
  ros__parameters:
    use_sim_time: false

amcl_rclcpp_node:
  ros__parameters:
    use_sim_time: false

map_server:
  ros__parameters:
    use_sim_time: false
    yaml_filename: ""

planner_server:
  ros__parameters:
    use_sim_time: false
    expected_planner_frequency: 2.0
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_smac_planner/SmacPlanner2D"
      tolerance: 0.25
      downsample_costmap: false
      allow_unknown: true
      max_iterations: 1000000
      max_on_approach_iterations: 1000
      cost_travel_multiplier: 1.0
      use_final_approach_orientation: false

planner_server_rclcpp_node:
  ros__parameters:
    use_sim_time: false

global_costmap:
  global_costmap:
    ros__parameters:
      use_sim_time: false
      global_frame: map
      robot_base_frame: base_link
      update_frequency: 2.0
      publish_frequency: 1.0
      resolution: 0.05
      robot_radius: 0.20
      track_unknown_space: false
      rolling_window: false
      plugins: ["static_layer", "inflation_layer"]
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: true
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.18

global_costmap_client:
  ros__parameters:
    use_sim_time: false

global_costmap_rclcpp_node:
  ros__parameters:
    use_sim_time: false

```

## `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/controller_core.py`

```python
from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np


class ObservationBuffer:
    """维护固定长度的历史观测堆叠。"""

    def __init__(self, history_len: int = 3, obs_dim: int = 82) -> None:
        self.history_len = history_len
        self.obs_dim = obs_dim
        self.buffer = deque(maxlen=history_len)
        self.reset()

    def reset(self) -> None:
        self.buffer.clear()
        for _ in range(self.history_len):
            self.buffer.append(np.zeros(self.obs_dim, dtype=np.float32))

    def update(self, current_obs: np.ndarray) -> None:
        if current_obs.shape[0] != self.obs_dim:
            raise ValueError(f"观测维度错误: 期望 {self.obs_dim}, 实际 {current_obs.shape[0]}")
        self.buffer.append(current_obs.astype(np.float32, copy=False))

    def stacked(self) -> np.ndarray:
        return np.concatenate(list(self.buffer)).astype(np.float32, copy=False)


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """将角度归一化到 [-pi, pi]。"""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def encode_goal_vector(distance: float, angle: float, max_distance: float) -> np.ndarray:
    """将极坐标目标编码为 [dist_norm, sin(theta), cos(theta)]。"""
    clipped_dist = float(np.clip(distance, 0.0, max_distance))
    return np.array(
        [
            clipped_dist / max_distance if max_distance > 0.0 else 0.0,
            np.sin(angle),
            np.cos(angle),
        ],
        dtype=np.float32,
    )


def scale_linear_speed_by_heading(
    linear_cmd: float,
    heading_angle: float,
    slowdown_angle: float = np.deg2rad(25.0),
    turn_in_place_angle: float = np.deg2rad(65.0),
) -> float:
    """根据局部目标夹角压低线速度，避免前进+急转形成绕圈。"""
    abs_angle = abs(float(wrap_angle(float(heading_angle))))
    slowdown_angle = max(float(slowdown_angle), 1.0e-3)
    turn_in_place_angle = max(float(turn_in_place_angle), slowdown_angle + 1.0e-3)

    if abs_angle >= turn_in_place_angle:
        return 0.0
    if abs_angle <= slowdown_angle:
        return float(linear_cmd)

    scale = (turn_in_place_angle - abs_angle) / (turn_in_place_angle - slowdown_angle)
    return float(linear_cmd * np.clip(scale, 0.0, 1.0))


def apply_heading_guard(
    linear_cmd: float,
    angular_cmd: float,
    heading_angle: float,
    max_angular_cmd: float,
    slowdown_angle: float = np.deg2rad(25.0),
    turn_in_place_angle: float = np.deg2rad(65.0),
) -> tuple[float, float]:
    """在大夹角或转向方向错误时接管命令，避免持续绕圈。"""
    wrapped_heading = float(wrap_angle(float(heading_angle)))
    abs_angle = abs(wrapped_heading)
    guarded_linear = scale_linear_speed_by_heading(
        linear_cmd,
        wrapped_heading,
        slowdown_angle=slowdown_angle,
        turn_in_place_angle=turn_in_place_angle,
    )
    heading_turn_cmd = float(np.clip(wrapped_heading, -max_angular_cmd, max_angular_cmd))

    if abs_angle >= float(turn_in_place_angle):
        return 0.0, heading_turn_cmd

    if abs_angle > float(slowdown_angle):
        return guarded_linear, heading_turn_cmd

    return guarded_linear, float(angular_cmd)


def compute_velocity_scaled_lookahead(
    linear_velocity: float,
    forward_min: float = 0.6,
    forward_gain: float = 3.0,
    forward_max: float = 1.2,
    reverse_min: float = 0.45,
    reverse_gain: float = 2.0,
    reverse_max: float = 0.8,
) -> float:
    """根据当前线速度计算训练/部署一致的前瞻距离。"""
    speed = abs(float(linear_velocity))
    if float(linear_velocity) < 0.0:
        reverse_lookahead = max(reverse_min, speed * reverse_gain)
        return float(np.clip(reverse_lookahead, reverse_min, reverse_max))

    forward_lookahead = max(forward_min, speed * forward_gain)
    return float(np.clip(forward_lookahead, forward_min, forward_max))


def process_lidar_ranges(
    ranges: Sequence[float],
    lidar_dim: int = 72,
    max_range: float = 12.0,
    front_index: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """将任意长度的雷达数据压缩为训练期使用的 72 维格式。"""
    raw_ranges = np.asarray(ranges, dtype=np.float32)
    if raw_ranges.size == 0:
        raise ValueError("雷达数据为空，无法生成观测。")

    raw_ranges = np.nan_to_num(raw_ranges, nan=max_range, posinf=max_range, neginf=0.0)
    raw_ranges = np.clip(raw_ranges, 0.0, max_range)
    if front_index is None:
        front_index = raw_ranges.shape[0] // 2
    front_index = int(np.clip(front_index, 0, raw_ranges.shape[0] - 1))
    raw_ranges = np.roll(raw_ranges, -front_index)

    input_len = raw_ranges.shape[0]
    if input_len >= lidar_dim:
        sector_size = input_len // lidar_dim
        truncated_len = lidar_dim * sector_size
        raw_truncated = raw_ranges[:truncated_len]
        processed = raw_truncated.reshape(lidar_dim, sector_size).min(axis=1)
    else:
        target_indices = np.linspace(0, input_len - 1, lidar_dim)
        processed = np.interp(target_indices, np.arange(input_len), raw_ranges)

    if processed.shape[0] < lidar_dim:
        padding = np.full(lidar_dim - processed.shape[0], max_range, dtype=np.float32)
        processed = np.concatenate([processed, padding])

    if normalize:
        processed = processed / max_range
    return processed.astype(np.float32, copy=False)


def select_waypoint_index(distances: Sequence[float], waypoint_dist: float = 1.0) -> int:
    """选择路径上第一个距离超过阈值的点，不足则回退到终点。"""
    if not distances:
        raise ValueError("路径为空，无法选择航点。")

    for index, distance in enumerate(distances):
        if distance >= waypoint_dist:
            return index
    return len(distances) - 1


def select_progressive_waypoint_index(
    path_points_in_base: np.ndarray,
    lookahead_dist: float = 1.0,
    min_forward_x: float = -0.05,
) -> int:
    """先选择当前最近的前向路径点，再沿路径向前取前瞻航点。"""
    path_points = np.asarray(path_points_in_base, dtype=np.float32)
    if path_points.ndim != 2 or path_points.shape[1] != 2 or path_points.shape[0] == 0:
        raise ValueError("路径点格式错误，应为 [N, 2] 且 N > 0。")

    distances = np.linalg.norm(path_points, axis=1)
    forward_indices = np.flatnonzero(path_points[:, 0] >= float(min_forward_x))
    if forward_indices.size > 0:
        nearest_index = int(forward_indices[np.argmin(distances[forward_indices])])
    else:
        nearest_index = int(np.argmin(distances))

    if nearest_index >= path_points.shape[0] - 1:
        return nearest_index

    cumulative = 0.0
    for index in range(nearest_index + 1, path_points.shape[0]):
        cumulative += float(np.linalg.norm(path_points[index] - path_points[index - 1]))
        if cumulative >= float(lookahead_dist):
            return index
    return path_points.shape[0] - 1

```

## `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/safety_filter.py`

```python
"""
ROS2 包内安全过滤器副本。

保持与仓库根目录 `safety_filter.py` 同步，确保安装后的节点可以直接导入。
"""

from __future__ import annotations

import numpy as np


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class DynamicsSafetyFilter:
    """在策略之外追加一层几何安全约束。"""

    def __init__(
        self,
        robot_radius: float = 0.20,
        max_accel: float = 1.0,
        max_ang_accel: float = 0.6,
        safety_margin: float = 0.10,
        front_sector_deg: float = 70.0,
        rear_sector_deg: float = 70.0,
        side_sector_deg: float = 50.0,
    ) -> None:
        self.radius = robot_radius
        self.max_accel = max(max_accel, 1.0e-3)
        self.max_ang_accel = max(max_ang_accel, 1.0e-3)
        self.margin = safety_margin
        self.front_sector = np.deg2rad(front_sector_deg / 2.0)
        self.rear_sector = np.deg2rad(rear_sector_deg / 2.0)
        self.side_sector = np.deg2rad(side_sector_deg / 2.0)

    def _min_distance_in_sector(
        self,
        scan_ranges: np.ndarray,
        angles: np.ndarray,
        center_angle: float,
        half_width: float,
        max_range: float,
    ) -> float:
        wrapped = np.abs(_wrap_angle(angles - center_angle))
        mask = wrapped <= half_width
        if not np.any(mask):
            return max_range

        sector = scan_ranges[mask]
        valid = sector[(sector > 0.05) & (sector < max_range)]
        if valid.size == 0:
            return max_range
        return float(np.min(valid))

    def _limit_linear_speed(self, cmd_v: float, clearance: float) -> float:
        braking_distance = (cmd_v**2) / (2.0 * self.max_accel)
        required_distance = braking_distance + self.radius + self.margin
        if clearance >= required_distance:
            return cmd_v

        available = max(clearance - self.radius - self.margin, 0.0)
        safe_speed = np.sqrt(max(0.0, 2.0 * self.max_accel * available))
        return float(np.sign(cmd_v) * min(abs(cmd_v), safe_speed))

    def _limit_angular_speed(self, cmd_w: float, left_clearance: float, right_clearance: float) -> float:
        side_clearance = min(left_clearance, right_clearance)
        safe_clearance = self.radius + self.margin
        if side_clearance >= safe_clearance:
            return cmd_w

        scale = max(0.0, side_clearance / safe_clearance)
        return float(cmd_w * scale)

    def filter(
        self,
        cmd_v: float,
        cmd_w: float,
        scan_ranges: np.ndarray,
        angle_min: float = -np.pi,
        angle_increment: float | None = None,
        max_range: float = 12.0,
    ) -> tuple[float, float]:
        scan = np.asarray(scan_ranges, dtype=np.float32)
        if scan.size == 0:
            return cmd_v, cmd_w

        scan = np.nan_to_num(scan, nan=max_range, posinf=max_range, neginf=0.0)
        scan = np.clip(scan, 0.0, max_range)

        if angle_increment is None:
            angle_increment = (2.0 * np.pi) / max(scan.size, 1)
        angles = angle_min + np.arange(scan.size, dtype=np.float32) * angle_increment

        front_clearance = self._min_distance_in_sector(scan, angles, 0.0, self.front_sector, max_range)
        rear_clearance = self._min_distance_in_sector(scan, angles, np.pi, self.rear_sector, max_range)
        left_clearance = self._min_distance_in_sector(scan, angles, np.pi / 2.0, self.side_sector, max_range)
        right_clearance = self._min_distance_in_sector(scan, angles, -np.pi / 2.0, self.side_sector, max_range)

        if cmd_v > 0.0:
            cmd_v = self._limit_linear_speed(cmd_v, front_clearance)
        elif cmd_v < 0.0:
            cmd_v = self._limit_linear_speed(cmd_v, rear_clearance)

        if abs(cmd_v) < 0.05:
            cmd_w = self._limit_angular_speed(cmd_w, left_clearance, right_clearance)

        return float(cmd_v), float(cmd_w)

```

## `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/goal_plan_bridge.py`

```python
from __future__ import annotations

from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile


class GoalPlanBridge(Node):
    """将 RViz 目标点转换为 Nav2 全局路径，供 RL 局部规划器消费。"""

    def __init__(self) -> None:
        super().__init__("goal_plan_bridge")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("goal_topic", "/goal_pose"),
                ("legacy_goal_topic", "/move_base_simple/goal"),
                ("plan_topic", "/dashgo/global_plan"),
                ("planner_action_name", "/compute_path_to_pose"),
                ("planner_id", "GridBased"),
                ("action_wait_timeout_sec", 2.0),
            ],
        )

        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.legacy_goal_topic = str(self.get_parameter("legacy_goal_topic").value)
        self.plan_topic = str(self.get_parameter("plan_topic").value)
        self.planner_action_name = str(self.get_parameter("planner_action_name").value)
        self.planner_id = str(self.get_parameter("planner_id").value)
        self.action_wait_timeout_sec = float(self.get_parameter("action_wait_timeout_sec").value)

        plan_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.plan_pub = self.create_publisher(Path, self.plan_topic, plan_qos)
        self.plan_client = ActionClient(self, ComputePathToPose, self.planner_action_name)

        self.create_subscription(PoseStamped, self.goal_topic, self.goal_cb, 10)
        self.create_subscription(PoseStamped, self.legacy_goal_topic, self.goal_cb, 10)

        self._request_serial = 0
        self._active_goal: Optional[PoseStamped] = None

        self.get_logger().info(
            f"目标桥接节点已启动: goal={self.goal_topic}, legacy_goal={self.legacy_goal_topic}, "
            f"plan={self.plan_topic}, action={self.planner_action_name}"
        )

    def goal_cb(self, msg: PoseStamped) -> None:
        self._active_goal = msg
        self._request_serial += 1
        request_id = self._request_serial

        if not self.plan_client.wait_for_server(timeout_sec=self.action_wait_timeout_sec):
            self.get_logger().warn("ComputePathToPose action server 未就绪，跳过本次目标规划。")
            return

        goal_request = ComputePathToPose.Goal()
        goal_request.goal = msg
        goal_request.planner_id = self.planner_id
        goal_request.use_start = False

        future = self.plan_client.send_goal_async(goal_request)
        future.add_done_callback(lambda done, rid=request_id: self.goal_response_cb(done, rid))

        self.get_logger().info(
            f"收到目标，开始请求全局路径: frame={msg.header.frame_id}, "
            f"xy=({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
        )

    def goal_response_cb(self, future, request_id: int) -> None:
        if request_id != self._request_serial:
            return

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("ComputePathToPose 请求被拒绝。")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda done, rid=request_id: self.result_cb(done, rid))

    def result_cb(self, future, request_id: int) -> None:
        if request_id != self._request_serial:
            return

        result = future.result()
        if result is None:
            self.get_logger().warn("未收到全局路径规划结果。")
            return

        path = result.result.path
        if not path.poses:
            self.get_logger().warn("全局路径为空，未发布到 RL 控制链。")
            return

        self.plan_pub.publish(path)
        self.get_logger().info(
            f"已发布全局路径，共 {len(path.poses)} 个路径点，frame={path.header.frame_id}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GoalPlanBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

```

## `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py`

```python
from __future__ import annotations

import os
import traceback
from typing import Optional

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener

from .controller_core import (
    apply_heading_guard,
    ObservationBuffer,
    compute_velocity_scaled_lookahead,
    encode_goal_vector,
    process_lidar_ranges,
    select_progressive_waypoint_index,
    select_waypoint_index,
)
from .safety_filter import DynamicsSafetyFilter

try:
    import torch
except ImportError:  # pragma: no cover - 运行期依赖，单元测试可跳过
    torch = None


class GeoNavNode(Node):
    """DashGo TorchScript 模型控制节点。"""

    def __init__(self) -> None:
        super().__init__("geo_nav_node")

        package_share = get_package_share_directory("dashgo_rl_ros2")
        default_model_path = os.path.join(package_share, "models", "policy_torchscript.pt")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("model_path", default_model_path),
                ("control_rate", 20.0),
                ("max_lin_vel", 0.3),
                ("max_ang_vel", 1.0),
                ("max_lin_acc", 1.0),
                ("max_ang_acc", 0.6),
                ("max_reverse_speed", 0.15),
                ("max_lidar_range", 12.0),
                ("lidar_dim", 72),
                ("single_obs_dim", 82),
                ("history_len", 3),
                ("waypoint_dist", 1.0),
                ("forward_lookahead_min", 0.6),
                ("forward_lookahead_gain", 3.0),
                ("forward_lookahead_max", 1.2),
                ("reverse_lookahead_min", 0.45),
                ("reverse_lookahead_gain", 2.0),
                ("reverse_lookahead_max", 0.8),
                ("goal_reached_dist", 0.25),
                ("near_goal_dist", 0.35),
                ("goal_reached_speed", 0.08),
                ("near_goal_speed", 0.05),
                ("goal_obs_max_dist", 8.0),
                ("waypoint_obs_max_dist", 1.0),
                ("heading_guard_enabled", True),
                ("heading_guard_slowdown_angle_deg", 25.0),
                ("heading_guard_turn_in_place_angle_deg", 65.0),
                ("recovery_enabled", True),
                ("recovery_front_blocked_dist", 0.30),
                ("recovery_rear_safe_dist", 0.28),
                ("recovery_stuck_speed", 0.03),
                ("recovery_goal_min_dist", 0.40),
                ("recovery_reverse_speed", 0.08),
                ("recovery_turn_speed", 0.80),
                ("recovery_duration_sec", 0.90),
                ("recovery_cooldown_sec", 1.20),
                ("recovery_side_sector_deg", 70.0),
                ("safety_filter_enabled", True),
                ("goal_topic", "/goal_pose"),
                ("legacy_goal_topic", "/move_base_simple/goal"),
                ("plan_topic", "/dashgo/global_plan"),
                ("cmd_vel_topic", "/cmd_vel"),
                ("scan_topic", "/scan"),
                ("odom_topic", "/odom"),
                ("base_frame", "base_link"),
            ],
        )

        self.model_path = str(self.get_parameter("model_path").value)
        self.control_rate = float(self.get_parameter("control_rate").value)
        self.dt = 1.0 / self.control_rate
        self.max_v = float(self.get_parameter("max_lin_vel").value)
        self.max_w = float(self.get_parameter("max_ang_vel").value)
        self.max_acc_lin = float(self.get_parameter("max_lin_acc").value)
        self.max_acc_ang = float(self.get_parameter("max_ang_acc").value)
        self.max_reverse_speed = float(self.get_parameter("max_reverse_speed").value)
        self.max_lidar_range = float(self.get_parameter("max_lidar_range").value)
        self.lidar_dim = int(self.get_parameter("lidar_dim").value)
        self.single_obs_dim = int(self.get_parameter("single_obs_dim").value)
        self.history_len = int(self.get_parameter("history_len").value)
        self.total_input_dim = self.single_obs_dim * self.history_len
        self.waypoint_dist = float(self.get_parameter("waypoint_dist").value)
        self.forward_lookahead_min = float(self.get_parameter("forward_lookahead_min").value)
        self.forward_lookahead_gain = float(self.get_parameter("forward_lookahead_gain").value)
        self.forward_lookahead_max = float(self.get_parameter("forward_lookahead_max").value)
        self.reverse_lookahead_min = float(self.get_parameter("reverse_lookahead_min").value)
        self.reverse_lookahead_gain = float(self.get_parameter("reverse_lookahead_gain").value)
        self.reverse_lookahead_max = float(self.get_parameter("reverse_lookahead_max").value)
        self.goal_reached_dist = float(self.get_parameter("goal_reached_dist").value)
        self.near_goal_dist = float(self.get_parameter("near_goal_dist").value)
        self.goal_reached_speed = float(self.get_parameter("goal_reached_speed").value)
        self.near_goal_speed = float(self.get_parameter("near_goal_speed").value)
        self.goal_obs_max_dist = float(self.get_parameter("goal_obs_max_dist").value)
        self.waypoint_obs_max_dist = float(self.get_parameter("waypoint_obs_max_dist").value)
        self.heading_guard_enabled = bool(self.get_parameter("heading_guard_enabled").value)
        self.heading_guard_slowdown_angle = np.deg2rad(
            float(self.get_parameter("heading_guard_slowdown_angle_deg").value)
        )
        self.heading_guard_turn_in_place_angle = np.deg2rad(
            float(self.get_parameter("heading_guard_turn_in_place_angle_deg").value)
        )
        self.recovery_enabled = bool(self.get_parameter("recovery_enabled").value)
        self.recovery_front_blocked_dist = float(self.get_parameter("recovery_front_blocked_dist").value)
        self.recovery_rear_safe_dist = float(self.get_parameter("recovery_rear_safe_dist").value)
        self.recovery_stuck_speed = float(self.get_parameter("recovery_stuck_speed").value)
        self.recovery_goal_min_dist = float(self.get_parameter("recovery_goal_min_dist").value)
        self.recovery_reverse_speed = float(self.get_parameter("recovery_reverse_speed").value)
        self.recovery_turn_speed = float(self.get_parameter("recovery_turn_speed").value)
        self.recovery_duration_sec = float(self.get_parameter("recovery_duration_sec").value)
        self.recovery_cooldown_sec = float(self.get_parameter("recovery_cooldown_sec").value)
        self.recovery_side_sector = np.deg2rad(
            float(self.get_parameter("recovery_side_sector_deg").value) / 2.0
        )
        self.safety_filter_enabled = bool(self.get_parameter("safety_filter_enabled").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.legacy_goal_topic = str(self.get_parameter("legacy_goal_topic").value)
        self.plan_topic = str(self.get_parameter("plan_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.base_frame = str(self.get_parameter("base_frame").value)

        self.obs_buffer = ObservationBuffer(self.history_len, self.single_obs_dim)
        self.last_action = np.zeros(2, dtype=np.float32)
        self.current_vel = np.zeros(6, dtype=np.float32)
        self.goal_vector = np.zeros(3, dtype=np.float32)
        self.waypoint_vector = np.zeros(3, dtype=np.float32)
        self.goal_heading = 0.0
        self.waypoint_heading = 0.0
        self.goal_distance = np.inf
        self.waypoint_distance = np.inf
        self.latest_scan: Optional[LaserScan] = None
        self.goal_pose: Optional[PoseStamped] = None
        self.latest_plan: Optional[Path] = None
        self.current_waypoint_index = -1
        self.recovery_active_until = 0.0
        self.recovery_cooldown_until = 0.0
        self.recovery_turn_dir = 1.0
        self._throttle_state = {}
        self.safety_filter = (
            DynamicsSafetyFilter(robot_radius=0.20, max_accel=self.max_acc_lin, max_ang_accel=self.max_acc_ang)
            if self.safety_filter_enabled
            else None
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, self.goal_topic, self.goal_cb, 10)
        self.create_subscription(PoseStamped, self.legacy_goal_topic, self.goal_cb, 10)
        self.create_subscription(Path, self.plan_topic, self.plan_cb, 10)

        self.device = None
        self.model = None
        self.load_model()

        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"GeoNav ROS2 节点已启动: model={self.model_path}, input_dim={self.total_input_dim}, "
            f"cmd_vel={self.cmd_vel_topic}, plan={self.plan_topic}"
        )

    def load_model(self) -> None:
        if torch is None:
            raise RuntimeError(
                "未检测到 torch。请使用 `/usr/bin/python3.10` 运行，并为该解释器安装 torch。"
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(self.model_path, map_location=self.device)
        self.model.eval()

        dummy_input = torch.randn(1, self.total_input_dim, device=self.device)
        try:
            model_output = self.model(dummy_input)
        except Exception:
            model_output = self.model({"policy": dummy_input})
        output_shape = getattr(model_output, "shape", None)
        self.get_logger().info(f"模型加载成功: device={self.device}, output_shape={output_shape}")

    def throttle_log(self, key: str, level: str, message: str, interval_sec: float = 2.0) -> None:
        now_sec = self.get_clock().now().nanoseconds / 1e9
        last_sec = self._throttle_state.get(key, 0.0)
        if now_sec - last_sec < interval_sec:
            return

        self._throttle_state[key] = now_sec
        logger = self.get_logger()
        normalized_level = level.lower()
        if normalized_level in {"warn", "warning"}:
            logger.warning(message)
        elif normalized_level == "error":
            logger.error(message)
        elif normalized_level == "debug":
            logger.debug(message)
        else:
            logger.info(message)

    def scan_cb(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def odom_cb(self, msg: Odometry) -> None:
        self.current_vel[0] = msg.twist.twist.linear.x
        self.current_vel[1] = msg.twist.twist.linear.y
        self.current_vel[2] = msg.twist.twist.linear.z
        self.current_vel[3] = msg.twist.twist.angular.x
        self.current_vel[4] = msg.twist.twist.angular.y
        self.current_vel[5] = msg.twist.twist.angular.z

    def goal_cb(self, msg: PoseStamped) -> None:
        self.goal_pose = msg
        self.obs_buffer.reset()
        self.last_action[:] = 0.0
        self.current_waypoint_index = -1
        self.get_logger().info(
            f"收到目标点: frame={msg.header.frame_id}, xy=({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
        )

    def plan_cb(self, msg: Path) -> None:
        self.latest_plan = msg if msg.poses else None
        if self.latest_plan is None:
            self.throttle_log("empty_plan", "warn", "收到空全局路径，将回退到目标点跟踪。", 5.0)
            return

        self.throttle_log(
            "plan_update",
            "info",
            f"收到全局路径: frame={msg.header.frame_id}, poses={len(msg.poses)}",
            2.0,
        )

    def clear_goal_state(self) -> None:
        self.goal_pose = None
        self.latest_plan = None
        self.current_waypoint_index = -1
        self.last_action[:] = 0.0
        self.goal_vector[:] = 0.0
        self.waypoint_vector[:] = 0.0
        self.goal_distance = np.inf
        self.waypoint_distance = np.inf
        self.goal_heading = 0.0
        self.waypoint_heading = 0.0
        self.recovery_active_until = 0.0
        self.recovery_cooldown_until = 0.0
        self.obs_buffer.reset()

    def transform_pose_to_base(self, pose: PoseStamped) -> Optional[PoseStamped]:
        frame_id = pose.header.frame_id or "map"
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                frame_id,
                Time(),
                timeout=Duration(seconds=0.1),
            )
            return do_transform_pose_stamped(pose, transform)
        except TransformException as exc:
            self.throttle_log("tf_transform", "warn", f"TF 变换失败: {exc}")
            return None

    def select_target_from_plan(self) -> Optional[PoseStamped]:
        if self.latest_plan is None or not self.latest_plan.poses:
            return None

        plan_frame = self.latest_plan.header.frame_id or "map"
        normalized_poses = []
        path_points_in_base = []

        for pose in self.latest_plan.poses:
            candidate = PoseStamped()
            candidate.header = pose.header
            if not candidate.header.frame_id:
                candidate.header.frame_id = plan_frame
            candidate.pose = pose.pose
            normalized_poses.append(candidate)

            pose_in_base = self.transform_pose_to_base(candidate)
            if pose_in_base is None:
                return None

            path_points_in_base.append(
                [
                    float(pose_in_base.pose.position.x),
                    float(pose_in_base.pose.position.y),
                ]
            )

        lookahead_distance = self.compute_waypoint_lookahead()
        self.current_waypoint_index = select_progressive_waypoint_index(
            np.asarray(path_points_in_base, dtype=np.float32),
            lookahead_dist=lookahead_distance,
        )
        return normalized_poses[self.current_waypoint_index]

    def resolve_target_pose(self) -> Optional[PoseStamped]:
        return self.select_target_from_plan() or self.goal_pose

    def scale_linear_action(self, action_v: float) -> float:
        return float(action_v * self.max_v if action_v >= 0.0 else action_v * self.max_reverse_speed)

    def compute_front_index(self, scan: LaserScan) -> int:
        num_points = len(scan.ranges)
        if num_points == 0 or abs(scan.angle_increment) < 1.0e-6:
            return num_points // 2
        raw_index = int(round((0.0 - scan.angle_min) / scan.angle_increment))
        return raw_index % num_points

    def compute_waypoint_lookahead(self) -> float:
        lookahead_distance = compute_velocity_scaled_lookahead(
            self.current_vel[0],
            forward_min=self.forward_lookahead_min,
            forward_gain=self.forward_lookahead_gain,
            forward_max=self.forward_lookahead_max,
            reverse_min=self.reverse_lookahead_min,
            reverse_gain=self.reverse_lookahead_gain,
            reverse_max=self.reverse_lookahead_max,
        )
        return float(max(lookahead_distance, 0.0))

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def update_last_action_from_cmd(self, cmd_v: float, cmd_w: float) -> None:
        if cmd_v >= 0.0:
            norm_v = cmd_v / max(self.max_v, 1.0e-6)
        else:
            norm_v = cmd_v / max(self.max_reverse_speed, 1.0e-6)
        norm_w = cmd_w / max(self.max_w, 1.0e-6)
        self.last_action = np.array(
            [
                float(np.clip(norm_v, -1.0, 1.0)),
                float(np.clip(norm_w, -1.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def compute_recovery_clearances(self) -> tuple[float, float, float, float]:
        if self.latest_scan is None:
            inf = float(self.max_lidar_range)
            return inf, inf, inf, inf

        scan = np.asarray(self.latest_scan.ranges, dtype=np.float32)
        scan = np.nan_to_num(scan, nan=self.max_lidar_range, posinf=self.max_lidar_range, neginf=0.0)
        scan = np.clip(scan, 0.0, self.max_lidar_range)
        if scan.size == 0:
            inf = float(self.max_lidar_range)
            return inf, inf, inf, inf

        angles = float(self.latest_scan.angle_min) + np.arange(scan.size, dtype=np.float32) * float(
            self.latest_scan.angle_increment
        )

        def min_clearance(center_angle: float, half_width: float) -> float:
            wrapped = np.abs((angles - center_angle + np.pi) % (2.0 * np.pi) - np.pi)
            mask = wrapped <= half_width
            if not np.any(mask):
                return float(self.max_lidar_range)
            sector = scan[mask]
            valid = sector[(sector > 0.05) & (sector < self.max_lidar_range)]
            if valid.size == 0:
                return float(self.max_lidar_range)
            return float(np.min(valid))

        front = min_clearance(0.0, self.recovery_side_sector)
        rear = min_clearance(np.pi, self.recovery_side_sector)
        left = min_clearance(np.pi / 2.0, self.recovery_side_sector)
        right = min_clearance(-np.pi / 2.0, self.recovery_side_sector)
        return front, rear, left, right

    def maybe_compute_recovery_command(self) -> Optional[tuple[float, float]]:
        if not self.recovery_enabled or self.goal_pose is None:
            return None

        now_sec = self.now_sec()
        front_clearance, rear_clearance, left_clearance, right_clearance = self.compute_recovery_clearances()

        if now_sec < self.recovery_active_until:
            reverse_cmd = -self.recovery_reverse_speed if rear_clearance >= self.recovery_rear_safe_dist else 0.0
            return reverse_cmd, self.recovery_turn_dir * self.recovery_turn_speed

        if now_sec < self.recovery_cooldown_until:
            return None

        front_blocked = front_clearance < self.recovery_front_blocked_dist
        stuck = abs(float(self.current_vel[0])) < self.recovery_stuck_speed
        far_enough = self.goal_distance > self.recovery_goal_min_dist
        if not (front_blocked and stuck and far_enough):
            return None

        self.recovery_turn_dir = 1.0 if left_clearance >= right_clearance else -1.0
        self.recovery_active_until = now_sec + self.recovery_duration_sec
        self.recovery_cooldown_until = self.recovery_active_until + self.recovery_cooldown_sec
        self.throttle_log(
            "recovery_trigger",
            "warn",
            "触发倒车脱困: "
            f"front={front_clearance:.2f}, rear={rear_clearance:.2f}, "
            f"left={left_clearance:.2f}, right={right_clearance:.2f}, "
            f"turn_dir={'left' if self.recovery_turn_dir > 0 else 'right'}",
            0.5,
        )
        reverse_cmd = -self.recovery_reverse_speed if rear_clearance >= self.recovery_rear_safe_dist else 0.0
        return reverse_cmd, self.recovery_turn_dir * self.recovery_turn_speed

    def update_target_vectors(self) -> bool:
        if self.goal_pose is None:
            return False

        goal_in_base = self.transform_pose_to_base(self.goal_pose)
        if goal_in_base is None:
            return False
        goal_dx = goal_in_base.pose.position.x
        goal_dy = goal_in_base.pose.position.y
        self.goal_distance = float(np.hypot(goal_dx, goal_dy))
        goal_angle = float(np.arctan2(goal_dy, goal_dx))
        self.goal_heading = goal_angle
        self.goal_vector = encode_goal_vector(self.goal_distance, goal_angle, self.goal_obs_max_dist)

        target_pose = self.resolve_target_pose()
        if target_pose is None:
            target_pose = self.goal_pose
        target_in_base = self.transform_pose_to_base(target_pose)
        if target_in_base is None:
            return False

        dx = target_in_base.pose.position.x
        dy = target_in_base.pose.position.y
        self.waypoint_distance = float(np.hypot(dx, dy))
        waypoint_angle = float(np.arctan2(dy, dx))
        self.waypoint_heading = waypoint_angle
        self.waypoint_vector = encode_goal_vector(
            self.waypoint_distance,
            waypoint_angle,
            self.waypoint_obs_max_dist,
        )
        return True

    def should_stop(self) -> bool:
        if self.goal_pose is None:
            return False

        goal_in_base = self.transform_pose_to_base(self.goal_pose)
        if goal_in_base is None:
            return False

        dist = float(np.hypot(goal_in_base.pose.position.x, goal_in_base.pose.position.y))
        speed = float(abs(self.current_vel[0]))
        yaw_rate = float(abs(self.current_vel[5]))

        if dist < self.goal_reached_dist and speed < self.goal_reached_speed and yaw_rate < 0.2:
            return True
        if dist < self.near_goal_dist and speed < self.near_goal_speed and yaw_rate < 0.15:
            return True
        return False

    def control_loop(self) -> None:
        if self.latest_scan is None or self.model is None:
            return

        if not self.update_target_vectors():
            return

        if self.should_stop():
            self.cmd_pub.publish(Twist())
            self.get_logger().info("已接近终点，发送停车指令并清理目标状态。")
            self.clear_goal_state()
            return

        lidar_data = process_lidar_ranges(
            self.latest_scan.ranges,
            lidar_dim=self.lidar_dim,
            max_range=self.max_lidar_range,
            front_index=self.compute_front_index(self.latest_scan),
            normalize=True,
        )

        current_obs_vec = np.concatenate(
            [
                lidar_data,
                self.waypoint_vector,
                self.goal_vector,
                np.array([self.current_vel[0]], dtype=np.float32),
                np.array([self.current_vel[5]], dtype=np.float32),
                self.last_action,
            ]
        ).astype(np.float32)

        self.obs_buffer.update(current_obs_vec)

        recovery_cmd = self.maybe_compute_recovery_command()
        if recovery_cmd is not None:
            cmd_v, cmd_w = recovery_cmd
            if self.safety_filter is not None:
                try:
                    cmd_v, cmd_w = self.safety_filter.filter(
                        cmd_v,
                        cmd_w,
                        np.asarray(self.latest_scan.ranges, dtype=np.float32),
                        angle_min=float(self.latest_scan.angle_min),
                        angle_increment=float(self.latest_scan.angle_increment),
                        max_range=self.max_lidar_range,
                    )
                except Exception as exc:
                    self.throttle_log("safety_filter", "warn", f"安全过滤失败，回退到未过滤命令: {exc}", 2.0)

            twist = Twist()
            twist.linear.x = cmd_v
            twist.angular.z = cmd_w
            self.cmd_pub.publish(twist)
            self.update_last_action_from_cmd(cmd_v, cmd_w)
            return

        stacked_obs = self.obs_buffer.stacked()
        input_tensor = torch.from_numpy(stacked_obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            try:
                model_output = self.model(input_tensor)
            except Exception:
                model_output = self.model({"policy": input_tensor})

        if isinstance(model_output, dict):
            model_output = next(iter(model_output.values()))
        raw_action = model_output.detach().cpu().numpy()[0].astype(np.float32)
        raw_action = np.nan_to_num(raw_action, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(np.abs(raw_action)) > 1.5:
            self.throttle_log(
                "action_saturation",
                "warn",
                f"模型输出超出训练动作范围，已裁剪到[-1,1]: raw={raw_action}",
                2.0,
            )
        action = np.clip(raw_action, -1.0, 1.0)

        cmd_v = self.scale_linear_action(float(action[0]))
        cmd_w = float(action[1]) * self.max_w

        last_cmd_v = self.scale_linear_action(float(self.last_action[0]))
        last_cmd_w = float(self.last_action[1]) * self.max_w
        acc_lin_per_tick = self.max_acc_lin * self.dt
        acc_ang_per_tick = self.max_acc_ang * self.dt

        cmd_v = float(np.clip(cmd_v, last_cmd_v - acc_lin_per_tick, last_cmd_v + acc_lin_per_tick))
        cmd_w = float(np.clip(cmd_w, last_cmd_w - acc_ang_per_tick, last_cmd_w + acc_ang_per_tick))
        cmd_v = float(np.clip(cmd_v, -self.max_reverse_speed, self.max_v))
        cmd_w = float(np.clip(cmd_w, -self.max_w, self.max_w))

        if self.heading_guard_enabled:
            guarded_cmd_v, guarded_cmd_w = apply_heading_guard(
                cmd_v,
                cmd_w,
                self.waypoint_heading,
                max_angular_cmd=self.max_w,
                slowdown_angle=self.heading_guard_slowdown_angle,
                turn_in_place_angle=self.heading_guard_turn_in_place_angle,
            )
            if abs(guarded_cmd_v - cmd_v) > 1.0e-5 or abs(guarded_cmd_w - cmd_w) > 1.0e-5:
                self.throttle_log(
                    "heading_guard",
                    "info",
                    "夹角保护生效: "
                    f"heading={np.rad2deg(self.waypoint_heading):.1f}deg, "
                    f"v={cmd_v:.3f}->{guarded_cmd_v:.3f}, "
                    f"w={cmd_w:.3f}->{guarded_cmd_w:.3f}",
                    1.0,
                )
            cmd_v = guarded_cmd_v
            cmd_w = guarded_cmd_w

        if self.goal_distance < self.goal_reached_dist:
            cmd_v = 0.0
            cmd_w = 0.0

        if self.safety_filter is not None:
            try:
                cmd_v, cmd_w = self.safety_filter.filter(
                    cmd_v,
                    cmd_w,
                    np.asarray(self.latest_scan.ranges, dtype=np.float32),
                    angle_min=float(self.latest_scan.angle_min),
                    angle_increment=float(self.latest_scan.angle_increment),
                    max_range=self.max_lidar_range,
                )
            except Exception as exc:
                self.throttle_log("safety_filter", "warn", f"安全过滤失败，回退到未过滤命令: {exc}", 2.0)

        twist = Twist()
        twist.linear.x = cmd_v
        twist.angular.z = cmd_w
        self.cmd_pub.publish(twist)
        self.last_action = action


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = GeoNavNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # pragma: no cover - 运行期保护
        print(f"[geo_nav_node] 异常退出: {exc}")
        traceback.print_exc()
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

```

## `workspaces/ros2_ws/src/dashgo_rl_ros2/launch/real_model_nav.launch.py`

```python
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("dashgo_rl_ros2")
    dashgo_params = LaunchConfiguration("dashgo_params")
    nav2_params = LaunchConfiguration("nav2_params")
    map_yaml = LaunchConfiguration("map")
    model_path = LaunchConfiguration("model_path")
    rviz_config = LaunchConfiguration("rviz_config")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_amcl = LaunchConfiguration("use_amcl")
    use_rviz = LaunchConfiguration("use_rviz")

    default_dashgo_params = os.path.join(pkg_share, "config", "dashgo_rl.yaml")
    default_nav2_params = os.path.join(pkg_share, "config", "nav2_planning.yaml")
    default_map = os.path.join(pkg_share, "maps", "nav.yaml")
    default_model_path = os.path.join(pkg_share, "models", "policy_torchscript.pt")
    default_rviz_config = os.path.join(pkg_share, "rviz", "dashgo_nav.rviz")

    lifecycle_nodes = ["map_server", "planner_server"]
    lifecycle_nodes_with_amcl = ["map_server", "planner_server", "amcl"]

    return LaunchDescription(
        [
            DeclareLaunchArgument("dashgo_params", default_value=default_dashgo_params),
            DeclareLaunchArgument("nav2_params", default_value=default_nav2_params),
            DeclareLaunchArgument("map", default_value=default_map),
            DeclareLaunchArgument("model_path", default_value=default_model_path),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz_config),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("use_amcl", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            Node(
                package="nav2_map_server",
                executable="map_server",
                name="map_server",
                output="screen",
                parameters=[nav2_params, {"use_sim_time": use_sim_time, "yaml_filename": map_yaml}],
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                parameters=[nav2_params, {"use_sim_time": use_sim_time}],
            ),
            Node(
                package="nav2_amcl",
                executable="amcl",
                name="amcl",
                output="screen",
                condition=IfCondition(use_amcl),
                parameters=[nav2_params, {"use_sim_time": use_sim_time}],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_map_to_odom",
                condition=UnlessCondition(use_amcl),
                arguments=[
                    "--x",
                    "0",
                    "--y",
                    "0",
                    "--z",
                    "0",
                    "--roll",
                    "0",
                    "--pitch",
                    "0",
                    "--yaw",
                    "0",
                    "--frame-id",
                    "map",
                    "--child-frame-id",
                    "odom",
                ],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_planning",
                output="screen",
                condition=UnlessCondition(use_amcl),
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "autostart": True,
                        "node_names": lifecycle_nodes,
                    }
                ],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_planning_amcl",
                output="screen",
                condition=IfCondition(use_amcl),
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "autostart": True,
                        "node_names": lifecycle_nodes_with_amcl,
                    }
                ],
            ),
            Node(
                package="dashgo_rl_ros2",
                executable="goal_plan_bridge",
                name="goal_plan_bridge",
                output="screen",
                parameters=[dashgo_params, {"use_sim_time": use_sim_time}],
            ),
            Node(
                package="dashgo_rl_ros2",
                executable="geo_nav_node",
                name="geo_nav_node",
                output="screen",
                parameters=[dashgo_params, {"use_sim_time": use_sim_time, "model_path": model_path}],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                condition=IfCondition(use_rviz),
                arguments=["-d", rviz_config],
                parameters=[{"use_sim_time": use_sim_time}],
            ),
        ]
    )

```

## `workspaces/ros2_ws/src/dashgo_rl_ros2/launch/real_robot_nav.launch.py`

```python
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    dashgo_rl_share = get_package_share_directory("dashgo_rl_ros2")
    dashgo_driver_share = get_package_share_directory("dashgo_driver_ros2")
    lakibeam_share = get_package_share_directory("lakibeam_driver_ros2")

    base_params = LaunchConfiguration("base_params")
    lidar_params = LaunchConfiguration("lidar_params")
    static_tf_params = LaunchConfiguration("static_tf_params")
    dashgo_params = LaunchConfiguration("dashgo_params")
    nav2_params = LaunchConfiguration("nav2_params")
    map_yaml = LaunchConfiguration("map")
    model_path = LaunchConfiguration("model_path")
    rviz_config = LaunchConfiguration("rviz_config")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_amcl = LaunchConfiguration("use_amcl")
    use_rviz = LaunchConfiguration("use_rviz")

    default_base_params = os.path.join(dashgo_driver_share, "config", "dashgo_driver.yaml")
    default_lidar_params = os.path.join(lakibeam_share, "config", "lakibeam_driver.yaml")
    default_static_tf = os.path.join(dashgo_driver_share, "config", "laser_static_tf.yaml")
    default_dashgo_params = os.path.join(dashgo_rl_share, "config", "dashgo_rl.yaml")
    default_nav2_params = os.path.join(dashgo_rl_share, "config", "nav2_planning.yaml")
    default_map = os.path.join(dashgo_rl_share, "maps", "nav.yaml")
    default_model_path = os.path.join(dashgo_rl_share, "models", "policy_torchscript.pt")
    default_rviz_config = os.path.join(dashgo_rl_share, "rviz", "dashgo_nav.rviz")

    base_launch = os.path.join(dashgo_driver_share, "launch", "base_only.launch.py")
    lidar_launch = os.path.join(lakibeam_share, "launch", "lidar_only.launch.py")
    nav_launch = os.path.join(dashgo_rl_share, "launch", "real_model_nav.launch.py")

    return LaunchDescription(
        [
            DeclareLaunchArgument("base_params", default_value=default_base_params),
            DeclareLaunchArgument("lidar_params", default_value=default_lidar_params),
            DeclareLaunchArgument("static_tf_params", default_value=default_static_tf),
            DeclareLaunchArgument("dashgo_params", default_value=default_dashgo_params),
            DeclareLaunchArgument("nav2_params", default_value=default_nav2_params),
            DeclareLaunchArgument("map", default_value=default_map),
            DeclareLaunchArgument("model_path", default_value=default_model_path),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz_config),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("use_amcl", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(base_launch),
                launch_arguments={"params_file": base_params}.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(lidar_launch),
                launch_arguments={
                    "params_file": lidar_params,
                    "static_tf_params": static_tf_params,
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(nav_launch),
                launch_arguments={
                    "dashgo_params": dashgo_params,
                    "nav2_params": nav2_params,
                    "map": map_yaml,
                    "model_path": model_path,
                    "rviz_config": rviz_config,
                    "use_sim_time": use_sim_time,
                    "use_amcl": use_amcl,
                    "use_rviz": use_rviz,
                }.items(),
            ),
        ]
    )

```

## `workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_controller_core.py`

```python
import numpy as np

from dashgo_rl_ros2.controller_core import (
    apply_heading_guard,
    ObservationBuffer,
    compute_velocity_scaled_lookahead,
    encode_goal_vector,
    process_lidar_ranges,
    select_progressive_waypoint_index,
    scale_linear_speed_by_heading,
    select_waypoint_index,
)


def test_observation_buffer_stacks_history_in_order():
    buffer = ObservationBuffer(history_len=3, obs_dim=4)
    buffer.update(np.array([1, 2, 3, 4], dtype=np.float32))
    buffer.update(np.array([5, 6, 7, 8], dtype=np.float32))

    stacked = buffer.stacked()

    np.testing.assert_array_equal(
        stacked,
        np.array(
            [
                0, 0, 0, 0,
                1, 2, 3, 4,
                5, 6, 7, 8,
            ],
            dtype=np.float32,
        ),
    )


def test_process_lidar_ranges_uses_min_pooling_for_dense_scan():
    dense_scan = np.linspace(0.1, 7.2, 360, dtype=np.float32)
    processed = process_lidar_ranges(dense_scan, lidar_dim=72, max_range=12.0)

    assert processed.shape == (72,)
    rolled = np.roll(dense_scan, -(dense_scan.shape[0] // 2))
    assert np.isclose(processed[0], rolled[:5].min() / 12.0)
    assert np.isclose(processed[-1], rolled[-5:].min() / 12.0)


def test_process_lidar_ranges_interpolates_for_sparse_scan():
    sparse_scan = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    processed = process_lidar_ranges(sparse_scan, lidar_dim=8, max_range=12.0)

    assert processed.shape == (8,)
    rolled = np.roll(sparse_scan, -(sparse_scan.shape[0] // 2))
    assert np.isclose(processed[0], rolled[0] / 12.0)
    assert np.isclose(processed[-1], rolled[-1] / 12.0)


def test_process_lidar_ranges_respects_explicit_front_index():
    scan = np.arange(12, dtype=np.float32)
    processed = process_lidar_ranges(scan, lidar_dim=3, max_range=12.0, front_index=3, normalize=False)

    rolled = np.roll(scan, -3)
    np.testing.assert_array_equal(processed, rolled.reshape(3, 4).min(axis=1))


def test_encode_goal_vector_uses_sin_cos_and_normalized_distance():
    encoded = encode_goal_vector(distance=2.0, angle=np.pi / 2.0, max_distance=8.0)

    np.testing.assert_allclose(
        encoded,
        np.array([0.25, 1.0, 0.0], dtype=np.float32),
        atol=1.0e-6,
    )


def test_compute_velocity_scaled_lookahead_uses_forward_rule():
    assert np.isclose(compute_velocity_scaled_lookahead(0.0), 0.6)
    assert np.isclose(compute_velocity_scaled_lookahead(0.2), 0.6)
    assert np.isclose(compute_velocity_scaled_lookahead(0.3), 0.9)
    assert np.isclose(compute_velocity_scaled_lookahead(0.8), 1.2)


def test_compute_velocity_scaled_lookahead_uses_reverse_rule():
    assert np.isclose(compute_velocity_scaled_lookahead(-0.05), 0.45)
    assert np.isclose(compute_velocity_scaled_lookahead(-0.3), 0.6)
    assert np.isclose(compute_velocity_scaled_lookahead(-0.6), 0.8)


def test_select_waypoint_index_returns_first_distance_over_threshold():
    distances = [0.2, 0.7, 1.1, 1.8]
    assert select_waypoint_index(distances, waypoint_dist=1.0) == 2


def test_select_waypoint_index_falls_back_to_last_pose():
    distances = [0.2, 0.3, 0.9]
    assert select_waypoint_index(distances, waypoint_dist=1.0) == 2


def test_select_waypoint_index_uses_forward_speed_scaled_lookahead():
    distances = [0.25, 0.58, 0.62, 0.95]
    lookahead = compute_velocity_scaled_lookahead(0.1)

    assert np.isclose(lookahead, 0.6)
    assert select_waypoint_index(distances, waypoint_dist=lookahead) == 2


def test_select_waypoint_index_uses_reverse_speed_scaled_lookahead():
    distances = [0.2, 0.42, 0.47, 0.75]
    lookahead = compute_velocity_scaled_lookahead(-0.1)

    assert np.isclose(lookahead, 0.45)
    assert select_waypoint_index(distances, waypoint_dist=lookahead) == 2


def test_scale_linear_speed_by_heading_keeps_speed_for_small_heading_error():
    scaled = scale_linear_speed_by_heading(0.3, np.deg2rad(10.0))

    assert np.isclose(scaled, 0.3)


def test_scale_linear_speed_by_heading_reduces_speed_for_medium_heading_error():
    scaled = scale_linear_speed_by_heading(
        0.3,
        np.deg2rad(45.0),
        slowdown_angle=np.deg2rad(25.0),
        turn_in_place_angle=np.deg2rad(65.0),
    )

    assert 0.0 < scaled < 0.3


def test_scale_linear_speed_by_heading_stops_for_large_heading_error():
    scaled = scale_linear_speed_by_heading(0.3, np.deg2rad(90.0))

    assert np.isclose(scaled, 0.0)


def test_apply_heading_guard_turns_in_place_for_large_heading_error():
    guarded_v, guarded_w = apply_heading_guard(
        0.3,
        -1.0,
        np.deg2rad(90.0),
        max_angular_cmd=1.0,
    )

    assert np.isclose(guarded_v, 0.0)
    assert np.isclose(guarded_w, 1.0)


def test_apply_heading_guard_overrides_wrong_turn_direction():
    guarded_v, guarded_w = apply_heading_guard(
        0.3,
        -0.8,
        np.deg2rad(40.0),
        max_angular_cmd=1.0,
    )

    assert 0.0 < guarded_v < 0.3
    assert guarded_w > 0.0


def test_select_progressive_waypoint_index_skips_old_path_points_behind_robot():
    path_points = np.array(
        [
            [-1.0, 0.0],
            [-0.5, 0.0],
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    index = select_progressive_waypoint_index(path_points, lookahead_dist=0.6)

    assert index == 4


def test_select_progressive_waypoint_index_falls_back_to_nearest_when_all_points_behind():
    path_points = np.array(
        [
            [-0.2, 0.0],
            [-0.4, 0.0],
            [-0.8, 0.0],
        ],
        dtype=np.float32,
    )

    index = select_progressive_waypoint_index(path_points, lookahead_dist=0.6)

    assert index == 2

```

## `workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_safety_filter.py`

```python
import numpy as np

from dashgo_rl_ros2.safety_filter import DynamicsSafetyFilter


def test_safety_filter_preserves_safe_forward_motion():
    filt = DynamicsSafetyFilter(robot_radius=0.2, max_accel=1.0)
    scan = np.full(360, 5.0, dtype=np.float32)

    safe_v, safe_w = filt.filter(0.2, 0.3, scan)

    assert np.isclose(safe_v, 0.2)
    assert np.isclose(safe_w, 0.3)


def test_safety_filter_limits_forward_speed_when_front_blocked():
    filt = DynamicsSafetyFilter(robot_radius=0.2, max_accel=1.0, safety_margin=0.1)
    scan = np.full(360, 5.0, dtype=np.float32)
    scan[170:190] = 0.28

    safe_v, _ = filt.filter(0.3, 0.0, scan)

    assert safe_v < 0.3
    assert safe_v >= 0.0


def test_safety_filter_allows_but_limits_reverse_motion():
    filt = DynamicsSafetyFilter(robot_radius=0.2, max_accel=1.0, safety_margin=0.1)
    scan = np.full(360, 5.0, dtype=np.float32)
    scan[:15] = 0.25
    scan[-15:] = 0.25

    safe_v, _ = filt.filter(-0.2, 0.0, scan)

    assert safe_v <= 0.0
    assert abs(safe_v) < 0.2

```

## `docs/07-ros2-migration/dashgo-real-robot-ros2-deployment_2026-03-20.md`

```markdown
# DashGo ROS2 实车部署说明

> 创建时间: 2026-03-20
> 适用范围: `/home/gwh/dashgo_rl_project/workspaces/ros2_ws`
> 目标: 在 ROS2 Humble 下原生驱动 DashGo 底盘与 Lakibeam 单雷达，并接入 `dashgo_rl_ros2` 的规划与 RL 控制链。

## 结论

当前 ROS2 实车链已经按旧 ROS1 驱动基线对齐：

- 底盘参数以 `drivers/EAI_DRIVER/src/config/my_dashgo_params.yaml` 为唯一真值源。
- 雷达参数以 `drivers/lakibeam_driver/src/launch/lakibeam1_scan.launch` 为单雷达真值源。
- ROS2 实车公共接口保持不变：`/scan`、`/odom`、`/tf`、`/cmd_vel`、`/goal_pose`、`/dashgo/global_plan`。

当前默认参数如下：

- 底盘串口: `/dev/dashgo`
- 波特率: `115200`
- 轮径: `0.1264`
- 轮距: `0.3420`
- 编码器分辨率: `1200`
- PID: `Kp=50`, `Kd=20`, `Ki=0`, `Ko=50`
- 加速度上限: `1.0`
- 雷达 IP: `192.168.8.2`
- 本机监听地址: `0.0.0.0`
- UDP 端口: `2368`
- 雷达 frame: `laser`
- 静态 TF: `base_link -> laser = (0, 0, 0, 0, 0, 0)`

## 代码位置

- 底盘 ROS1 权威源: `drivers/EAI_DRIVER/src/nodes/dashgo_driver.py`
- 雷达 ROS1 权威源: `drivers/lakibeam_driver/src/src/lakibeam1_scan.cpp`
- 底盘 ROS2 包: `workspaces/ros2_ws/src/dashgo_driver_ros2`
- 雷达 ROS2 包: `workspaces/ros2_ws/src/lakibeam_driver_ros2`
- 实车导航 ROS2 包: `workspaces/ros2_ws/src/dashgo_rl_ros2`
- 实车总启动文件: `workspaces/ros2_ws/src/dashgo_rl_ros2/launch/real_robot_nav.launch.py`

## 依赖安装

先确认系统 ROS 版本为 Humble：

```bash
ls /opt/ros
```

安装常用依赖：

```bash
sudo apt update
sudo apt install -y \
  ros-humble-rviz2 \
  ros-humble-nav2-amcl \
  ros-humble-nav2-map-server \
  ros-humble-nav2-planner \
  ros-humble-nav2-lifecycle-manager \
  ros-humble-tf2-ros \
  ros-humble-tf2-geometry-msgs \
  python3-serial \
  python3-yaml \
  python3-numpy \
  libcurl4-openssl-dev
```

如果 `geo_nav_node` 所用模型依赖 TorchScript，请确保 `/usr/bin/python3.10` 环境内可导入 `torch`。

## 串口 udev

旧 ROS1 脚本 `drivers/EAI_DRIVER/src/startup/create_dashgo_udev.sh` 的目标是把底盘串口固定成 `/dev/dashgo`。ROS2 继续沿用这个思路。

创建规则文件：

```bash
sudo tee /etc/udev/rules.d/dashgo.rules >/dev/null <<'RULE'
KERNEL=="ttyACM*", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="0042", MODE:="0666", GROUP:="dialout", SYMLINK+="dashgo"
RULE

sudo tee /etc/udev/rules.d/ch34x.rules >/dev/null <<'RULE'
KERNEL=="ttyUSB*", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", MODE:="0666", GROUP:="dialout", SYMLINK+="dashgo"
RULE

sudo udevadm control --reload-rules
sudo udevadm trigger
```

确认设备名：

```bash
ls -l /dev/dashgo
```

如果没有权限，补用户组：

```bash
sudo usermod -aG dialout $USER
```

重新登录后再验证。

## 雷达网络配置

Lakibeam 默认 IP 采用旧 ROS1 单雷达配置：`192.168.8.2`。

先找到实际接雷达的网卡名：

```bash
ip -br addr
```

给该网卡配置同网段地址，示例：

```bash
sudo ip addr add 192.168.8.10/24 dev <网卡名>
sudo ip link set <网卡名> up
```

连通性验证：

```bash
ping -c 3 192.168.8.2
curl http://192.168.8.2/api/v1/system/firmware
```

如果你的小车雷达不是 `192.168.8.2`，不要改代码，直接复制一份 YAML 后覆盖启动参数：

- 底盘参数文件: `workspaces/ros2_ws/src/dashgo_driver_ros2/config/dashgo_driver.yaml`
- 雷达参数文件: `workspaces/ros2_ws/src/lakibeam_driver_ros2/config/lakibeam_driver.yaml`

## 构建

```bash
cd /home/gwh/dashgo_rl_project/workspaces/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install \
  --packages-select dashgo_driver_ros2 lakibeam_driver_ros2 dashgo_rl_ros2 \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

如果你的 shell 正在激活 conda，优先继续使用上面的 `-DPython3_EXECUTABLE=/usr/bin/python3`。否则 `ament_cmake` 可能误用 conda 的 Python，触发 `No module named 'catkin_pkg'`。

## 分阶段上车验收

### 1. 只验底盘

启动底盘驱动：

```bash
ros2 launch dashgo_driver_ros2 base_only.launch.py
```

另开终端执行：

```bash
cd /home/gwh/dashgo_rl_project/workspaces/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic echo /odom --once
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

验收标准：

- `/odom` 持续更新。
- 按键前进、后退、原地转向都能执行。
- 松键后车辆减速并最终停车。
- 停掉节点后车辆不会继续运动。

### 2. 只验雷达

启动雷达驱动：

```bash
ros2 launch lakibeam_driver_ros2 lidar_only.launch.py
```

另开终端执行：

```bash
cd /home/gwh/dashgo_rl_project/workspaces/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic hz /scan
ros2 run tf2_ros tf2_echo base_link laser
```

验收标准：

- `/scan` 频率稳定。
- `base_link -> laser` 静态 TF 可查询。
- 雷达网络不丢包到无法成圈。

### 3. 验规划 + RL 控制，不启 AMCL

```bash
ros2 launch dashgo_rl_ros2 real_robot_nav.launch.py use_amcl:=false use_rviz:=true
```

另开终端手动发目标点：

```bash
cd /home/gwh/dashgo_rl_project/workspaces/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped '{header: {frame_id: map}, pose: {position: {x: 1.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}'
```

验收标准：

- `/dashgo/global_plan` 有路径输出。
- `/cmd_vel` 持续输出控制指令。
- 小车能依据局部策略响应路径方向和障碍物。

### 4. 验 AMCL + 实际地图

```bash
ros2 launch dashgo_rl_ros2 real_robot_nav.launch.py \
  use_amcl:=true \
  map:=/绝对路径/你的地图.yaml \
  use_rviz:=true
```

验收标准：

- RViz 中 `map -> odom -> base_link -> laser` 关系正常。
- 在 RViz 发送目标点后，能完成全局规划和局部控制闭环。

## 常用启动方式

只起底盘：

```bash
ros2 launch dashgo_driver_ros2 base_only.launch.py
```

只起雷达：

```bash
ros2 launch lakibeam_driver_ros2 lidar_only.launch.py
```

起完整实车导航：

```bash
ros2 launch dashgo_rl_ros2 real_robot_nav.launch.py
```

自定义参数文件：

```bash
ros2 launch dashgo_rl_ros2 real_robot_nav.launch.py \
  base_params:=/绝对路径/base.yaml \
  lidar_params:=/绝对路径/lidar.yaml \
  static_tf_params:=/绝对路径/laser_tf.yaml \
  dashgo_params:=/绝对路径/dashgo_rl.yaml \
  nav2_params:=/绝对路径/nav2.yaml
```

## 失败门槛与回退

- 如果 `base_only.launch.py` 下 teleop 不能稳定驱动，先停在底盘层，不继续上 RL/Nav2。
- 如果 `lidar_only.launch.py` 下 `/scan` 不稳定，先修网络与雷达参数，不改底盘 MCU 固件。
- 只有在底盘与雷达都通过、但原生 ROS2 启动链仍无法稳定上线时，才考虑 ROS1 + bridge 作为临时保底。

## 现实边界

这次代码已经把参数来源、串口协议、雷达默认网络参数和 ROS2 话题接口全部锁到仓库中的旧驱动基线。

但“真的都可以驱动”这件事，最终仍然必须以实车四步验收为准，因为当前环境不能直接替你连接底盘串口和雷达网口。也就是说：

- 参数一致性可以在代码和测试里保证。
- 实际电气连通、底盘方向定义、轮胎磨损、雷达安装误差，只能通过上车验收确认。

因此建议先通过第 1 步和第 2 步，再放开第 3 步和第 4 步。

```
