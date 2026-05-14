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