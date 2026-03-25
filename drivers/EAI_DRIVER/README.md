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
