#!/usr/bin/env python
"""
GeoNavPolicy v3.2 Sim2Real 部署节点 (架构师修正版 + 安全增强)

修复点:
1. ✅ 实现了与 RSL-RL 完全一致的历史帧堆叠 (History Buffer)
2. ✅ 增加了 /odom 订阅以获取真实速度
3. ✅ 对齐了观测空间维度 (246维)
4. ✅ [新增] 动态控制频率 (从launch读取)
5. ✅ [新增] 加速度数学修正 (解决角加速度计算错误)
6. ✅ [新增] 模型加载时维度熔断检查
7. ✅ [新增] 使用rospkg优化模型路径

作者: Isaac Sim Architect
版本: v3.2-Safe
日期: 2026-01-28
"""
import rospy
import rospkg
import torch
import numpy as np
import collections
import tf2_ros
import os
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf2_geometry_msgs import do_transform_pose

# ==============================================================================
# 1. 观测缓冲区类（核心修正）
# ==============================================================================
class ObservationBuffer:
    """
    管理观测历史的环形缓冲区

    功能:
    - 维护历史帧堆叠（3帧）
    - 自动挤掉最旧的观测
    - 输出堆叠后的Tensor [1, 246]
    """
    def __init__(self, history_len=3, obs_dim=82):
        self.history_len = history_len
        self.obs_dim = obs_dim

        # 初始化为全0，避免启动抖动
        self.buffer = collections.deque(maxlen=history_len)
        for _ in range(history_len):
            self.buffer.append(np.zeros(obs_dim, dtype=np.float32))

    def update(self, current_obs):
        """
        插入最新一帧观测

        Args:
            current_obs: numpy array [82]
        """
        assert current_obs.shape[0] == self.obs_dim, \
            f"观测维度错误: 期望{self.obs_dim}, 实际{current_obs.shape[0]}"
        self.buffer.append(current_obs)

    def get_stacked_obs(self):
        """
        获取堆叠后的Tensor [1, 246]

        Returns:
            torch.Tensor: [1, 246] - 历史帧堆叠
        """
        # RSL-RL通常是将历史帧拼接在一起
        stacked = np.concatenate(list(self.buffer))
        return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)


# ==============================================================================
# 2. 主节点类
# ==============================================================================
class GeoNavNode:
    def __init__(self):
        rospy.init_node('geo_nav_node', anonymous=False)

        # =========================================================
        # 1. [新增] 动态控制频率配置
        # =========================================================
        self.control_rate = rospy.get_param('~control_rate', 20)
        self.dt = 1.0 / self.control_rate
        rospy.loginfo(f"📊 控制频率: {self.control_rate}Hz (dt={self.dt:.4f}s)")

        # =========================================================
        # 2. [新增] 加速度限制参数（从launch读取）
        # =========================================================
        self.max_acc_lin = rospy.get_param('~max_lin_acc', 1.0)  # m/s²
        self.max_acc_ang = rospy.get_param('~max_ang_acc', 0.6)  # rad/s²
        rospy.loginfo(f"🛡️  加速度限制: Lin={self.max_acc_lin} m/s², Ang={self.max_acc_ang} rad/s²")

        # =========================================================
        # 3. [修正] 模型路径优化（使用rospkg动态查找）
        # =========================================================
        try:
            default_model_path = os.path.join(
                rospkg.RosPack().get_path('dashgo_rl'),
                'models/policy_torchscript.pt'
            )
        except rospkg.ResourceNotFound:
            # Fallback到相对路径
            default_model_path = '../models/policy_torchscript.pt'
            rospy.logwarn(f"⚠️ 未找到dashgo_rl包，使用相对路径：{default_model_path}")

        self.model_path = rospy.get_param('~model_path', default_model_path)

        # --- 其他参数配置 ---
        self.max_v = rospy.get_param('~max_lin_vel', 0.3)  # 线速度缩放
        self.max_w = rospy.get_param('~max_ang_vel', 1.0)  # 角速度缩放
        self.lidar_dim = 72  # 训练时的雷达采样数
        self.single_obs_dim = 82  # 72(Lidar) + 2(Target) + 3(LinVel) + 3(AngVel) + 2(Action)
        self.history_len = 3
        self.total_input_dim = self.single_obs_dim * self.history_len  # 246

        # --- 4. 加载模型 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"使用设备: {self.device}")

        rospy.loginfo(f"加载模型: {self.model_path}")
        try:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()

            # =========================================================
            # 5. [新增] 维度熔断检查（致命问题防护）
            # =========================================================
            rospy.loginfo("🔍 正在验证模型输入维度...")

            # 🔥 修正：封装为字典（架构师建议）
            raw_tensor = torch.randn(1, self.total_input_dim).to(self.device)
            dummy_input_dict = {"policy": raw_tensor}  # 键名必须是 "policy"

            try:
                dummy_output = self.model(dummy_input_dict)
                rospy.loginfo(f"✅ 维度检查通过：输入{raw_tensor.shape} → 输出{dummy_output.shape}")
            except Exception as dim_error:
                rospy.logerr(f"💀 致命错误：模型维度不匹配！")
                rospy.logerr(f"   模型期望输入：Dict[str, Tensor] 格式")
                rospy.logerr(f"   期望键名：'policy'")
                rospy.logerr(f"   错误信息：{dim_error}")
                rospy.signal_shutdown("Dimension Mismatch")
                exit(1)

            rospy.loginfo("✅ 模型加载成功")
        except Exception as e:
            rospy.logerr(f"❌ 模型加载失败: {e}")
            exit(1)

        # --- 6. 状态管理 ---
        self.obs_buffer = ObservationBuffer(self.history_len, self.single_obs_dim)
        self.last_action = np.zeros(2, dtype=np.float32)
        self.current_vel = np.zeros(6, dtype=np.float32)  # [vx, vy, vz, wx, wy, wz]
        self.goal_polar = np.zeros(2, dtype=np.float32)  # [dist, heading]
        self.latest_scan = None

        # ========== [新增] 保存完整路径用于到达判定 ==========
        self.global_path = None  # 保存完整路径
        # ========================================

        # ========== MVP新增：全局路径追踪 ==========
        self.local_waypoint = None
        self.waypoint_dist = 1.0  # 固定1m前瞻距离

        # 订阅全局路径话题（诊断结果：/move_base/NavfnROS/plan）
        from nav_msgs.msg import Path
        plan_topic = "/move_base/NavfnROS/plan"
        self.path_sub = rospy.Subscriber(
            plan_topic, Path, self.mvp_path_cb,
            queue_size=1  # 避免路径堆积
        )

        rospy.loginfo("✅ MVP模式：已启用全局路径追踪")
        rospy.loginfo(f"   监听话题: {plan_topic}")
        # =============================================

        # --- 7. ROS通讯 ---
        self.tf_buf = tf2_ros.Buffer()
        self.tf_lis = tf2_ros.TransformListener(self.tf_buf)

        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        rospy.Subscriber('/scan', LaserScan, self.scan_cb, queue_size=1)
        rospy.Subscriber('/odom', Odometry, self.odom_cb, queue_size=1)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_cb, queue_size=1)

        # [修正] 使用动态频率
        rospy.Timer(rospy.Duration(self.dt), self.control_loop)

        rospy.loginfo("=" * 80)
        rospy.loginfo("✅ GeoNav Sim2Real 节点启动就绪 (安全增强版 v3.2)")
        rospy.loginfo(f"   - 观测维度: {self.total_input_dim} (单帧: {self.single_obs_dim} × {self.history_len})")
        rospy.loginfo(f"   - LiDAR降采样: {self.lidar_dim}维")
        rospy.loginfo(f"   - 历史帧堆叠: {self.history_len}帧")
        rospy.loginfo(f"   - 加速度限制: {self.max_acc_lin*self.dt:.4f}/{self.max_acc_ang*self.dt:.4f} per tick")
        rospy.loginfo("=" * 80)
        rospy.loginfo("🎯 等待目标点...")

    def odom_cb(self, msg):
        """
        更新机器人当前速度

        注意: Isaac Sim训练使用的是base_link坐标系下的速度
        ROS的odom通常也是base_link下的速度 (child_frame_id)
        """
        self.current_vel[0] = msg.twist.twist.linear.x
        self.current_vel[1] = msg.twist.twist.linear.y
        self.current_vel[2] = msg.twist.twist.linear.z
        self.current_vel[3] = msg.twist.twist.angular.x
        self.current_vel[4] = msg.twist.twist.angular.y
        self.current_vel[5] = msg.twist.twist.angular.z

    def goal_cb(self, msg):
        """目标点回调"""
        self.goal_pose = msg
        rospy.loginfo(f"🎯 收到新目标: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

        # [架构师建议] 重置观测缓冲区，避免旧状态干扰
        self.obs_buffer = ObservationBuffer(self.history_len, self.single_obs_dim)
        self.last_action = np.zeros(2, dtype=np.float32)

    def scan_cb(self, msg):
        """LiDAR回调（存储最新扫描）"""
        self.latest_scan = msg

    def mvp_path_cb(self, msg):
        """MVP版全局路径回调（修正版算法）

        核心逻辑：追踪路径上前方约1m的点
        """
        # ========== [新增] 保存完整路径 ==========
        self.global_path = msg  # 保存完整路径用于到达判定
        # ========================================

        if not msg.poses:
            rospy.logwarn("⚠️ 收到空路径")
            return

        try:
            # 1. 获取TF变换（base_link ← map）
            trans = self.tf_buf.lookup_transform(
                "base_link", "map",
                rospy.Time(0), rospy.Duration(0.1)
            )

            # 2. 遍历路径，寻找前方约1m的点
            for i, pose in enumerate(msg.poses):
                pose_in_base = do_transform_pose(pose, trans)
                dist = np.sqrt(
                    pose_in_base.pose.position.x**2 +
                    pose_in_base.pose.position.y**2
                )

                if dist >= self.waypoint_dist:
                    self.local_waypoint = pose
                    rospy.loginfo_throttle(2.0,
                        f"✅ 追踪航点: idx={i}/{len(msg.poses)}, dist={dist:.2f}m")
                    return

            # 3. Fallback：所有点都<1m，追踪终点
            self.local_waypoint = msg.poses[-1]
            rospy.loginfo("🏁 接近终点，追踪最后一点")

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, f"⚠️ TF查询失败: {e}")

    def process_lidar(self, msg):
        """
        [架构师修正] 将任意线数的雷达处理成训练时的72维格式

        处理流程:
        1. 替换Inf/NaN
        2. Min-Pooling降采样到72维（保留最近障碍物）
        3. 填充不足的点

        修正原因：
        - 等间隔采样可能漏掉近距离障碍物
        - Min-Pooling确保每个扇区保留最近点（安全优先）
        """
        raw_ranges = np.array(msg.ranges, dtype=np.float32)

        # 1. 替换 Inf/NaN
        raw_ranges = np.nan_to_num(raw_ranges, nan=12.0, posinf=12.0, neginf=0.0)
        raw_ranges = np.clip(raw_ranges, 0.0, 12.0)

        input_len = len(raw_ranges)

        # 2. Min-Pooling降采样（架构师修正 - 安全优先）
        if input_len >= self.lidar_dim:
            # 计算每个扇区的大小 (向下取整)
            sector_size = input_len // self.lidar_dim

            # 截断多余的点，确保能被整除
            truncated_len = self.lidar_dim * sector_size
            raw_truncated = raw_ranges[:truncated_len]

            # Reshape 成 (72, N) 然后在第二个维度取 Min
            # 这样每个扇区取最小值（最近障碍物）
            processed = raw_truncated.reshape(self.lidar_dim, sector_size).min(axis=1)
        else:
            # 如果点数不够（罕见），进行线性插值
            rospy.logwarn_throttle(5.0, f"⚠️ 雷达点数不足 ({input_len} < {self.lidar_dim})，进行插值")
            indices = np.linspace(0, input_len-1, self.lidar_dim)
            processed = np.interp(indices, np.arange(input_len), raw_ranges)

        # 3. 如果凑不够72个点，进行填充
        if len(processed) < self.lidar_dim:
            padding = np.zeros(self.lidar_dim - len(processed)) + 12.0
            processed = np.concatenate([processed, padding])

        # 注意: 这里不除以12.0，因为网络中有LayerNorm会自动归一化
        return processed

    def update_goal_polar(self):
        """
        计算目标点在机器人坐标系下的极坐标 (dist, heading)

        MVP修改：优先级调整（支持局部航点追踪）
        Returns:
            bool: 是否成功获取目标
        """
        # ========== MVP修改：优先级调整 ==========
        # 优先级1: 追踪局部航点（方案C）
        if self.local_waypoint is not None:
            target = self.local_waypoint
        # 优先级2: 追踪最终目标（fallback）
        elif hasattr(self, 'goal_pose'):
            target = self.goal_pose
        else:
            return False
        # ========================================

        try:
            # 获取robot -> target的变换
            trans = self.tf_buf.lookup_transform(
                'base_link',
                target.header.frame_id,  # 使用动态target而非固定的goal_pose
                rospy.Time(0),
                rospy.Duration(0.1)
            )

            # 将目标点转换到base_link坐标系
            target_in_base = do_transform_pose(target, trans)

            dx = target_in_base.pose.position.x
            dy = target_in_base.pose.position.y

            dist = np.sqrt(dx**2 + dy**2)
            heading = np.arctan2(dy, dx)

            self.goal_polar = np.array([dist, heading], dtype=np.float32)
            return True

        except Exception as e:
            rospy.logwarn_throttle(2.0, f"⚠️ TF查询失败: {e}")
            return False

    def control_loop(self, event):
        """
        主控制循环 (20Hz)

        流程:
        1. 更新目标向量
        2. 组装当前帧观测
        3. 更新历史Buffer
        4. 获取堆叠观测 [1, 246]
        5. 模型推理
        6. 动作后处理
        7. 安全保护
        8. 发布命令
        """
        if self.latest_scan is None:
            return

        # 1. 更新目标向量
        has_goal = self.update_goal_polar()
        if not has_goal:
            return # 没有目标就不动

        # ========== [新增] 到达判定逻辑 ==========
        dist = self.goal_polar[0]

        # 判断是否到达终点
        if hasattr(self, 'global_path') and self.global_path is not None and self.global_path.poses:
            # 检查当前追踪的点是否是路径终点
            # 注意：需要比较pose对象本身，而非位置坐标
            is_last_waypoint = (self.local_waypoint is not None and
                                len(self.global_path.poses) > 0 and
                                self.local_waypoint == self.global_path.poses[-1])

            if dist < 0.3 and is_last_waypoint:
                rospy.loginfo("🏁 已到达终点，停车")
                # 发送零速度
                stop_cmd = Twist()
                self.pub_cmd.publish(stop_cmd)
                # 清除目标，防止抖动
                self.local_waypoint = None
                self.goal_pose = None
                return  # 跳过后续控制逻辑
        # ========================================

        # 2. 组装当前帧观测 (Single Frame Obs)
        # 结构: LiDAR(72) + Target(2) + LinVel(3) + AngVel(3) + LastAction(2) = 82
        lidar_data = self.process_lidar(self.latest_scan)

        # 注意维度拼接顺序，必须与Isaac Sim里的顺序一模一样！
        current_obs_vec = np.concatenate([
            lidar_data,                 # 72
            self.goal_polar,            # 2
            self.current_vel[:3],       # 3 (Lin Vel)
            self.current_vel[3:],       # 3 (Ang Vel)
            self.last_action            # 2
        ]).astype(np.float32)

        # 3. 更新历史Buffer
        self.obs_buffer.update(current_obs_vec)

        # 4. 获取网络输入 (Stacked History) -> [1, 246]
        input_tensor = self.obs_buffer.get_stacked_obs().to(self.device)

        # 5. 推理
        with torch.no_grad():
            # 🔥 修正：封装为字典（架构师建议）
            obs_dict = {"policy": input_tensor}  # 键名必须是 "policy"
            action = self.model(obs_dict).cpu().numpy()[0]  # 输出通常是raw action (未缩放)

        # 6. 动作后处理
        # 假设训练时output range是[-1, 1]或者无限制
        # 这里需要映射回真实速度
        # 如果你的GeoNavPolicy最后没有Tanh，输出可能是任意值
        action = np.clip(action, -10.0, 10.0)

        # 缩放 (Scale)
        cmd_v = action[0] * self.max_v  # 线速度
        cmd_w = action[1] * self.max_w  # 角速度

        # [架构师修正] 软件限速与加速度限制 (Safety Filter - 数学修正版)
        # 🔥 关键修正：根据物理加速度限制计算每周期限制
        # acc_per_tick = max_acc * dt
        acc_lin_per_tick = self.max_acc_lin * self.dt
        acc_ang_per_tick = self.max_acc_ang * self.dt

        # 计算上一次的真实速度（从之前的action恢复）
        last_cmd_v = self.last_action[0] * self.max_v
        last_cmd_w = self.last_action[1] * self.max_w

        # 限制速度变化量（使用动态计算的加速度限制）
        cmd_v = np.clip(cmd_v, last_cmd_v - acc_lin_per_tick, last_cmd_v + acc_lin_per_tick)
        cmd_w = np.clip(cmd_w, last_cmd_w - acc_ang_per_tick, last_cmd_w + acc_ang_per_tick)

        # 7. 安全保护 (Sim2Real Gap保护)
        if self.goal_polar[0] < 0.2:  # 到达目标
            cmd_v = 0.0
            cmd_w = 0.0

        # 绝对倒车禁止（双重保障）
        if cmd_v < -0.05:
            rospy.logwarn_throttle(1.0, "🚫 倒车已禁止")
            cmd_v = 0.0

        # 8. 发布
        twist = Twist()
        twist.linear.x = cmd_v
        twist.angular.z = cmd_w
        self.pub_cmd.publish(twist)

        # 更新状态
        self.last_action = action

# ==============================================================================
# 3. 主函数
# ==============================================================================
if __name__ == '__main__':
    try:
        node = GeoNavNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("🛑 节点已停止")
    except Exception as e:
        rospy.logerr(f"❌ 节点异常退出: {e}")
        import traceback
        traceback.print_exc()
