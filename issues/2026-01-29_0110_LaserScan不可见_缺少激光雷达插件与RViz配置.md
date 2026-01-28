# LaserScan不可见问题 - 缺少激光雷达插件与RViz配置

> **创建时间**: 2026-01-29 01:10:00
> **严重程度**: 🟡 警告（传感器数据缺失）
> **状态**: ✅ 已修复
> **相关文件**: dashgo_d1_sim.urdf.xacro, RViz配置
> **对话来源**: dialogue-001-gazebo-rviz-diagnosis.md, dialogue-002-rviz-blind-diagnosis-physics-clipping.md

---

## 问题描述

### 现象1：/scan话题不存在

**描述**: RViz中LaserScan的Topic为空

**影响**:
- ❌ 无法看到激光点云
- ❌ 导航算法无法工作
- ❌ 避障功能失效

### 现象2：点云太小看不见

**描述**: /scan数据存在，但RViz中看不到红点

**原因**:
- 点云尺寸太小（默认0.01）
- 环境空旷，数据为inf被过滤
- Decay Time=0，瞬间消失

---

## 根本原因

### 原因1：缺少Gazebo激光雷达插件

URDF中缺少`libgazebo_ros_laser.so`插件

### 原因2：RViz配置不当

- Size太小（0.01）
- Style不对（Flat Squares）
- Decay Time=0（无轨迹）

---

## 解决方案

### 修复1：添加激光雷达插件

**文件**: `dashgo_d1_sim.urdf.xacro`
**位置**: 第173-204行

```xml
<gazebo reference="laser_link">
  <sensor type="ray" name="laser_sensor">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.10</min>
        <max>12.0</max>
        <resolution>0.01</resolution>
      </range>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.01</stddev>
      </noise>
    </ray>
    <plugin name="laser_controller" filename="libgazebo_ros_laser.so">
      <topicName>/scan</topicName>
      <frameName>laser_link</frameName>
    </plugin>
  </sensor>
</gazebo>
```

**功能**:
- ✅ 发布/scan话题（360度，12m范围）
- ✅ 添加高斯噪声（模拟真实雷达）

### 修复2：修改RViz配置

**RViz左侧面板 > LaserScan**:

| 参数 | 默认值 | 建议值 | 效果 |
|------|--------|--------|------|
| **Topic** | 空 | /scan | 选择正确话题 |
| **Style** | Flat Squares | Points | 圆点更明显 |
| **Size** | 0.01 | 0.1-0.2 | 放大10-20倍 |
| **Decay Time** | 0 | 10 | 显示轨迹 |
| **Alpha** | 1.0 | 1.0 | 完全不透明 |
| **Color Transformer** | Intensity | FlatColor | 颜色固定 |

**架构师建议**:
> "因为仿真环境很空旷，如果点太小，你根本看不见。必须改成0.1甚至0.2。"

---

## 验证方法

### 1. 检查/scan话题

```bash
# 检查数据频率
rostopic hz /scan
# 预期: average rate: 10.0

# 检查数据内容
rostopic echo /scan -n1
# 预期: 输出ranges数组
```

### 2. RViz可视化

- ✅ 能看到一圈红点（或扫描到的墙壁）
- ✅ 点云大小适中（0.1或更大）
- ✅ 点云有轨迹（Decay Time=10的效果）

---

## 预期结果

- ✅ /scan话题正常发布（10Hz）
- ✅ RViz能看到激光点云
- ✅ 导航算法能接收数据
- ✅ 避障功能正常工作

---

## 经验教训

### 1. 点云Size要适应环境

**小房间**（<5m）: Size = 0.05
**大房间**（5-20m）: Size = 0.1
**大空旷**（>20m）: Size = 0.2-0.5

### 2. Decay Time的作用

设置为0: 瞬间消失，看不到历史轨迹
设置为10: 点停留10秒，形成运动轨迹

---

## 相关文档

- **完整对话**:
  - `docs/architect-dialogues/dialogue-001-gazebo-rviz-diagnosis.md`
  - `docs/architect-dialogues/dialogue-002-rviz-blind-diagnosis-physics-clipping.md`
- **URDF文件**: `catkin_ws/src/dashgo_rl/urdf/dashgo_d1_sim.urdf.xacro`

---

**维护者**: Claude Code AI Assistant (基于Isaac Sim架构师诊断)
