# DashGo D1 网格模型文件夹

> 创建时间: 2026-01-28 00:20:00
> 状态: 占位符（使用简化几何模型）

## 说明

此文件夹目前为空，因为dashgo_d1_sim.urdf.xacro使用简化几何体（圆柱体），
不需要.stl或.dae网格文件。

## 以后添加精细模型

如果有精细的3D模型文件（.stl或.dae），请放在这里：
- chassis.stl (底盘)
- wheel.stl (轮子)
- lidar.stl (雷达)

然后在URDF中使用mesh标签引用：
<visual>
  <geometry>
    <mesh filename="package://dashgo_rl/meshes/chassis.stl"/>
  </geometry>
</visual>
