# ROS yaml模块缺失修复脚本

> **创建时间**: 2026-01-28 23:05:00
> **问题**: ROS节点找不到yaml模块
> **根本原因**: shebang使用conda环境的python3，而系统python3才有yaml模块

---

## 🔧 快速修复（复制粘贴执行）

### 步骤1: 备份原文件

```bash
sudo cp /opt/ros/noetic/lib/joint_state_publisher/joint_state_publisher \
           /opt/ros/noetic/lib/joint_state_publisher/joint_state_publisher.bak
```

### 步骤2: 修改shebang为绝对路径

```bash
sudo sed -i '1s|#!/usr/bin/env python3|#!/usr/bin/python3|' \
           /opt/ros/noetic/lib/joint_state_publisher/joint_state_publisher
```

### 步骤3: 验证修改

```bash
head -n 1 /opt/ros/noetic/lib/joint_state_publisher/joint_state_publisher
# 应显示: #!/usr/bin/python3
```

### 步骤4: 启动ROS节点

```bash
roslaunch dashgo_rl sim2real_golden.launch
```

---

## 📊 技术细节

### 问题诊断

```bash
# 当前shebang
#!/usr/bin/env python3  # ❌ 使用conda环境 (无yaml)

# 应该改为
#!/usr/bin/python3      # ✅ 使用系统Python (有yaml)
```

### 验证

```bash
# 检查conda环境python3（无yaml）
/home/gwh/.conda/envs/env_isaaclab/bin/python3 -c "import yaml"
# 结果: ModuleNotFoundError

# 检查系统python3（有yaml）
/usr/bin/python3 -c "import yaml"
# 结果: 成功
```

---

## ⚠️ 注意事项

1. **需要sudo权限**: 修改系统文件需要管理员权限
2. **ROS更新后可能需要重新修改**: 系统更新ROS时会恢复shebang
3. **建议记录到部署文档**: 方便后续查找

---

## 🔙 回滚方法

如果需要恢复原文件：

```bash
sudo cp /opt/ros/noetic/lib/joint_state_publisher/joint_state_publisher.bak \
           /opt/ros/noetic/lib/joint_state_publisher/joint_state_publisher
```
