# PolarFree-Lite-Visual-v2

一个面向本科毕设场景的轻量级偏振图像去反射项目。项目参考 PolarFree 的“偏振输入 + 先验引导 + 深度去反射网络”思想，但不复现原版扩散模型，而是设计了一套低算力、可解释、可在个人 GPU 上训练和演示的 PolarFree-Lite-Visual 方法。

本版本重点增强了：

- 强光高亮反射去除
- 横向灯带 / 线状反射压制
- 大面积玻璃雾状反射去除
- 窗格 / 建筑倒影抑制
- DirectClean 残差扣除
- `visual_plus` / `visual_extreme` 展示效果
- hard case 难例加权训练

> 项目目标：在有限算力条件下实现具有较强视觉冲击力的偏振图像去反射结果，服务于本科毕业设计论文、实验对比和答辩展示。

---

## 1. 项目特点

### 1.1 偏振物理先验

项目使用 4 个偏振角图像：

```text
0deg
45deg
90deg
135deg
```

从中计算平均图像、偏振振幅、DoLP、swing、std、darkest view、continuous minimum candidate、aggressive percentile candidate 和 visual prior。

这些先验用于辅助网络判断反射区域和背景区域。

---

### 1.2 深度网络去反射

项目采用轻量 U-Net 结构，输入多通道偏振特征和先验信息，输出多个 head：

- DirectClean
- positive reflection residual
- signed clean delta
- reflection confidence
- line-band confidence
- text-reflection confidence
- tint confidence
- low-frequency reflection confidence
- dark-shadow confidence
- blend / support mask

当前配置：

```python
MODEL_INPUT_CHANNELS = 31
MODEL_OUTPUT_CHANNELS = 16
```

---

### 1.3 强反射接管式融合

本项目的核心思想是：

```text
强反射 / 强光区域：
    由 DirectClean / residual clean 强制接管

普通非反射区域：
    保护背景结构、颜色和纹理
```

最终输出支持：

```text
raw
hybrid
visual
visual_plus
visual_extreme
```

其中 `visual_extreme` 更偏向展示效果，适合答辩图像展示。

---

## 2. 项目文件结构

```text
PolarFree-Lite-Visual-v2/
│
├── polarfree_lite_glass_v2.py
│   └── 主入口脚本，负责训练、测试、参数解析
│
├── main.py
│   └── 偏振图像读取、偏振特征计算、物理先验和基础 mask 构造
│
├── pfl_config.py
│   └── 全局配置，包括输入输出通道、cache version、metric/debug 字段
│
├── pfl_data.py
│   └── 数据集扫描、样本组织、cache 读写、hard case 处理
│
├── pfl_model.py
│   └── U-Net 模型、head 解码、DirectClean、residual clean、融合逻辑
│
├── pfl_losses.py
│   └── 损失函数，包括强光、低频、线状反射、hard case 等专项 loss
│
├── pfl_eval.py
│   └── 测试评估、图片保存、debug mask 输出、CSV 指标统计
│
├── AGENTS.md
│   └── Codex / coding agent 项目级约束文件
│
├── .gitignore
│   └── 忽略权重、cache、输出结果、数据集等大文件
│
└── README.md
    └── 项目说明文档
```

---

## 3. 环境要求

推荐环境：

```text
Windows 11
Python 3.10+
PyTorch
CUDA GPU
NVIDIA RTX 4050 Laptop GPU 或更高
```

用户当前常用环境：

```text
Conda 环境名：pytorch
Python 路径：E:\ana\envs\pytorch\python.exe
项目路径：C:\Users\86166\Desktop\PythonProject
训练数据：D:\jibi\train
测试数据：D:\jibi\data\test
输出目录：D:\jibi\output
```

---

## 4. 数据集格式

项目期望数据集中包含 `input` 和 `gt` 两类文件夹，每个 scene 下包含多角度偏振图像。

示例结构：

```text
D:\jibi\train
├── input
│   └── scene_xxx
│       ├── 0000_000.png
│       ├── 0000_045.png
│       ├── 0000_090.png
│       ├── 0000_135.png
│       └── 0000_rgb.png
│
└── gt
    └── scene_xxx
        ├── 0000_000.png
        ├── 0000_045.png
        ├── 0000_090.png
        ├── 0000_135.png
        └── 0000_rgb.png
```

角度后缀对应关系：

```text
000 -> 0deg
045 -> 45deg
090 -> 90deg
135 -> 135deg
```

---

## 5. 快速语法检查

在训练前建议先运行：

```powershell
cd C:\Users\86166\Desktop\PythonProject

&E:\ana\envs\pytorch\python.exe -m py_compile `
  .\main.py `
  .\pfl_config.py `
  .\pfl_data.py `
  .\pfl_model.py `
  .\pfl_losses.py `
  .\pfl_eval.py `
  .\polarfree_lite_glass_v2.py
```

如果无输出，一般表示语法检查通过。

---

## 6. 快速测试

使用已有 checkpoint 测试少量图片：

```powershell
cd C:\Users\86166\Desktop\PythonProject

&E:\ana\envs\pytorch\python.exe -u .\polarfree_lite_glass_v2.py `
  --mode test `
  --test_root D:\jibi\data\test `
  --save_dir D:\jibi\output\eval_v2_quick `
  --weights 你的checkpoint路径\best_model.pth `
  --model quality `
  --img_size 384 `
  --batch_size 1 `
  --amp `
  --channels_last `
  --prior_profile visual `
  --mask_profile aggressive `
  --loss_profile visual `
  --render-mode visual_extreme `
  --showcase_mode `
  --showcase_hard_mode `
  --package_cache `
  --save_eval_images `
  --test_limit 5
```

重点查看：Raw、Prior、DirectClean、VisualPlus、VisualExtreme、strong_glare_core、takeover_mask。

---

## 7. 推荐训练流程

### 7.1 DirectClean 预训练

目标：先让 DirectClean 学会明显压制反射。

```powershell
cd C:\Users\86166\Desktop\PythonProject

&E:\ana\envs\pytorch\python.exe -u .\polarfree_lite_glass_v2.py `
  --mode train `
  --train_root D:\jibi\train `
  --test_root D:\jibi\data\test `
  --save_dir D:\jibi\output\direct_pretrain_v2 `
  --model quality `
  --img_size 384 `
  --batch_size 1 `
  --grad_accum_steps 4 `
  --epochs 8 `
  --direct_pretrain_epochs 8 `
  --lr 8e-5 `
  --weight_decay 1e-4 `
  --eval_every 2 `
  --test_limit 20 `
  --num_workers 0 `
  --render-mode visual_extreme `
  --prior_profile visual `
  --mask_profile aggressive `
  --loss_profile visual `
  --package_cache `
  --save_eval_images `
  --amp `
  --channels_last
```

---

### 7.2 Visual 主训练

目标：让 Raw / VisualPlus / VisualExtreme 整体稳定优于 Prior。

```powershell
&E:\ana\envs\pytorch\python.exe -u .\polarfree_lite_glass_v2.py `
  --mode train `
  --train_root D:\jibi\train `
  --test_root D:\jibi\data\test `
  --save_dir D:\jibi\output\visual_main_v2 `
  --weights 上一步真实checkpoint路径\best_model.pth `
  --model quality `
  --img_size 384 `
  --batch_size 1 `
  --grad_accum_steps 4 `
  --epochs 15 `
  --lr 8e-5 `
  --weight_decay 1e-4 `
  --eval_every 2 `
  --test_limit 20 `
  --num_workers 0 `
  --render-mode visual_extreme `
  --prior_profile visual `
  --mask_profile aggressive `
  --loss_profile visual `
  --package_cache `
  --save_eval_images `
  --amp `
  --channels_last
```

---

### 7.3 Showcase 微调

目标：服务答辩展示图，优先肉眼反射压制效果。

```powershell
&E:\ana\envs\pytorch\python.exe -u .\polarfree_lite_glass_v2.py `
  --mode train `
  --train_root D:\jibi\train `
  --test_root D:\jibi\data\test `
  --save_dir D:\jibi\output\showcase_ft_v2 `
  --weights 上一步真实checkpoint路径\best_model.pth `
  --model quality `
  --img_size 384 `
  --batch_size 1 `
  --grad_accum_steps 4 `
  --epochs 4 `
  --lr 2e-5 `
  --weight_decay 1e-4 `
  --eval_every 1 `
  --test_limit 20 `
  --num_workers 0 `
  --render-mode visual_extreme `
  --prior_profile visual `
  --mask_profile aggressive `
  --loss_profile visual `
  --visual_strength 1.6 `
  --showcase_reflection_boost 1.45 `
  --showcase_mode `
  --showcase_hard_mode `
  --package_cache `
  --save_eval_images `
  --amp `
  --channels_last
```

---

## 8. Cache 说明

本版本升级了 mask / prior / package 逻辑，因此 cache version 已更新。

示例：

```python
PRIOR_CACHE_VERSION = "quality_v6_visual_v2_masks"
PACKAGE_CACHE_VERSION = "quality_tensor_v14_visual_v2_masks"
```

第一次运行时，程序可能会自动重建：

```text
D:\jibi\train\.polarfree_cache
D:\jibi\train\.polarfree_tensor_cache
D:\jibi\data\test\.polarfree_cache
D:\jibi\data\test\.polarfree_tensor_cache
```

这是正常现象，不需要手动删除数据集。

---

## 9. GitHub 备份注意事项

本项目不建议提交以下文件：

```text
*.pth
*.pt
*.ckpt
*.onnx
*.npz
*.npy
output/
outputs/
runs/
checkpoints/
.polarfree_cache/
.polarfree_tensor_cache/
D:\jibi\
```

建议只提交：

```text
*.py
*.md
*.txt
*.json
*.yaml
*.yml
.gitignore
AGENTS.md
README.md
```

---

## 10. 毕设论文表述建议

可以在论文中这样描述本方法：

> 本文参考 PolarFree 中偏振图像去反射的思想，设计了一种轻量化 PolarFree-Lite-Visual 方法。该方法利用四角度偏振图像构建物理先验，并通过轻量 U-Net 学习反射残差、干净图像估计和反射置信度。在强反射区域，模型采用 DirectClean 残差扣除与强接管式融合策略；在非反射区域，则通过背景保护机制保持图像结构和颜色稳定。该方法在有限算力条件下提升了强光、灯带、玻璃雾状反射等复杂场景的视觉去反射效果，适用于本科毕业设计中的算法实现与实验展示。

---

## 11. 当前版本核心思路

```text
PolarFree-Lite-Visual v2
=
偏振物理先验
+ 多源强光检测
+ 反射残差学习
+ DirectClean 强接管融合
+ 强光 / 低频 / 线状反射专项 loss
+ hard case 难例加权
+ visual_extreme 展示模式
```

---

## 12. 免责声明

本项目主要用于本科毕业设计、算法学习与实验展示。对于严重过曝、完全饱和、背景信息已经丢失的强光区域，模型只能根据上下文进行合理恢复，不能保证真实细节的完全重建。
