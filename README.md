# APFNet 目标检测项目
## APFNet: Attention-Guided Pyramid Fusion Network for Small Object Detection in Remote Sensing Images

## 项目结构

```
RT-DETR/
├── cfg/                  # 模型配置文件目录
├── dataset/              # 数据集目录(YOLO格式)
├── heatmaps/             # 热图可视化结果
├── result/               # 预测结果目录
├── runs/                 # 训练和检测结果
│   ├── train/           # 训练文件目录
│   └── val/              # 验证结果
├── convert_coco_to_yolo.py  # COCO 格式转 YOLO 格式脚本
├── heatmap.py            # 热图可视化脚本
├── blocks.py             # 自定义模块具体实现
└── README.md             # 项目说明文档
```

## 主要功能

### 1. 数据集转换

`convert_coco_to_yolo.py` 脚本用于将 COCO 格式的数据集转换为 YOLO 格式。

**使用方法：**

```bash
python convert_coco_to_yolo.py
```

### 2. 热图可视化

`heatmap.py` 脚本用于生成模型预测的热图，支持多种 CAM (Class Activation Mapping) 方法，包括：

- GradCAM
- GradCAMPlusPlus
- XGradCAM
- EigenCAM
- HiResCAM
- LayerCAM
- RandomCAM
- EigenGradCAM

**使用方法：**

1. 安装依赖：
   ```bash
   pip install grad-cam==1.5.4
   ```

2. 修改 `get_params()` 函数中的参数，设置模型权重路径、设备、CAM 方法等。

3. 运行脚本：
   ```bash
   python heatmap.py
   ```

### 3. 模型训练与推理

本项目基于 Ultralytics YOLO 框架，支持多种模型配置和数据集。

修改Ultralytics/nn/modules/block.py中内容，添加如下类等类别，并在开始的all中添加注册声明，具体类实现在blocks.py中：
```Python
class Atrous_Gateway(nn.Module):

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = ParallelAtrousConv(c_)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
... 
```
修改Ultralytics/nn/modules/__init__.py。

修改Ultralytics/nn/tasks.py的参数推理内容。

**训练示例：**

```bash
# 使用自定义配置训练模型
python train.py --data dataset/DIOR_YOLO/data.yaml --cfg cfg/mine.yaml --epochs 72 --batch-size 16
```

**推理示例：**

```bash
# 使用训练好的模型进行推理
python detect.py --weights runs/train/exp/weights/best.pt --source dataset/DIOR_YOLO/images/test --conf 0.25
```

## 支持的数据集

- **DIOR**：遥感目标检测数据集
- **LEVIR**：遥感目标检测数据集



## 配置文件说明

`cfg/` 目录下包含多种模型配置文件，

## 热图可视化结果

`heatmaps/` 目录下包含使用不同 CAM 方法生成的热图结果，可用于分析模型的注意力机制和预测依据。

## 预测结果

`result/` 目录下包含模型在不同数据集上的预测结果，包括边界框和热图可视化。

## 训练结果

`runs/` 目录下包含模型训练的详细结果，包括：

- 训练和验证损失曲线
- 模型权重文件
- 预测示例
- 评估指标（mAP、F1 分数等）

## 依赖项

- Python 3.7+
- PyTorch 1.8+
- Ultralytics
- OpenCV
- NumPy
- Matplotlib
- tqdm
- grad-cam==1.5.4 (用于热图可视化)

## 安装方法

1. 克隆项目：
   ```bash
   git clone https://github.com/husizhtu/APFNet.git
   cd APFNet
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   # 安装热图可视化依赖
   pip install grad-cam==1.5.4
   ```

## 使用示例

### 1. 转换数据集

```bash
# 将 COCO 格式转换为 YOLO 格式
python convert_coco_to_yolo.py
```

### 2. 训练模型

```bash
# 训练自定义模型
python train.py --data dataset/DIOR_YOLO/data.yaml --cfg cfg/mine.yaml --epochs 72 --batch-size 16

# 训练 R50 模型
python train.py --data dataset/DIOR_YOLO/data.yaml --cfg cfg/rtdetr-r50.yaml --epochs 72 --batch-size 16
```

### 3. 生成热图

```bash
# 修改 get_params() 函数中的参数后运行
python heatmap.py
```

### 4. 模型推理

```bash
# 使用训练好的模型进行推理
python detect.py --weights runs/train/*/weights/best.pt --source dataset/DIOR_YOLO/images/test --conf 0.25
```

## 许可证

本项目基于 AGPL-3.0 许可证。

## 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [RT-DETR](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rtdetr)
- [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam)

## 联系方式

如有问题或建议，请通过以下方式联系：

- Email: virtue9847h@163.com
- GitHub: [husizhtu](https://github.com/husizhtu)
