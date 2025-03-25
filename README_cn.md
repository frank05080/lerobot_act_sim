# lerobot_act_sim

## 1. 简介

该仓库为LeRobot的ACT模仿学习模型在RDK X5上运行的样例。

具体而言，该仓库使用了ALOHA仿真，该仿真环境模拟了两个机械臂，一个机械臂将积木拾起，并将拾起的积木传递给另一个机械臂的完整过程。

## 2. 环境搭建

这套环境分为两个部分，我们在电脑本地拉起仿真环境，并在RDK X5上完成推理。

我们先将github整个进行拉取：

```
https://github.com/frank05080/lerobot_act_sim.git
```

注意到该github包含两个部分，lerobot和onboard，其中lerobot将拉起仿真环境，在本地运行。而onboard会在RDK X5上运行，负责模型的推理部分。

### 2.1 本地环境搭建

本地环境推荐使用Ubuntu 20.04或Ubuntu 22.04，且在该系统中安装有conda环境。我们使用conda创建虚拟环境：

```
conda create -n lerobot python=3.10
```

激活虚拟环境

```
conda activate lerobot
```

随后，我们找到一个文件夹（假设是主目录，即~），我们将lerobot复制过来，并进入该目录：

```
cd lerobot
```

在该虚拟环境中进行依赖的安装：

```
pip install -e .
```

这会将相关依赖和lerobot包安装到虚拟环境中。此外，我们再进行其他依赖的安装：

```
pip install onnx onnxruntime gym-aloha
```

随后，我们将huggingface上的模型配置进行拉取，我们前往：https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human

我们将所有文件下载下来，放置在~/lerobot_act/lerobot/outputs/train/act_aloha_sim_transfer_cube_human目录中

### 2.2 RDK环境安装

我们使用一块已经烧好镜像的SD卡插入至RDK X5 开发板。登录进去。

我们参考RDK MODEL ZOO仓库，完成bpu_infer_lib的安装，Model Zoo网址：https://github.com/D-Robotics/rdk_model_zoo

随后，我们将该仓库中的onboard文件夹复制到RDK X5环境中的某处（这里假设是主目录下，即~）

## 3. 样例运行

### 3.1 RDK环境运行

在RDK X5中，我们首先切换到onboard目录下：

```
cd onboard
```

假设环境已按上述描述完成安装，则此时我们可以运行：

```
python3 main.py
```

### 3.2 本地环境运行

随后，我们切换至本地环境，切换到examples目录：

```
cd ~/lerobot_act_sim/lerobot/examples
```

我们对eval_bpu.py的两个变量进行修改，分别为：

1. OUTPUT_VIDEO_PATH：即仿真结束后的视频保存的路径
2. WEIGHT_PATH：即之前下载huggingface相关配置的路径，默认为~/lerobot_act/lerobot/outputs/train/act_aloha_sim_transfer_cube_human中

完成对上述变量的修改后，我们启动仿真环境：

```
python eval_bpu.py
```

我们稍作等待，直到看到如下回显：

```
step=340 reward=2 terminated=False
step=341 reward=2 terminated=False
step=342 reward=2 terminated=False
step=343 reward=2 terminated=False
step=344 reward=4 terminated=True
Success!
Video of the evaluation is available in '/home/ros/share_dir/gitrepos/lerobot_act/lerobot/outputs/eval/act_aloha_sim_transfer_cube_human/rollout.mp4'.
```