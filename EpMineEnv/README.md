# 强化学习机器人视觉导航

## SSH远程连接

下面是使用VSCode远程连接服务器的config格式:

```shell
HostName dev-modelarts-cnnorth4.huaweicloud.com
    Port xxxx
    User ma-user
    IdentityFile /path/to/your/private/key
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ForwardAgent yes
```

注意密钥文件的权限问题，最好放在C盘用户目录下。

## Installation

```bash
pip install mlagents-envs gym opencv-python==4.5.5.64 stable-baselines3==1.5.0 importlib-metadata==4.13.0 seaborn
```

## Environment

环境的改进主要包括奖励的记录、以及距离和角度密集奖励的计算方法。

- 奖励的记录：在环境中增加变量，记录每个episode的奖励和步数，对其可视化。
- 奖励的改进：密集奖励是根据上一时刻和当前时刻的距离和角度变化计算的，涉及对密集奖励范围的控制和调整。
- - 距离奖励：上一时刻距目标的距离减去当前时刻距目标的距离，乘以一个尺度系数10，范围限定在$[-1, 1]$。
- - 角度奖励：上一时刻和当前时刻的角度变化，乘以一个尺度系数0.1，范围限定在$[-0.04, 0.04]$。在原始环境中，只对俯仰角度进行了计算，我们增加了偏航角度的计算，当距目标的距离小于0.5时，计算偏航角度奖励，鼓励agent直线前进。
- - 靠近奖励：agent靠近目标0.5距离时直接给予大小为1的奖励，鼓励agent靠近目标（比较了一下去除这个奖励的效果，加上之后可以确保奖励为正，减小震荡）

## Parameter

- algorithm: PPO, A2C, DDPG (TD3 SAC太慢)
- total_timesteps: 1e6
- learning_rate: 5e-4
- max_episode_steps: 1000
- gamma: 0.99
- batch_size: 256

## Result

结果文件可查看txt, log, csv以及tensorboard文件。以下是包含了靠近奖励的结果图，收敛结果并不理想，成功率一般在10%左右，且奖励始终震荡。

DDPG结果跑差不多了...

## Analysis

通过从均匀分布上对动作采样来观察奖励函数分布，以此为参考设置clipping（angle reward是yaw reward和pitch reward的和，最后一张rewards对应每一步的奖励，即distance和angle的和）
