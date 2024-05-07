# 强化学习机器人视觉导航

## SSH远程连接

下面是使用VSCode远程连接服务器的config格式:

```shell
HostName dev-modelarts-cnnorth4.huaweicloud.com
    Port 31589
    User ma-user
    IdentityFile /path/to/your/private/key
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ForwardAgent yes
```

注意密钥文件的权限问题，最好放在C盘用户目录下。

## Installation

```bash
pip install mlagents-envs gym opencv-python==4.5.5.64 stable-baseline3==1.5.0
```
