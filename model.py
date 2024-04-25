import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore import Tensor, CSRTensor, COOTensor
from mindspore.common.initializer import One, Normal


class CNN(nn.Cell):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 4)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 2)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(2, 2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x


if __name__ == "__main__":
    cnn = CNN()
    print(cnn)
    x = ms.Tensor(shape=[8, 3, 224, 224], dtype=ms.float32, init=Normal())
    print(x.shape)
    y = cnn(x)
    print(y.shape)
