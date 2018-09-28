# hands-on-tf
Hands-on tensorflow for deploying.

## Tensorflow Serving on Ubuntu 18.04

### 环境配置

1. 安装tensorflow

```shell
sudo pip install tensorflow
```

2. 安装docker

```shell
sudo snap install docker
```

3. 获取tensorflow serving的docker镜像

```shell
docker pull tensorflow/serving
```

检查是否安装成功
```shell
sudo docker image ls
REPOSITORY           TAG                 IMAGE ID            CREATED             SIZE
tensorflow/serving   latest              57d56d4f5df7        6 weeks ago         215MB
```

### Demo测试

1. 下载Demo代码：
* [mnist_saved_model.py](https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/example/mnist_saved_model.py)
* [mnist_input_data.py](https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/example/mnist_input_data.py)

2. 训练并保存模型
```shell
python mnist_saved_model.py models/mnist
```
