本文档记录了Tensorflow Serving的环境搭建及Demo测试过程

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
* [mnist_client.py](https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/example/mnist_client.py)

2. 训练并保存模型
```shell
python mnist_saved_model.py models/mnist
```
该命令会生成`models/mnist`文件夹，目录结构如下:

```
models/mnist
  --- 1/
    --- saved_model.pb  # the serialized tensorflow::SavedModel.
    --- variables #  the serialized variables of the graphs.
```

3. 从docker启动tensorflow serving，并加载训练好的模型
```shell
sudo docker run -p 8500:8500 --mount type=bind,source=/home/hengd/Desktop/WorkSpace/tf-serving-examples/models/mnist,target=/models/mnist -e MODEL_NAME=mnist -t tensorflow/serving
```
4. 测试tensorflow serving
```shell
python mnist_client.py --num_tests=1000 --server=127.0.0.1:8500
Extracting /tmp/train-images-idx3-ubyte.gz
Extracting /tmp/train-labels-idx1-ubyte.gz
Extracting /tmp/t10k-images-idx3-ubyte.gz
Extracting /tmp/t10k-labels-idx1-ubyte.gz
........................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
Inference error rate: 10.4%
```
