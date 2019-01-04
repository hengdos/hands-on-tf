本文档记录了如何通过Tensorboard可视化tf.keras模型训练过程。

模型训练代码如下，其中核心是使用`tf.keras.callbacks.TensorBoard(log_dir='./logs')`将训练过程中的参数记录下来。

```python
import tensorflow as tf

# 以fashion_mnist数据集为例
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt-top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0

# 构建一个基本的模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(), 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# 设置Tensorboard callbacks
callbacks = [
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

# 训练模型
model.fit(train_images, train_labels, epochs=5, batch_size=32,
          validation_data=(test_images, test_labels), callbacks=callbacks)
```

启动Tensorboard查看
```shell
cd $ROOT # ROOT是训练代码的存放路径
tensorboard --logdir=./logs/
```
