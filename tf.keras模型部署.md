本文档记录了如何将tf.keras训练好的模型部署到tensorflow serving中。

具体的过程如下：
1. 使用tf.keras构建网络训练模型
2. 将tf.keras模型导出为TS可加载形式(tf.saved_model.builder.SavedModelBuilder)
3. 使用TS服务加载模型
4. 编写client使用服务


## 数据下载
```shell
python download.py
```

```python
# download.py
import tensorflow as tf
import os, cv2

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt-top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

i = 0
for im, label in zip(test_images, test_labels):
    if not os.path.exists(class_names[label]):
        os.mkdir(class_names[label])
    cv2.imwrite(class_names[label]+"/"+str(i)+".jpg", im)
    i += 1

    if i > 100:
        break
```

## 模型训练和保存
```shell
python train_and_save.py
```

```python
# train_and_save.py
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt-top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(), 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

loss, acc = model.evaluate(test_images, test_labels)
print("Test collection accuracy: {:5.2f}%".format(100*acc))

model.save('fashion_mnist_model.h5')
```

## 模型转化

```shell
python model_convert.py --keras_model_path=./fashion_mnist_model.h5 --model_version=1 models/fashion-mnist
```

```python
# model_convert.py

from __future__ import print_function

import os
import sys

import tensorflow as tf

tf.app.flags.DEFINE_string("keras_model_path", "", "the path of keras h5 model file.")
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS


def main(_):
    if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
        print('Usage: model_convert.py [--keras_model_path=x] '
        '[--model_version=y] export_dir')
        sys.exit(-1)
    if FLAGS.keras_model_path.strip() == "":
        print('Please specify the path for keras h5 model file.')
        sys.exit(-1)
    if FLAGS.model_version < 0:
        print('Please specify a positive value for version number.')
        sys.exit(-1)

    new_model = tf.keras.models.load_model(FLAGS.keras_model_path)
    new_model.summary()

    if (new_model.uses_learning_phase):
        raise ValueError('Model using learning phase.')

    export_path_base = sys.argv[-1]
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(FLAGS.model_version)))
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'images_name': new_model.input}, 
        outputs={'outputs_name': new_model.output})

    builder.add_meta_graph_and_variables(
        sess=tf.keras.backend.get_session(),                                                                                                                    
        tags=[tf.saved_model.tag_constants.SERVING],                                                                                             
        signature_def_map={                                                                                                                      
            'classification': signature,                                                                                                                     
        },
        main_op=tf.tables_initializer())
    builder.save()


if __name__ == '__main__':
    tf.app.run()
```

## 启动TS服务，加载模型

```shell
tensorflow_model_server --port=8500 --model_name="fashion" --model_base_path="/home/hengd/Desktop/WorkSpace/tf-serving-examples/fashion-mnist-example/models/fashion-mnist"
```
注意`--model_base_path=`需要填写完整的绝对路径，`--model_name=`需要与client服务脚本对应。


## 编写client服务脚本

