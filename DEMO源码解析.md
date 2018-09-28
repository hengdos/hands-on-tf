本文档对Tensorflow Serving DEMO代码进行了详细说明.

使用Tensorflow Serving部署服务包含两个步骤：
* (1) 使用`SavedModelBuilder`将模型保存;
* (2) 使用Docker和TS镜像加载模型，提供端口服务;
* (3) 服务端访问端口，获取服务。

## 模型保存

使用 SavedModelBuilder module 可以将训练好的模型导出成TS服务可加载的形式，下面的代码解释了整个过程。

```python
export_path_base = sys.argv[-1]
export_path = os.path.join(
      compat.as_bytes(export_path_base),
      compat.as_bytes(str(FLAGS.model_version)))
print 'Exporting trained model to', export_path
builder = tf.saved_model.builder.SavedModelBuilder(export_path)
builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
           'predict_images':
               prediction_signature,
           signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
               classification_signature,
      },
      legacy_init_op=legacy_init_op)
builder.save()
```

核心步骤：
```python
# 1. create builder
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

# 2. add meta graph and variables to the builder
builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
           'predict_images':
               prediction_signature,
           signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
               classification_signature,
      },
      legacy_init_op=legacy_init_op)

# 3. save builder
builder.save()
```
注意`builder.add_meta_graph_and_variables()`方法需要传递以下几个重要参数:
* [1] sess, The TensorFlow session from which to save the meta graph and variables.
* [2] tags, The set of tags with which to save the meta graph. 可选参数[REF](https://www.tensorflow.org/api_docs/python/tf/saved_model/tag_constants)：
    * tf.saved_model.tag_constants.SERVING (使用TS服务一般选此项)
    * tf.saved_model.tag_constants.TRAINING
    * tf.saved_model.tag_constants.GPU
    * tf.saved_model.tag_constants.TPU
* [3] signature_def_map, The map of signature def map to add to the meta graph def. Specifies the map of user-supplied key for a signature to a tensorflow::SignatureDef to add to the meta graph. **Signature specifies what type of model is being exported, and the input/output tensors to bind to when running inference.**

`signature_def_map`需要仔细处理，其主要有两个作用:
* [1] 指定导出模型的类型
* [2] 指定inference过程中的输入/输出

其他参数见[REF](https://www.tensorflow.org/api_docs/python/tf/saved_model/builder/SavedModelBuilder)


### TODO
1. 如何构造signature_def_map
2. 如何使用python client访问TS服务
3. 如何使用c++ client访问TS服务
