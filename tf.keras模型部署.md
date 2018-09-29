本文档记录了如何将tf.keras训练好的模型部署到tensorflow serving中。

具体的过程如下：
1. 使用tf.keras构建网络训练模型
2. 将tf.keras模型导出为TS可加载形式(tf.saved_model.builder.SavedModelBuilder)
3. 使用TS服务加载模型
4. 编写client使用服务
