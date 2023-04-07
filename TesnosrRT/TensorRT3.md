# TensorRT 入门

前面两篇已经基本介绍了TensorRT一些基础的信息，以及安装步骤，这篇文章将主要描述TensorRT是如何运行的。

## TensorRT的工作

TensorRT主要的工作可以分为两个部分：构建期和运行期

### 构建期（推理优化器）

模型解析/建立           （加载ONNX等其他格式的模型/使用原生的API搭建模型）

计算图优化              （横向层融合（Conv），纵向层融合（Conv+add+ReLU）....）

节点消除                （去除无用层，节点变换（Pad, Slice，Concat，Shuffle），....）

多精度支持              （FP32 / FP16 / INT8 / TF32（可能插入 reformat 节点））

优选kernel/format       （硬件有关优化）

导入 plugin             （实现自定义操作）

显存优化                （显存池复用）

### 运行期（运行时环境）

运行时环境              （对象生命期管理，内存显存管理，异常处理）

序列化反序列化          （推理引擎保存为文件或从文件中加载）

## 基本流程 

刚刚上面介绍的为TensorRT的工作流程，而在实际的使用当中，很多参数的配置都包含在了同一个API里面。

基本的操作流程为：

➢ 构建期 

建立 Builder（引擎构建器）

创建 Network（计算图内容）

生成 SerializedNetwork（网络的 TRT 内部表示）

➢ 运行期

建立 Engine 和 Context

Buffer 相关准备（Host 端 + Device 端 + 拷贝操作）

执行推理（Execute）
