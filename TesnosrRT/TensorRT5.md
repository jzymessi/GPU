# TensorRT engine 模型推理

上篇文章主要讲述了如何将onnx模型转化为trt模型，这篇文章主要介绍如何对推理出来的trt模型进行推理。
目前TensorRT提供了python和C++两套推理框架，下面将详细介绍这两种方式具体步骤，根据我的理解这两中方式所用的步骤应该是差不多的，只是调用的API有所不同，本章节主要介绍python的调用流程，C++的调用流程见下个章节。

## 推理流程

### python 

1. 创建一个名为pred的预测器对象，用于对输入数据进行推理
(1) 创建一个TensorRT的logger，定义Logger 的初始最低严重性级别。
```
logger = trt.Logger(trt.Logger.WARNING)
logger.min_severity = trt.Logger.Severity.ERROR
```
(2) 创建一个运行时
```
runtime = trt.Runtime(logger)
```
(3)初始化TensorRT的plugins
```
python:
trt.init_libnvinfer_plugins(logger,'')
```

2. 创建一个执行上下文对象（context）
```
context = engine.create_execution_context()
```

3. 创建一个CUDA流对象,用于在GPU上异步地执行TensorRT推理计算

```
stream = cuda.Stream()
```

4. 读取trt文件，将引擎文件反序列化为一个CUDA引擎。
```
with open(engine_path, "rb") as f:
    serialized_engine = f.read()
engine = runtime.deserialize_cuda_engine(serialized_engine)
```
5. 遍历了TensorRT引擎对象中的每个绑定，以便 device 和 host 之间传输

```
for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding))
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    self.bindings.append(int(device_mem))
    if engine.binding_is_input(binding):
        self.inputs.append({'host': host_mem, 'device': device_mem})
    else:
        self.outputs.append({'host': host_mem, 'device': device_mem})
```

6. 图片前处理，传统的前处理方案，这边暂不赘述

7. 正式推理

(1)将名为 "img" 的 Numpy 数组（即输入数据）转化为一维数组（使用np.ravel()函数），并将其赋值给 TensorRT 引擎的第一个输入张量的 "host" 属性
```
self.inputs[0]['host'] = np.ravel(img)
```
(2)将 TensorRT 引擎的输入数据从主机内存（host）异步地复制到设备内存（device）
```
for inp in self.inputs:
    cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
```
(3)执行异步推理计算
```
context.execute_async_v2(bindings=self.bindings,stream_handle=self.stream.handle)
```
(4)将 TensorRT 引擎的输出结果从设备内存（device）异步地复制到主机内存（host）
```
for out in self.outputs:
    cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
```
(5) 等待 CUDA 流中的所有异步任务完成
```
stream.synchronize()
```
(6)将TensorRT 推理的输出结果从设备内存复制到主机内存，并将其存储在一个列表中
```
data = [out['host'] for out in self.outputs]
```

8. 对输出的结果进行后处理，本篇文章不再赘述，可以查看源码


