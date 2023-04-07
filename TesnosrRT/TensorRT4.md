# Demo Yolov5

这篇文章将从头到尾运行一个yolov5模型，来学习整个TensorRT的运行过程，以及如何进行程序编写。

由于当前的神经网络的框架较多，主要有tf，torch等框架。为了能够方便统一TensorRT的代码架构，并且方便理解和修改。
为此，本demo采用onnx --> TRT的方式进行TensorRT的部署推理。

主要分为以下的步骤：

## 转化onnx模型

当前的主流神经网络的推理框架分为 Tensorflow 和 pytorch (caffe啥的基本已近被主流的所抛弃了)
目前本篇文章所采用的demo是，将fp32模型转换成onnx，再将onnx模型转换成trt。该种方式可以应对当前绝大数的情况。

### Tensorflow转onnx

调用tf2onnx库，详细的步骤可以参考下方的官方链接：

https://onnxruntime.ai/docs/tutorials/tf-get-started.html#getting-started-converting-tensorflow-to-onnx

（该方式没有进行尝试，但是应该是可靠的，毕竟是onnx官方推荐的方式）

### torch转onnx

调用torch.onnx.export函数可以将其torch转为onnx

```
def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    check_requirements('onnx>=1.12.0')
    import onnx

    LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
    f = file.with_suffix('.onnx')

    output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output0']
    if dynamic:
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
        if isinstance(model, SegmentationModel):
            dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
            dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
        elif isinstance(model, DetectionModel):
            dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic or None)

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Metadata
    d = {'stride': int(max(model.stride)), 'names': model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)

    # Simplify
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(('onnxruntime-gpu' if cuda else 'onnxruntime', 'onnx-simplifier>=0.4.1'))
            import onnxsim

            LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f'{prefix} simplifier failure: {e}')
    return f, model_onnx
```

## 将onnx转为trt模型


### 解析onnx graph：

1. 创建一个TensorRT的logger

```
trt_logger = trt.Logger(trt.Logger.INFO)
```

2. 初始化TensorRT插件，允许在TensorRT推理中使用自定义层和激活函数

```
init_libnvinfer_plugins(self.trt_logger, namespace="")
```

3. 创建了一个TensorRT（trt）Builder实例和一个BuilderConfig对象
Builder用于通过优化训练的神经网络的计算图构建TensorRT引擎

```
builder = trt.Builder(self.trt_logger)
```

BuilderConfig对象用于为Builder指定各种配置选项，例如最大批处理大小、精度模式和工作区大小。
```
config = self.builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (2 ** 30)) # 1G
```

4. 使用标志位是用来指定TensorRT网络的创建方式。（即使用EXPLICIT_BATCH模式来创建显式的batch维度）

```
network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
```

5. 通过TensorRT库中的create_network方法来创建新的网络，传入network_flags参数来指定网络创建的方式

```
network = self.builder.create_network(network_flags)
```
6. 创建一个OnnxParser对象，用于解析ONNX模型

```
parser = trt.OnnxParser(self.network, self.trt_logger)
```
7. 打开ONNX模型文件

```
with open(onnx_path, "rb") as f:
    if not self.parser.parse(f.read()):
        print("Failed to load ONNX file: {}".format(onnx_path))
        for error in range(self.parser.num_errors):
            print(self.parser.get_error(error))
        sys.exit(1)
```

8. 获取网络的输入和输出

```
inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]
```

### 建立一个 TensorRT engine
1. 设置TensorRT引擎构建器的标志位，指定了严格类型限制

```
config.set_flag(trt.BuilderFlag.STRICT_TYPES)
```

2. 判断当前平台是否支持fp16/int8

```
#fp16
if not self.builder.platform_has_fast_fp16: #判断当前平台是否支持fp16
    print("FP16 is not supported natively on this platform/device")
else:
    self.config.set_flag(trt.BuilderFlag.FP16)
#int8
 if not self.builder.platform_has_fast_int8: #判断当前平台是否支持int8
    print("INT8 is not supported natively on this platform/device")
else:
    self.config.set_flag(trt.BuilderFlag.INT8)
```

3. 使用TensorRT的Builder对象（self.builder）将深度学习模型编译成可执行的TensorRT引擎，并将其序列化后写入文中

```
with self.builder.build_serialized_network(self.network, self.config) as engine, open(engine_path, "wb") as f:
            print("Serializing engine to file: {:}".format(engine_path))
            f.write(engine)  # .serialize()
```

注：
如果选择的是int8的量化，可以选择PTQ或者QAT量化。目前仅介绍PTQ量化，QAT量化将在后续的文档中更新
PTQ量化需要提供一个校准图片
```
#设置int8量化校准器
self.config.int8_calibrator = EngineCalibrator(calib_cache)
#检查量化校准器缓存文件是否存在，如果该文件不存在，则需要对模型进行量化校准。
if not os.path.exists(calib_cache):
    calib_shape = [calib_batch_size] + list(inputs[0].shape[1:]) #calib的数据维度 8+除了batch_size的其他维度
    calib_dtype = trt.nptype(inputs[0].dtype) #量化校准器输入数据的数据类型，这里使用了trt.nptype()函数将TensorFlow数据类型转换为TensorRT数据类型。
    #指定了一个用于生成校准数据的图像批处理器（ImageBatcher）, ImageBatcher将校准数据（calib_input）作为输入，并对其进行一系列的变换（例如缩放、裁剪、归一化等），以生成一批适当的数据样本，用于训练量化器。
    #calib_shape指定了校准数据的形状，calib_dtype指定了数据类型，max_num_images指定了生成的最大图像数，exact_batches指定了是否生成精确数目的批次
    self.config.int8_calibrator.set_image_batcher(
        ImageBatcher(calib_input, calib_shape, calib_dtype, max_num_images=calib_num_images,
                        exact_batches=True))
```
