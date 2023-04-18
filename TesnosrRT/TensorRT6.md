# TensorRT engine 模型推理

上一章节主要介绍了有关python的推理流程，本章节将介绍有关C++的推理流程代码，其基本的流程与python类似。

## 推理流程

### C++

1. 使用二进制方式读取trt文件，将其存放到字符数组trtModelStream中

```
 std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) //判断文件是否成功打开，如果成功打开则执行后续操作
    {
        file.seekg(0, file.end); //将文件指针移到文件的结尾，这里是为了获取文件的大小。
        size = file.tellg();    //获取文件的大小，并将其保存到变量 size 中。
        file.seekg(0, file.beg);  //将文件指针重新指向文件的开头,这里是为了读取文件内容.
        trtModelStream = new char[size]; //创建一个字符数组（char数组），大小为文件的大小。
        assert(trtModelStream);  //确保创建字符数组成功，如果创建失败程序将终止。
        file.read(trtModelStream, size);  //读取文件的内容到字符数组中。
        file.close();
    }
```

2. 创建一个 TensorRT 的运行时（runtime）实例，以便在 TensorRT 中执行推理（inference）

```
runtime = createInferRuntime(gLogger);
```
3. 使用 runtime 中的 deserializeCudaEngine 函数，将字符数组中的内容反序列化为一个 ICudaEngine 对象。

```
engine = runtime->deserializeCudaEngine(trtModelStream, size);
```

4. 创建一个执行上下文（context）
```
context = engine->createExecutionContext(); 
```

5. engine 的 getBindingDimensions 方法，获取引擎的第 1 个输出张量的维度信息，并将其赋值给变量 out_dims。

```
auto out_dims = engine->getBindingDimensions(1);
```

6. 遍历 out_dims 中的每一个维度，并将其相乘，得到输出张量的大小。

```
for(int j=0;j<out_dims.nbDims;j++) {
        std::cout << "ouput size: " << j  << " " << out_dims.d[j] << std::endl;
        this->output_size *= out_dims.d[j];
    }
```

7. 创建一个大小为 output_size 的 float 类型的数组，用于存放推理结果。

```
this->prob = new float[this->output_size];
```

8. 图片前处理，传统的前处理方案，这边暂不赘述,但是其中需要做一步数据处理的步骤：
将图像转换为浮点数类型的 blob 数据，方便后续将其输入到 TensorRT 模型中进行推理
```
blob = blobFromImage(pr_img)
```

9. 正式推理

(1)获取与推断上下文相关联的CUDA引擎。 context对象是IExecutionContext接口的一个实例，它表示用于在CUDA引擎上执行推断的执行上下文的实例。

```
const ICudaEngine& engine = context.getEngine();
```

(2)创建一个大小为 2 的 void 类型的指针数组，用于存放输入和输出张量的 GPU 缓冲区。

```
void* buffers[2]; 
```

(3)获取了输入输出张量在TensorRT引擎中的绑定索引和引擎的最大批量大小

```
const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
int mBatchSize = engine.getMaxBatchSize();
```

(4)分配输入输出张量的 GPU 缓冲区
```
CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));
```

(5) 创建一个 CUDA 流，用于异步执行 CUDA 操作, 可以在不等待先前提交的操作完成的情况下，继续提交新的操作。

```
cudaStream_t stream;
CHECK(cudaStreamCreate(&stream));
```

(6)使用CUDA的异步内存拷贝函数cudaMemcpyAsync，将主机上的输入数据input从主机内存复制到设备上的输入缓冲区buffers[inputIndex]中。
```
CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
```

(7)使用输入和输出缓冲区以及 CUDA 流在执行上下文中异步执行推理。

```
context.enqueue(1, buffers, stream, nullptr);
```

(8)使用异步内存拷贝函数cudaMemcpyAsync，将设备上的输出数据output从设备内存复制到主机上的输出缓冲区buffers[outputIndex]中。

```
CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));

```

(9)等待所有的异步操作完成

```
cudaStreamSynchronize(stream); 
```

(10)释放stream 和 buffers

```
cudaStreamDestroy(stream);
CHECK(cudaFree(buffers[inputIndex]));
CHECK(cudaFree(buffers[outputIndex]));
```

10. 对输出的结果进行后处理，本篇文章不再赘述，可以查看源码


