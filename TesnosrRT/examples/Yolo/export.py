import os
import sys
import logging
import argparse

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from image_batch import ImageBatcher

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    def set_image_batcher(self, image_batcher: ImageBatcher):
        """
        Define the image batcher to use, if any. If using only the cache file, an image batcher doesn't need
        to be defined.
        :param image_batcher: The ImageBatcher object
        """
        self.image_batcher = image_batcher
        size = int(np.dtype(self.image_batcher.dtype).itemsize * np.prod(self.image_batcher.shape))
        self.batch_allocation = cuda.mem_alloc(size)
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _, _ = next(self.batch_generator)
            log.info("Calibrating image {} / {}".format(self.image_batcher.image_index, self.image_batcher.num_images))
            cuda.memcpy_htod(self.batch_allocation, np.ascontiguousarray(batch))
            return [int(self.batch_allocation)]
        except StopIteration:
            log.info("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                log.info("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        with open(self.cache_file, "wb") as f:
            log.info("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)

class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """
    def __init__(self, verbose=False, workspace=8):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in Gb.
        """
        # Create the TensorRT logger
        '''
        TensorRT的logger类提供了5个不同的日志级别，分别是：VERBOSE、INFO、WARNING、ERROR和INTERNAL_ERROR。
        Logger.Severity.VERBOSE：最详细的级别，记录所有详细的信息，包括调试信息和跟踪信息。
        Logger.Severity.INFO：记录有关TensorRT的基本信息，如版本号、设备信息等。
        Logger.Severity.WARNING：记录TensorRT遇到的警告信息，这些信息通常意味着TensorRT仍然可以工作，但可能存在一些问题。
        Logger.Severity.ERROR：记录TensorRT遇到的错误信息，这些信息通常意味着TensorRT无法正常工作，需要修复问题。
        Logger.Severity.INTERNAL_ERROR：记录TensorRT内部遇到的错误信息，这些信息通常意味着TensorRT无法正常工作，需要修复问题。
        
        '''
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE
        #初始化TensorRT插件，允许在TensorRT推理中使用自定义层和激活函数
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        #创建了一个TensorRT（trt）Builder实例和一个BuilderConfig对象
        #Builder用于通过优化训练的神经网络的计算图构建TensorRT引擎。
        self.builder = trt.Builder(self.trt_logger)
        #BuilderConfig对象用于为Builder指定各种配置选项，例如最大批处理大小、精度模式和工作区大小。
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (2 ** 30)) # 1G
        # self.config.max_workspace_size = workspace * (2 ** 30)  # Deprecation

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path, end2end, conf_thres, iou_thres, max_det, **kwargs):
        """
        解析ONNX图并创建相应的Tensor
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        v8 = kwargs['v8']
        #通过将1左移一个整数值，将整数值转换为一个二进制位表示的标志位，这里的标志位是用来指定TensorRT网络的创建方式，即使用EXPLICIT_BATCH模式来创建显式的batch维度。
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        #通过TensorRT库中的create_network方法来创建新的网络，传入network_flags参数来指定网络创建的方式
        self.network = self.builder.create_network(network_flags)
        #创建一个OnnxParser对象，用于解析ONNX模型
        self.parser = trt.OnnxParser(self.network, self.trt_logger)
        #将ONNX模型的路径转换为绝对路径
        onnx_path = os.path.realpath(onnx_path)
        #打开ONNX模型文件
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                print("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    print(self.parser.get_error(error))
                sys.exit(1)
        #获取网络的输入和输出
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        print("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            print("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        for output in outputs:
            print("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))
        
        #判断batch_size是否大于0，如果不是，则抛出异常
        assert self.batch_size > 0
        # self.builder.max_batch_size = self.batch_size  # This no effect for networks created with explicit batch dimension mode. Also DEPRECATED.

        if end2end:
            previous_output = self.network.get_output(0)
            self.network.unmark_output(previous_output)
            if not v8: 
                # output [1, 8400, 85]
                # slice boxes, obj_score, class_scores
                strides = trt.Dims([1,1,1])
                starts = trt.Dims([0,0,0])
                bs, num_boxes, temp = previous_output.shape
                shapes = trt.Dims([bs, num_boxes, 4])
                # [0, 0, 0] [1, 8400, 4] [1, 1, 1]
                boxes = self.network.add_slice(previous_output, starts, shapes, strides)
                num_classes = temp -5 
                starts[2] = 4
                shapes[2] = 1
                # [0, 0, 4] [1, 8400, 1] [1, 1, 1]
                obj_score = self.network.add_slice(previous_output, starts, shapes, strides)
                starts[2] = 5
                shapes[2] = num_classes
                # [0, 0, 5] [1, 8400, 80] [1, 1, 1]
                scores = self.network.add_slice(previous_output, starts, shapes, strides)
                # scores = obj_score * class_scores => [bs, num_boxes, nc]
                scores = self.network.add_elementwise(obj_score.get_output(0), scores.get_output(0), trt.ElementWiseOperation.PROD)
            else:
                strides = trt.Dims([1,1,1])
                starts = trt.Dims([0,0,0])
                previous_output = self.network.add_shuffle(previous_output)
                previous_output.second_transpose    = (0, 2, 1)
                print(previous_output.get_output(0).shape)
                bs, num_boxes, temp = previous_output.get_output(0).shape
                shapes = trt.Dims([bs, num_boxes, 4])
                # [0, 0, 0] [1, 8400, 4] [1, 1, 1]
                boxes = self.network.add_slice(previous_output.get_output(0), starts, shapes, strides)
                num_classes = temp -4 
                starts[2] = 4
                shapes[2] = num_classes
                # [0, 0, 4] [1, 8400, 80] [1, 1, 1]
                scores = self.network.add_slice(previous_output.get_output(0), starts, shapes, strides)
            '''
            "plugin_version": "1",
            "background_class": -1,  # no background class
            "max_output_boxes": detections_per_img,
            "score_threshold": score_thresh,
            "iou_threshold": nms_thresh,
            "score_activation": False,
            "box_coding": 1,
            '''
            registry = trt.get_plugin_registry()
            assert(registry)
            creator = registry.get_plugin_creator("EfficientNMS_TRT", "1")
            assert(creator)
            fc = []
            fc.append(trt.PluginField("background_class", np.array([-1], dtype=np.int32), trt.PluginFieldType.INT32))
            fc.append(trt.PluginField("max_output_boxes", np.array([max_det], dtype=np.int32), trt.PluginFieldType.INT32))
            fc.append(trt.PluginField("score_threshold", np.array([conf_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
            fc.append(trt.PluginField("iou_threshold", np.array([iou_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
            fc.append(trt.PluginField("box_coding", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32))
            fc.append(trt.PluginField("score_activation", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32))
            
            fc = trt.PluginFieldCollection(fc) 
            nms_layer = creator.create_plugin("nms_layer", fc)

            layer = self.network.add_plugin_v2([boxes.get_output(0), scores.get_output(0)], nms_layer)
            layer.get_output(0).name = "num"
            layer.get_output(1).name = "boxes"
            layer.get_output(2).name = "scores"
            layer.get_output(3).name = "classes"
            for i in range(4):
                self.network.mark_output(layer.get_output(i))


    def create_engine(self, engine_path, precision, calib_input=None, calib_cache=None, calib_num_images=5000,
                      calib_batch_size=8):
        """
        #将建立起来的TensorRT引擎进行构建，然后将其序列化到磁盘上
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        :param calib_batch_size: The batch size to use for the calibration process.
        """

        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        print("Building {} Engine in {}".format(precision, engine_path))
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        # TODO: Strict type is only needed If the per-layer precision overrides are used
        # If a better method is found to deal with that issue, this flag can be removed.
        #设置TensorRT引擎构建器的标志位，指定了严格类型限制。TensorRT会对数据类型进行严格检查，避免不同类型的数据之间进行不合法的操作，以保证代码的正确性和可靠性。
        self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16: #判断当前平台是否支持fp16
                print("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not self.builder.platform_has_fast_int8: #判断当前平台是否支持int8
                print("INT8 is not supported natively on this platform/device")
            else:
                if self.builder.platform_has_fast_fp16:
                    # Also enable fp16, as some layers may be even more efficient in fp16 than int8
                    self.config.set_flag(trt.BuilderFlag.FP16)
                self.config.set_flag(trt.BuilderFlag.INT8)
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

        # with self.builder.build_engine(self.network, self.config) as engine, open(engine_path, "wb") as f:
        #使用TensorRT的Builder对象（self.builder）将深度学习模型编译成可执行的TensorRT引擎，并将其序列化后写入文件中
        #代码中的build_serialized_network函数接受两个参数：一个是TensorRT的网络对象（self.network），另一个是TensorRT的配置对象（self.config）。通过调用这个函数，我们可以将深度学习模型编译成一个可执行的TensorRT引擎，并返回一个序列化后的二进制数据（engine）
        #使用Python的内置函数open打开一个文件，并将序列化后的二进制数据写入文件中。
        with self.builder.build_serialized_network(self.network, self.config) as engine, open(engine_path, "wb") as f:
            print("Serializing engine to file: {:}".format(engine_path))
            f.write(engine)  # .serialize()

def main(args):
    builder = EngineBuilder(args.verbose, args.workspace)
    #创建网络,解析onnx模型
    builder.create_network(args.onnx, args.end2end, args.conf_thres, args.iou_thres, args.max_det, v8=args.v8)
    builder.create_engine(args.engine, args.precision, args.calib_input, args.calib_cache, args.calib_num_images,
                          args.calib_batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    parser.add_argument("-p", "--precision", default="fp16", choices=["fp32", "fp16", "int8"],
                        help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    parser.add_argument("-w", "--workspace", default=1, type=int, help="The max memory workspace size to allow in Gb, "
                                                                       "default: 1")
    parser.add_argument("--calib_input", help="The directory holding images to use for calibration")
    parser.add_argument("--calib_cache", default="./calibration.cache",
                        help="The file path for INT8 calibration cache to use, default: ./calibration.cache")
    parser.add_argument("--calib_num_images", default=5000, type=int,
                        help="The maximum number of images to use for calibration, default: 5000")
    parser.add_argument("--calib_batch_size", default=8, type=int,
                        help="The batch size for the calibration process, default: 8")
    parser.add_argument("--end2end", default=False, action="store_true",
                        help="export the engine include nms plugin, default: False")
    parser.add_argument("--conf_thres", default=0.4, type=float,
                        help="The conf threshold for the nms, default: 0.4")
    parser.add_argument("--iou_thres", default=0.5, type=float,
                        help="The iou threshold for the nms, default: 0.5")
    parser.add_argument("--max_det", default=100, type=int,
                        help="The total num for results, default: 100")
    parser.add_argument("--v8", default=False, action="store_true",
                        help="use yolov8 model, default: False")
    args = parser.parse_args()
    print(args)
    if not all([args.onnx, args.engine]):
        parser.print_help()
        log.error("These arguments are required: --onnx and --engine")
        sys.exit(1)
    if args.precision == "int8" and not (args.calib_input or os.path.exists(args.calib_cache)):
        parser.print_help()
        log.error("When building in int8 precision, --calib_input or an existing --calib_cache file is required")
        sys.exit(1)
    
    main(args)

