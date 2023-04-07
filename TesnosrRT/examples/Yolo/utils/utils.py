import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import matplotlib.pyplot as plt

class BaseEngine(object):
    def __init__(self, engine_path):
        self.mean = None
        self.std = None
        self.n_classes = 80
        self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

        logger = trt.Logger(trt.Logger.WARNING)
        #定义Logger 的初始最低严重性级别。默认为 WARNING。可选的严重性级别为：VERBOSE、INFO、WARNING、ERROR 和
        logger.min_severity = trt.Logger.Severity.ERROR
        #创建一个运行时
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        '''
        将一个序列化的TensorRT引擎文件（serialized_engine）作为输入，通过调用运行时对象的deserialize_cuda_engine方法，
        将引擎文件反序列化为一个CUDA引擎。反序列化是将二进制数据转换为TensorRT引擎对象的过程。
        引擎文件通常包含模型的计算图结构、参数和配置信息。
        '''
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # 获取engine第0个节点的shape(NCHW) [2:]指代获取 HW
        #使用TensorRT引擎对象（engine）创建一个执行上下文对象（context）其作用是为了执行深度学习模型的推理计算
        #执行上下文包含了TensorRT引擎中所有的输入和输出张量，以及中间计算结果的内存缓冲区
        self.context = engine.create_execution_context()
        '''
        在这段代码中，我们定义了三个空列表，以便在执行推理计算时存储输入和输出数据。
        具体来说，self.inputs列表将存储所有输入数据的numpy数组或CUDA内存指针；
        self.outputs列表将存储所有输出数据的numpy数组或CUDA内存指针；
        self.bindings列表将存储TensorRT引擎中所有绑定张量的CUDA内存缓冲区的指针。
        
        在TensorRT中，引擎对象包含输入和输出张量的元数据信息，例如名称、数据类型、形状等等。
        我们可以使用引擎对象的get_binding_shape和get_binding_dtype方法来获取张量的形状和数据类型。
        然后，我们可以使用numpy数组或CUDA内存指针来存储和传递输入和输出数据。
        '''
        self.inputs, self.outputs, self.bindings = [], [], []
        #使用CUDA Python API中的cuda.Stream()函数来创建一个CUDA流对象，并将其赋值给类的属性self.stream。
        # 流对象可以用于在GPU上异步地执行TensorRT推理计算，并在必要时与主机端的代码进行同步。
        self.stream = cuda.Stream()
        
        '''
        在这段代码中，我们遍历了TensorRT引擎对象中的每个绑定，
        使用trt.volume函数计算了每个绑定张量的大小（size）。
        使用trt.nptype函数获取每个绑定的数据类型（dtype），
        使用cuda.pagelocked_empty函数在主机端内存中分配了一个相应大小和类型的numpy数组（host_mem）。
        使用cuda.mem_alloc函数在GPU设备内存中分配了一个相应大小的内存缓冲区（device_mem）。
        将所有绑定张量的CUDA内存缓冲区的指针存储到类的属性self.bindings中。
        如果当前绑定是输入张量，则将主机端的numpy数组和GPU设备的内存缓冲区的指针存储到self.inputs列表中；
        如果当前绑定是输出张量，则将它们存储到self.outputs列表中。
        '''
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


    def infer(self, img):
        #将名为 "img" 的 Numpy 数组（即输入数据）转化为一维数组（使用np.ravel()函数），并将其赋值给 TensorRT 引擎的第一个输入张量的 "host" 属性
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu 将 TensorRT 引擎的输入数据从主机内存（host）异步地复制到设备内存（device）
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference 执行异步推理计算
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu 将 TensorRT 引擎的输出结果从设备内存（device）异步地复制到主机内存（host）
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream 等待 CUDA 流中的所有异步任务完成
        self.stream.synchronize()
        #将TensorRT 推理的输出结果从设备内存复制到主机内存，并将其存储在一个列表中
        data = [out['host'] for out in self.outputs]
        return data

    def detect_video(self, video_path, conf=0.5, end2end=False):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('results.avi',fourcc,fps,(width,height))
        fps = 0
        import time
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            blob, ratio = preproc(frame, self.imgsz, self.mean, self.std)
            t1 = time.time()
            data = self.infer(blob)
            fps = (fps + (1. / (time.time() - t1))) / 2
            frame = cv2.putText(frame, "FPS:%d " %fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
            if end2end:
                num, final_boxes, final_scores, final_cls_inds = data
                final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
                dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
            else:
                predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
                dets = self.postprocess(predictions,ratio)

            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:,
                                                                :4], dets[:, 4], dets[:, 5]
                frame = vis(frame, final_boxes, final_scores, final_cls_inds,
                                conf=conf, class_names=self.class_names)
            cv2.imshow('frame', frame)
            out.write(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()

    def inference(self, img_path, conf=0.5, end2end=False):
        origin_img = cv2.imread(img_path)
        img, ratio = preproc(origin_img, self.imgsz, self.mean, self.std)
        data = self.infer(img)
        if end2end:
            num, final_boxes, final_scores, final_cls_inds = data
            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        else:
            predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
            print(predictions.shape)
            dets = self.postprocess(predictions,ratio)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:,
                                                             :4], dets[:, 4], dets[:, 5]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=conf, class_names=self.class_names)
        return origin_img

    @staticmethod
    def postprocess(predictions, ratio):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        return dets

    def get_fps(self):
        import time
        img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32) 
        #使用numpy的ascontiguousarray函数将numpy数组转换为连续的内存块，以便在后续的计算中能够更快地访问和处理。
        #此外，我们还将数据类型转换为np.float32，以便在后续的计算中使用浮点数进行计算。
        for _ in range(5):  # warmup
            _ = self.infer(img)

        t0 = time.perf_counter()
        for _ in range(100):  # calculate average time
            _ = self.infer(img)
        print(100/(time.perf_counter() - t0), 'FPS')


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3: #判断图像的维度
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # if use yolox set
    # padded_img = padded_img[:, :, ::-1]
    # padded_img /= 255.0
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def rainbow_fill(size=50):  # simpler way to generate rainbow color
    cmap = plt.get_cmap('jet')
    color_list = []

    for n in range(size):
        color = cmap(n/size)
        color_list.append(color[:3])  # might need rounding? (round(x, 3) for x in color)[:3]

    return np.array(color_list)


_COLORS = rainbow_fill(80).astype(np.float32).reshape(-1, 3)


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img