#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.2

using namespace nvinfer1;
static Logger gLogger;

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void generate_yolo_proposals(float* feat_blob, int output_size, float prob_threshold, std::vector<Object>& objects)
{
    const int num_class = 80;
    auto dets = output_size / (num_class + 5);
    for (int boxs_idx = 0; boxs_idx < dets; boxs_idx++)
    {
        const int basic_pos = boxs_idx *(num_class + 5);
        float x_center = feat_blob[basic_pos+0];
        float y_center = feat_blob[basic_pos+1];
        float w = feat_blob[basic_pos+2];
        float h = feat_blob[basic_pos+3];
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;
        float box_objectness = feat_blob[basic_pos+4];
        // std::cout<<*feat_blob<<std::endl;
        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop
    }

}

static void decode_outputs(float* prob, int output_size, std::vector<Object>& objects, float scale, const int img_w, const int img_h) {
        std::vector<Object> proposals;
        generate_yolo_proposals(prob, output_size, BBOX_CONF_THRESH, proposals);
        std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

        qsort_descent_inplace(proposals);

        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, NMS_THRESH);


        int count = picked.size();

        std::cout << "num of boxes: " << count << std::endl;

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x) / scale;
            float y0 = (objects[i].rect.y) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }
}

const float color_list[80][3] =
{
    {0.000, 0.447, 0.741},
    {0.850, 0.325, 0.098},
    {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556},
    {0.466, 0.674, 0.188},
    {0.301, 0.745, 0.933},
    {0.635, 0.078, 0.184},
    {0.300, 0.300, 0.300},
    {0.600, 0.600, 0.600},
    {1.000, 0.000, 0.000},
    {1.000, 0.500, 0.000},
    {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.333, 0.333, 0.000},
    {0.333, 0.667, 0.000},
    {0.333, 1.000, 0.000},
    {0.667, 0.333, 0.000},
    {0.667, 0.667, 0.000},
    {0.667, 1.000, 0.000},
    {1.000, 0.333, 0.000},
    {1.000, 0.667, 0.000},
    {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500},
    {0.000, 0.667, 0.500},
    {0.000, 1.000, 0.500},
    {0.333, 0.000, 0.500},
    {0.333, 0.333, 0.500},
    {0.333, 0.667, 0.500},
    {0.333, 1.000, 0.500},
    {0.667, 0.000, 0.500},
    {0.667, 0.333, 0.500},
    {0.667, 0.667, 0.500},
    {0.667, 1.000, 0.500},
    {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500},
    {1.000, 0.667, 0.500},
    {1.000, 1.000, 0.500},
    {0.000, 0.333, 1.000},
    {0.000, 0.667, 1.000},
    {0.000, 1.000, 1.000},
    {0.333, 0.000, 1.000},
    {0.333, 0.333, 1.000},
    {0.333, 0.667, 1.000},
    {0.333, 1.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000},
    {0.667, 1.000, 1.000},
    {1.000, 0.000, 1.000},
    {1.000, 0.333, 1.000},
    {1.000, 0.667, 1.000},
    {0.333, 0.000, 0.000},
    {0.500, 0.000, 0.000},
    {0.667, 0.000, 0.000},
    {0.833, 0.000, 0.000},
    {1.000, 0.000, 0.000},
    {0.000, 0.167, 0.000},
    {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000},
    {0.000, 0.667, 0.000},
    {0.000, 0.833, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 0.167},
    {0.000, 0.000, 0.333},
    {0.000, 0.000, 0.500},
    {0.000, 0.000, 0.667},
    {0.000, 0.000, 0.833},
    {0.000, 0.000, 1.000},
    {0.000, 0.000, 0.000},
    {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286},
    {0.429, 0.429, 0.429},
    {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714},
    {0.857, 0.857, 0.857},
    {0.000, 0.447, 0.741},
    {0.314, 0.717, 0.741},
    {0.50, 0.5, 0}
};


static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, std::string f)
{
    static const char* class_names[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5){
            txt_color = cv::Scalar(0, 0, 0);
        }else{
            txt_color = cv::Scalar(255, 255, 255);
        }

        cv::rectangle(image, obj.rect, color * 255, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = obj.rect.x;
        int y = obj.rect.y + 1;
        //int y = obj.rect.y - label_size.height - baseLine;
        if (y > image.rows)
            y = image.rows;
        //if (x + label_size.width > image.cols)
            //x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }

    cv::imwrite("det_res.jpg", image);
    fprintf(stderr, "save vis file\n");
    /* cv::imshow("image", image); */
    /* cv::waitKey(0); */
}


class YOLO
{
    public:
        YOLO(std::string engine_file_path);
        virtual ~YOLO();
        void detect_img(std::string image_path);
        void detect_video(std::string video_path);
        cv::Mat static_resize(cv::Mat& img);
        float* blobFromImage(cv::Mat& img);
        void doInference(IExecutionContext& context, float* input, float* output, const int output_size, cv::Size input_shape);

    private:
        static const int INPUT_W = 640;
        static const int INPUT_H = 640;
        const char* INPUT_BLOB_NAME = "images";
        const char* OUTPUT_BLOB_NAME = "output0";
        float* prob;
        int output_size = 1;
        ICudaEngine* engine;
        IRuntime* runtime;
        IExecutionContext* context;

};

YOLO::YOLO(std::string engine_file_path)
{
    size_t size{0};
    char *trtModelStream{nullptr};
    //使用二进制方式读取文件，将其存放到字符数组trtModelStream中。
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
    std::cout << "engine init finished" << std::endl;
    //创建一个 TensorRT 的运行时（runtime）实例，以便在 TensorRT 中执行推理（inference）
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr); //确保 runtime 变量不为空指针 
    //使用 runtime 中的 deserializeCudaEngine 函数，将字符数组中的内容反序列化为一个 ICudaEngine 对象。
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    //创建一个执行上下文（execution context）
    context = engine->createExecutionContext(); 
    assert(context != nullptr);
    delete[] trtModelStream; //释放字符数组的内存空间。
    //engine 的 getBindingDimensions 方法，获取引擎的第 1 个输出张量的维度信息，并将其赋值给变量 out_dims。
    auto out_dims = engine->getBindingDimensions(1);
    //遍历 out_dims 中的每一个维度，并将其相乘，得到输出张量的大小。
    for(int j=0;j<out_dims.nbDims;j++) {
        std::cout << "ouput size: " << j  << " " << out_dims.d[j] << std::endl;
        this->output_size *= out_dims.d[j];
    }
    //创建一个大小为 output_size 的 float 类型的数组，用于存放推理结果。
    this->prob = new float[this->output_size];
}

YOLO::~YOLO()
{
    std::cout<<"yolo destroy"<<std::endl;
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
    
}

void YOLO::detect_img(std::string image_path)
{
    cv::Mat img = cv::imread(image_path);
    int img_w = img.cols;
    int img_h = img.rows;
    cv::Mat pr_img = this->static_resize(img); // resize to 640x640
    cv::imwrite("resize.jpg", pr_img);
    std::cout << "blob image" << std::endl;

    float* blob;
    blob = blobFromImage(pr_img); //将图像转换为浮点数类型的 blob 数据，方便后续将其输入到 TensorRT 模型中进行推理
    float scale = std::min(this->INPUT_W / (img.cols*1.0), this->INPUT_H / (img.rows*1.0)); //计算图像缩放比例

    // run inference
    auto start = std::chrono::system_clock::now();
    doInference(*context, blob, this->prob, output_size, pr_img.size()); //执行推理
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    
    // decode outputs
    std::vector<Object> objects;
    decode_outputs(this->prob, this->output_size, objects, scale, img_w, img_h);
    draw_objects(img, objects, image_path);
    delete blob;

}

cv::Mat YOLO::static_resize(cv::Mat& img) {
    float r = std::min(this->INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(this->INPUT_W, this->INPUT_H, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

float* YOLO::blobFromImage(cv::Mat& img){
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB); //将图像从 BGR 转换为 RGB

    float* blob = new float[img.total()*3]; //创建一个大小为图像像素数乘以 3 的浮点数类型的数组，用于存放图像的 blob 数据。img.total() 函数用于获取图像的像素数。
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f);  //将图像的像素值归一化到 0-1 之间，并赋值给 blob 数组。
            }
        }
    }
    return blob;
}

void YOLO::doInference(IExecutionContext& context, float* input, float* output, const int output_size, cv::Size input_shape) {
    //获取与推断上下文相关联的CUDA引擎。 context对象是IExecutionContext接口的一个实例，它表示用于在CUDA引擎上执行推断的执行上下文的实例。
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);  // 1 input and 1 output
    void* buffers[2]; //创建一个大小为 2 的 void 类型的指针数组，用于存放输入和输出张量的 GPU 缓冲区。

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);//获取了输入张量在TensorRT引擎中的绑定索引
    std::cout << "inputIndex: " << inputIndex << std::endl;
    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT); //检查输入张量的数据类型是否为 float 类型
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);//获取了输出张量在TensorRT引擎中的绑定索引
    std::cout << "outputIndex: " << outputIndex << std::endl;
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT); //检查输出张量的数据类型是否为 float 类型
    int mBatchSize = engine.getMaxBatchSize(); //获取引擎的最大批量大小

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float))); //分配输入张量的 GPU 缓冲区
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float))); //分配输出张量的 GPU 缓冲区

    // Create stream
    //创建一个 CUDA 流，用于异步执行 CUDA 操作, 可以在不等待先前提交的操作完成的情况下，继续提交新的操作。
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    
    /*
    使用CUDA的异步内存拷贝函数cudaMemcpyAsync，将主机上的输入数据input从主机内存复制到设备上的输入缓冲区buffers[inputIndex]中。
    cudaMemcpyAsync函数的第一个参数是目标缓冲区的指针，
    第二个参数是源缓冲区的指针，
    第三个参数是要复制的字节数，
    第四个参数指定了内存拷贝的方向（这里是从主机到设备），
    第五个参数是CUDA流句柄。
    由于使用异步内存拷贝，该函数返回时不会等待数据复制完成，而是立即返回。
    */
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    //使用输入和输出缓冲区以及 CUDA 流在执行上下文中异步执行推理。
    context.enqueue(1, buffers, stream, nullptr);
    //使用异步内存拷贝函数cudaMemcpyAsync，将设备上的输出数据output从设备内存复制到主机上的输出缓冲区buffers[outputIndex]中。
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream); //等待所有的异步操作完成

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}