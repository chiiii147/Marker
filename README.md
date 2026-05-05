# Marker


## Convert ONNX model to TensorRT model
 
main()
 ├── initialize params
 ├── build()
 │    ├── create builder
 │    ├── create network
 │    ├── create config
 │    ├── create parser
 │    ├── parse mnist.onnx
 │    ├── optimize + compile
 │    ├── serialized engine plan
 │    ├── deserialize engine
 │    └── get input/output dims
 │
 └── infer()
      ├── allocate host/device buffers
      ├── create execution context
      ├── read random pgm input
      ├── copy input to GPU
      ├── executeV2()
      ├── copy output to CPU
      └── softmax + verify


| Nhóm | Tên Layer / Operator | Khái niệm / Định nghĩa | Ví dụ |
|------|----------------------|-------------------------|-------|
| Feature Extraction | Convolution (Conv) | Dùng kernel quét qua dữ liệu đầu vào để trích xuất đặc trưng như biên, góc, texture | phát hiện cạnh vật thể |
| Feature Extraction | Depthwise Convolution | Thực hiện convolution riêng trên từng channel để giảm số phép tính | MobileNet |
| Feature Extraction | Group Convolution | Chia channel thành nhiều nhóm và tích chập độc lập | ResNeXt |
| Feature Extraction | Dilated Convolution | Convolution có khoảng cách kernel giãn ra để tăng receptive field | segmentation |
| Feature Extraction | Deconvolution / TransposedConv | Tích chập ngược dùng để tăng kích thước feature map | super resolution |
| Activation | ReLU | Hàm kích hoạt đưa giá trị âm về 0, giữ nguyên giá trị dương | [-2,3] → [0,3] |
| Activation | LeakyReLU | ReLU nhưng vẫn giữ một phần nhỏ giá trị âm | -2 → -0.02 |
| Activation | Sigmoid | Ép đầu ra về khoảng 0 đến 1 | xác suất nhị phân |
| Activation | Tanh | Ép đầu ra về khoảng -1 đến 1 | RNN |
| Activation | Mish / Swish / SiLU | Hàm kích hoạt phi tuyến mượt hơn ReLU | YOLOv5+ |
| Pooling | MaxPool | Lấy giá trị lớn nhất trong từng vùng | giữ feature nổi bật |
| Pooling | AvgPool | Lấy giá trị trung bình trong vùng | làm mượt |
| Pooling | GlobalAvgPool | Lấy trung bình toàn bộ feature map thành vector nhỏ | classifier cuối |
| Pooling | AdaptivePool | Pooling để thu về kích thước mong muốn | output 1x1 |
| Normalization | BatchNormalization | Chuẩn hóa dữ liệu theo mean và variance của batch | Conv→BN→ReLU |
| Normalization | LayerNormalization | Chuẩn hóa theo từng sample/feature | Transformer |
| Normalization | InstanceNormalization | Chuẩn hóa riêng cho từng ảnh | style transfer |
| Normalization | GroupNormalization | Chuẩn hóa theo nhóm channel | batch nhỏ |
| Fully Connected | Fully Connected (FC) | Lớp kết nối đầy đủ, mọi neuron nối với lớp sau | classifier |
| Fully Connected | Gemm | Phép nhân ma trận tổng quát dùng trong FC | Y = A×B + C |
| Fully Connected | MatMul | Phép nhân ma trận tensor | attention |
| Fully Connected | Linear | Tên khác của fully connected trong PyTorch | nn.Linear |
| Tensor Manipulation | Reshape | Thay đổi hình dạng tensor nhưng giữ dữ liệu | (1,28,28)→(1,784) |
| Tensor Manipulation | Flatten | Trải tensor nhiều chiều thành vector 1 chiều | trước FC |
| Tensor Manipulation | Squeeze | Loại bỏ chiều có kích thước bằng 1 | (1,1,10)→(10) |
| Tensor Manipulation | Unsqueeze | Thêm chiều mới vào tensor | (10)→(1,10) |
| Tensor Manipulation | Transpose | Hoán đổi thứ tự các trục tensor | NHWC↔NCHW |
| Tensor Manipulation | Permute | Đổi vị trí trục tensor tổng quát | ViT |
| Tensor Manipulation | Expand | Mở rộng tensor bằng broadcast | nhân bản shape |
| Tensor Manipulation | Tile | Lặp tensor nhiều lần | copy feature |
| Elementwise Math | Add | Cộng từng phần tử tensor | residual add |
| Elementwise Math | Sub | Trừ từng phần tử tensor | error calc |
| Elementwise Math | Mul | Nhân từng phần tử tensor | attention weight |
| Elementwise Math | Div | Chia từng phần tử tensor | normalization |
| Elementwise Math | Pow | Lũy thừa tensor | custom op |
| Elementwise Math | Exp | Hàm số mũ e^x | softmax |
| Elementwise Math | Log | Hàm logarit tự nhiên | log loss |
| Elementwise Math | Sqrt | Căn bậc hai tensor | norm |
| Elementwise Math | Abs | Giá trị tuyệt đối | L1 |
| Merge/Split | Concat | Nối nhiều tensor lại với nhau | YOLO fusion |
| Merge/Split | Split | Tách tensor thành nhiều phần | multi head |
| Merge/Split | Slice | Cắt một phần tensor | crop tensor |
| Merge/Split | Gather | Lấy dữ liệu theo chỉ số index | token select |
| Merge/Split | Scatter | Ghi dữ liệu vào index xác định | custom assign |
| Reduction | ReduceMean | Lấy trung bình toàn tensor/theo chiều | mean feature |
| Reduction | ReduceSum | Cộng các phần tử | total sum |
| Reduction | ReduceMax | Lấy giá trị lớn nhất | strongest feature |
| Reduction | ArgMax | Lấy vị trí phần tử lớn nhất | class id |
| Reduction | TopK | Lấy K giá trị lớn nhất | top5 classes |
| Output Activation | Softmax | Chuyển score thành phân phối xác suất tổng =1 | digit classify |
| Output Activation | LogSoftmax | Log của softmax | NLL loss |
| Output Activation | Sigmoid Output | Xác suất độc lập từng lớp | multilabel |
| Upsampling | Upsample | Phóng to kích thước tensor | 20x20→40x40 |
| Upsampling | Resize | Thay đổi kích thước theo nội suy | bilinear |
| Upsampling | Interpolate | Nội suy dữ liệu tensor | segmentation |
| Residual | Identity | Truyền thẳng input không đổi | shortcut |
| Residual | Skip Add | Cộng nhánh tắt với nhánh chính | ResNet |
| Detection Postprocess | Non-Maximum Suppression (NMS) | Loại bỏ các bounding box trùng lặp | object detection |
| Detection Postprocess | DecodeBBox | Giải mã tọa độ anchor box | YOLO/SSD |
| Detection Postprocess | PriorBox | Sinh anchor box chuẩn | SSD |
| Transformer/NLP | Embedding | Chuyển token thành vector số | NLP input |
| Transformer/NLP | Positional Encoding | Thêm thông tin vị trí chuỗi | Transformer |
| Transformer/NLP | Attention | Tính mức độ liên quan giữa các token | GPT/BERT |
| Transformer/NLP | MultiHead Attention | Nhiều attention song song | LLM |
| Transformer/NLP | Mask | Che token không cần nhìn | decoder |
| Control Logic | Shape | Lấy kích thước tensor | dynamic model |
| Control Logic | Constant | Tensor hằng số | fixed weight |
| Control Logic | Cast | Đổi kiểu dữ liệu tensor | FP32→FP16 |
| Control Logic | Where | Chọn dữ liệu theo điều kiện | if-else tensor |
| Control Logic | Equal/Greater/Less | So sánh tensor | threshold |
| Custom Operator | Plugin | Layer TensorRT không hỗ trợ sẵn, cần custom CUDA implementation | YOLO decode |
| Custom Operator | CustomOp | Operator framework tự định nghĩa | research model |