#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <iostream>
#include <memory>
#include <vector>
#include <torch/torch.h>
#include <algorithm> 
#include <torch/script.h>

using namespace cv;
using namespace dnn;
using namespace std;





torch::Tensor CVtoTensor(cv::Mat img,int unsqueeze_dim = 0) {
    cv::resize(img, img, cv::Size{ img.rows, img.cols }, 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    auto img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte);
    img_tensor = img_tensor.permute({ 2, 0, 1 }).toType(torch::kFloat).div_(255);
    std::cout << "tensor shape: " << img_tensor.sizes() << std::endl;
    img_tensor.unsqueeze_(unsqueeze_dim);
    std::cout << "tensors new shape: " << img_tensor.sizes() << std::endl;
    return img_tensor;
}




int main() {
  
    
    
        


     
    string filename = "C:/Users/murat/source/repos/torch_model/torch_model/scriptmodule2.pt";
    auto model = torch::jit::load(filename);
    cout << "Model good to go !!!!" << endl;
    vector<String> fn;
    vector<int>hasta;
    vector<int>saglikli;
    glob("C:/Users/murat/OneDrive/Masaüstü/test/0/*.bmp", fn);
    for (auto f : fn) {
        Mat img = imread(f);
        int down_width = 224;
        int down_height = 224;
        Mat img1;
        resize(img, img1, Size(down_width, down_height), INTER_LINEAR);
        at::Tensor tensor1 = CVtoTensor(img1);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor1);
        model.eval();
        auto output = model.forward(inputs);
        cout << output << endl;
    }
	return 0;
 }
