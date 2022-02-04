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
int main() {
string filename = "C:/Users/murat/source/repos/torch_model/torch_model/scriptmodule2.pt";
	auto model =torch::jit::load(filename);
   cout << "Model good to go !!!!"<<endl;
    Mat img = imread("C:/Users/murat/OneDrive/Masaüstü/image.bmp");
    int down_width = 224;
    int down_height = 224;
    Mat img1;
    resize(img, img1, Size(down_width, down_height), INTER_LINEAR);
    vector<Mat> various_images;
    various_images.push_back(resized_down);
    
    at::Tensor tensor1 = CVtoTensor(img1);
   
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor1);
    model.eval();
    auto output = model.forward(inputs).toTensor();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  return 0;
 }
