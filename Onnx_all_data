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



using namespace cv;
using namespace dnn;
using namespace std;


int main(int argc, char** argv)
{
   

    
    //Onnx----------------------------------
    String onnxFile = "C:/Users/murat/source/repos/papsmear/papsmear/resnet50.onnx";
         
    auto model =readNetFromONNX(onnxFile);
    cout << "onnx model okundu..." << endl;


    
    vector<String> fn;
    vector<int>hasta;
    vector<int>saglikli;
    //vector<String> fn;
    glob("C:/Users/murat/OneDrive/Masaüstü/preproccessed_test/images/*.bmp", fn);
    for (auto f : fn){
        Mat img1 = imread(f);
        Mat blob1 = blobFromImage(img1, (1/255), Size(224, 224));
        model.setInput(blob1);
        vector <double> outputs = model.forward();
        cout << outputs << endl;
        //finding a class----------------------------------
        int maxElementIndex = max_element(outputs.begin(), outputs.end()) - outputs.begin();
        //cout << "maxElementIndex:" << maxElementIndex << '\n';
        if (maxElementIndex == 1) {
            hasta.push_back(maxElementIndex);
        }
        else {
            saglikli.push_back(maxElementIndex);
        }
    }
    cout << " saglikli : " << size(saglikli) << endl;;
    cout << "hasta : " << size(hasta) << endl;
    
    return 0;
}
