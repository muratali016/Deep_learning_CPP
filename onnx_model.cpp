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

//Some libraries not used


using namespace cv;
using namespace dnn;
using namespace std;


int main(int argc, char** argv)
{
    

    //Onnx----------------------------------
    String onnxFile = "..../resnet50.onnx"; // onnx path--------
    
    cout << "onnx okundu !"; 

    auto model =readNetFromONNX( onnxFile);
    cout << "onnx model okundu..." << endl;

    Mat img = imread("..../image.bmp"); // image path--------
   
    Mat blob = blobFromImage(img, 0.0039215686274509803921568627451, Size(224, 224));
   
    model.setInput(blob);

    //using model    ----------------------------------
    vector <double> outputs1 = model.forward();
    cout << outputs1 << endl;
    //finding a class----------------------------------
    int maxElementIndex =max_element(outputs1.begin(), outputs1.end()) - outputs1.begin();
    cout << "maxElementIndex:" << maxElementIndex << '\n';
   
    string text = to_string(maxElementIndex);
    
    
   putText(img,text, Point(25, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0), 2);
   imshow("img", img);
   waitKey(0);
    
    return 0;
}
