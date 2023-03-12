#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <iostream>
#include <chrono>
#include <random>

constexpr char kDnnName[] = "generator_dcgan.onnx";
constexpr int kNoiseSize = 100;

// size of generated image
constexpr int kImageHeight = 28;
constexpr int kImageWidth = 28;

cv::Mat generateRandomBlobMat();
void showMat(cv::Mat image);

int main(){
    // load model from ONNX-file
    cv::dnn::Net model = cv::dnn::readNetFromONNX(kDnnName);

    // prepare blob of noise vector
    cv::Mat blob = generateRandomBlobMat();

    // set model input
    model.setInput(blob);

    // get model results
    cv::Mat output = model.forward();

    // get pointer to raw data of resulted image
    float* ptr = output.ptr<float>();

    // create mat from raw pointer
    cv::Mat generatedImage = cv::Mat(kImageHeight, kImageWidth, CV_32FC1, ptr);
    showMat(generatedImage);

    return 0;
}

cv::Mat generateRandomBlobMat(){
    // random values generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.f, 1.f);// between 0.0 and 1.0

    int size[4] = {1, kNoiseSize, 1, 1};
    cv::Mat blob(4, size, CV_32FC1, cv::Scalar(0));// 4-dimensions float 32 type

    // get pointer to raw data
    float* blobValues = blob.ptr<float>();

    for(size_t i = 0; i < kNoiseSize; ++i){
        // set random data to noise vector
        blobValues[i] = distrib(gen);
    }

    return blob;
}

void showMat(cv::Mat image){
    // resize image to larger size
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(200, 200), cv::INTER_LINEAR);

    // show image in window
    cv::imshow("generation result", resizedImage);
    cv::waitKey(0);// wait for key press
    cv::destroyAllWindows();
}
