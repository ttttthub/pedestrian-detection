#include<opencv2/opencv.hpp>
#include<opencv2/video/background_segm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include<iostream>
#include<string>
#include"dataset.h"
using namespace cv;
using namespace std;
using namespace cv::ml;
template <class Type>
Type stringToNum(const string& str)
{
	istringstream iss(str);
	Type num;
	iss >> num;
	return num;
}
void display(Mat, vector<Rect>&);
void Crop_picture();
void train();
void save_hard_example();
void pic_test();
void video_test();
int main()
{
	


	////////////////////////训练
	//Crop_picture();     //裁切负样本图片，每张负样本图片随机裁成10张
	//train();          //训练正负样本
	//save_hardexample()     //根据正负样本得到的检测子，对INRIAPerson/Train/neg/中的图片进行测试，并保存难例样本
	//train();               //根据正负样本及难例重新训练

	//pic_test();    //对图片进行行人检测
	video_test();       //对视频进行行人检测
	system("pause");
	return 0;
}