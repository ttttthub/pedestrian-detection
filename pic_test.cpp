#include<opencv2/opencv.hpp>
#include<opencv2/video/background_segm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include<iostream>
#include<string>
#include<vector>
#include"dataset.h"

#define PictestPath  "Test.jpg"


using namespace std;
using namespace cv;
template <class Type>
Type stringToNum(const string& str)
{
	istringstream iss(str);
	Type num;
	iss >> num;
	return num;
}
void pic_test() {



	string str;
	vector<float>  detector;
	ifstream fin("HOGDetectorForOpenCV.txt");  
	
	while (getline(fin, str))
	{
	detector.push_back(stringToNum<float>(str));
	}
	
	Mat src = imread(PictestPath);
	Mat out = src;
	HOGDescriptor hog;
	
	hog.setSVMDetector(detector);
	//hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	
	vector<Rect>  people;
	double start = (double)getTickCount();
	hog.detectMultiScale(src, people, 0, Size(4, 4), Size(0, 0), 1.05, 2);
	double t = ((double)getTickCount() - start) / getTickFrequency();
	for (auto i = 0; i != people.size(); i++)
	{
		rectangle(src, people[i], cv::Scalar(0, 0, 255), 1);
	}
	
	cout << "时间: " << t * 1000 << "ms" << endl;     //多尺度检测的时间
	imshow("yuantu ", src); 
	waitKey(0);










}
