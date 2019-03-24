#include<opencv2/opencv.hpp>
#include<opencv2/video/background_segm.hpp>
#include<iostream>
#include<vector>
#include"dataset.h"
using namespace std;
using namespace cv;

void display(Mat gray_diff, vector<Rect>& rect)
{
	//Mat res = src.clone();
	vector<vector<Point>> cts;  //定义轮廓数组
	findContours(gray_diff, cts, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); //查找轮廓,，模式为只检测外轮廓，并存储所有的轮廓点
																		  //vector<Rect> rect; //定义矩形边框
	for (int i = 0; i < cts.size(); i++)
	{
		if (contourArea(cts[i])>th_area)       //计算轮廓的面积，排除小的干扰轮廓
			
			  //查找外部矩形边界  
			rect.push_back(boundingRect(cts[i]));   //计算轮廓的垂直边界最小矩形

	}
	cout << rect.size() << endl;     //输出轮廓个数
}