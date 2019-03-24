#include"dataset.h"
#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
#include<opencv2/video/background_segm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>



using namespace std;
using namespace cv;
int HardExampleCount = 0;
template <class Type>
Type stringToNum(const string& str)
{
	istringstream iss(str);
	Type num;
	iss >> num;
	return num;
}

void save_hardexample()
{
	
	


	HOGDescriptor myHOG;
	string str;
	vector<float> detector;
	ifstream fin("HOGDetectorForOpenCV.txt");
	while (getline(fin, str))
	{
		detector.push_back(stringToNum<float>(str));
	}

	myHOG.setSVMDetector(detector);

	string ImgName;
	
	char saveName[256];//找出来的HardExample图片文件名
	ifstream in("INRIANegativeImageList.txt");//打开原始负样本图片文件列表

	Mat src;
	while (getline(in, ImgName))
	{
		cout << "处理：" << ImgName << endl;
		ImgName = "INRIAPerson/Train/neg/" + ImgName;
		src = imread(ImgName, 1);//读取图片

		vector<Rect> found, found_filtered;
		//double t = (double)getTickCount();
		
		myHOG.detectMultiScale(src, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
		//t = (double)getTickCount() - t;
		
		//消除重合的矩形框
		size_t i, j;                    
		for (i = 0; i < found.size(); i++)   
		{
			Rect r = found[i];
			for (j = 0; j < found.size(); j++)
				if (j != i && (r & found[j]) == r)
					break;
			if (j == found.size())
				found_filtered.push_back(r);
		}
		       //对消除重合的框进行尺度判断，改变其尺寸为64*128，并保存到dataset/HardExample/文件中
		for (i = 0; i < found_filtered.size(); i++)
		{
			Rect r = found_filtered[i];
			if (r.x < 0)
				r.x = 0;
			if (r.y < 0)
				r.y = 0;
			if (r.x + r.width > src.cols)
				r.width = src.cols - r.x;
			if (r.y + r.height > src.rows)
				r.height = src.rows - r.y;
			Mat imgROI = src(Rect(r.x, r.y, r.width, r.height));
			resize(imgROI, imgROI, Size(64, 128));    //图像缩放
			sprintf_s(saveName, "dataset/HardExample/hardexample%06d.jpg", ++HardExampleCount);   
			imwrite(saveName, imgROI);
			
		}
	}

	cout << "HardExampleCount: " << HardExampleCount << endl;

	
}








