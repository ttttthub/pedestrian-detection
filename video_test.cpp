#include<opencv2/opencv.hpp>
#include<opencv2/video/background_segm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include<iostream>
#include<string>
#include"dataset.h"

using namespace std;
using namespace cv;

#define VideotestPath "2.avi"
void video_test() {
	void display(Mat, vector<Rect>&);            
	//void Crop_picture();     
	//void train();
	//void save_hard_example();
	
	//Crop_picture();     //裁切负样本图片，每张负样本图片随机裁成10张
	//train();          //训练正负样本
	//save_hardexample()     //根据正负样本得到的检测子，对INRIAPerson/Train/neg/中的图片进行测试，并将错检的样本保存
	//train();      //训练正负样本及难例样本
	

	//加载svm分类器的系数
	HOGDescriptor hog; string str;
	vector<float> detector;
	/*ifstream fin("HOGDetectorForOpenCV.txt");
	while (getline(fin, str))
	{
		detector.push_back(stringToNum<float>(str));
	}
*/

	vector<Rect> people;
	VideoCapture capture(VideotestPath);
	/*if (!capture.isOpened())
	return -1;*/
	Mat frame, foreground;
	

	int num = 0; 
	Ptr<BackgroundSubtractorMOG2> mod = createBackgroundSubtractorMOG2();

	while (true)
	{
		vector<Rect> rect6;
		if (!capture.read(frame))
		break;
		mod->apply(frame, foreground, 0.01);
		hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
		//hog.setSVMDetector(detector);
		vector<Rect> rect5;
		display(foreground, rect5);
		vector<Rect> ret = rect5;
		for (auto i = 0; i != ret.size(); i++)
		{
			Mat a = frame;
		

			if (ret[i].x > 50 && ret[i].y > 50 && ret[i].x + ret[i].width <670 && ret[i].y + ret[i].height < 520)
			{
				ret[i].x = ret[i].x - 50;
				ret[i].y = ret[i].y - 50; ret[i].width = ret[i].width + 100; ret[i].height = ret[i].height + 100;
			}
			Mat src(a(ret[i]));
			cout << ret[i].x << " " << ret[i].y << " " << ret[i].width << " " << ret[i].height << endl;
			// imshow("aa", src); waitKey(0);
			// cv::namedWindow("src", CV_WINDOW_NORMAL); 



			if (ret[i].width >= 64 && ret[i].height >= 128)

				hog.detectMultiScale(src, people, 0, Size(4, 4), Size(0, 0), 1.07, 2);
			//cout << people.size()<<endl;
			for (size_t j = 0; j < people.size(); j++)
			{
				people[j].x += ret[i].x; people[j].y += ret[i].y;
				rect6.push_back(people[j]);
				//rectangle(frame, people[j], cv::Scalar(0, 0, 255), 2);
			}
			//imshow(" ", frame); waitKey(0);

		}
		//////因为多尺度检测得到的结果矩形框较大，按比例缩减矩形框
		for (auto h = 0; h != rect6.size(); h++)
		{
			rect6[h].x += cvRound(rect6[h].width*0.1);
			rect6[h].width = cvRound(rect6[h].width*0.8);
			rect6[h].y += cvRound(rect6[h].height*0.07);
			rect6[h].height = cvRound(rect6[h].height*0.8);
			rectangle(frame, rect6[h], cv::Scalar(0, 0, 255), 1);
			//rect2[h] = boundingRect(frame);
		}
		imshow(" ", frame); waitKey(1);
	}
	waitKey();
}