
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
void train() 
{
	HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);//HOG检测器，用来计算HOG描述子的，其描述子维度为3780

	int DescriptorDim = 0;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定

	Ptr<SVM> svm = SVM::create();//SVM分类器

	//   svm参数设置
	svm->setType(ml::SVM::C_SVC);   //设置svm的类型，即训练数据是非线性分离的
	svm->setKernel(SVM::LINEAR);      //设置核函数的类型
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 50000, FLT_EPSILON));   //设置迭代终止准则，其中分别为准则类型，最大迭代次数，目标精度
	
	if (TRAIN)
	{
		string ImgName;
		ifstream finPos(PosSamListFile);//正样本图片的文件名列表
		ifstream finNeg(NegSamListFile);//负样本图片的文件名列表

		Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数（即正样本，负样本，难例样本之和），列数等于HOG描述子维数  即样本个数*3780的矩阵
		Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人  ，即样本个数*1的矩阵


		//依次读取正样本图片，生成HOG描述子
		for (int num = 0; num < PosSamNO && getline(finPos, ImgName); num++)
		{
			cout << "处理：" << ImgName << endl;
			
			ImgName = "dataset/pos/" + ImgName;//加上正样本的路径名
			Mat src = imread(ImgName);
			if (CENTRAL_CROP)
				if (src.cols >=96 && src.rows >= 160)
					src = src(Rect(16, 16, 64, 128));//将96*160的INRIA正样本图片剪裁为64*128，即剪去上下左右各16个像素
					//  resize(src,src,Size(64,128));

			vector<float> descriptors;//HOG描述子向量
			hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)
			//cout<<"描述子维数："<<descriptors.size()<<endl;
			cout << descriptors.size() << endl;
			//处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
			if (0 == num)
			{
				DescriptorDim = descriptors.size();//HOG描述子的维数
				 //初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
				sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
				//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
				sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32SC1);
			}

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i < DescriptorDim; i++)
				sampleFeatureMat.at<float>(num, i) = descriptors[i];//第num个样本的特征向量中的第i个元素
			sampleLabelMat.at<float>(num, 0) = 1;//正样本类别为1，有人
			 //CV_MAT_ELEM(sampleFeatureMat,float,num,0)=1;
		}

		//依次读取负样本图片，生成HOG描述子
		for (int num = 0; num < NegSamNO && getline(finNeg, ImgName); num++)
		{
			cout << "处理：" << ImgName << endl;
		
			ImgName = "dataset/neg/" + ImgName;
			Mat src = imread(ImgName);//读取图片
			
			vector<float> descriptors;//HOG描述子向量
			hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)
			cout << "描述子维数：" << descriptors.size() << endl;

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int i = 0; i < DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
			sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;//负样本类别为-1，无人

		}

		//处理HardExample负样本
		if (HardExampleNO > 0)
		{
			ifstream finHardExample(HardExampleListFile);
			 //依次读取HardExample负样本图片，生成HOG描述子
			for (int num = 0; num < HardExampleNO && getline(finHardExample, ImgName); num++)
			{
				cout << "处理：" << ImgName << endl;
				
				ImgName = "dataset/HardExample/" + ImgName;//加上HardExample负样本的路径名
				Mat src = imread(ImgName);
				

				vector<float> descriptors;//HOG描述子向量
				hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)
				cout << "描述子维数：" << descriptors.size() << endl;

				//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
				for (int i = 0; i < DescriptorDim; i++)
					sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
				sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;//负样本类别为-1，无人
			}
		}

		//输出样本的HOG特征向量矩阵到文件
		ofstream fout("SampleFeatureMat.txt");
		for(int i=0; i<PosSamNO+NegSamNO; i++)
		{
		fout<<i<<endl;
		for(int j=0; j<DescriptorDim; j++)
		{	fout<<sampleFeatureMat.at<float>(i,j)<<"  ";
		}
		fout<<endl;
		}

		cout << "开始训练SVM分类器" << endl;
		svm->train(sampleFeatureMat, ROW_SAMPLE, sampleLabelMat);//训练分类器
		cout << "训练完成" << endl;
		svm->save("SVM_HOG.xml");//将训练好的SVM模型保存为xml文件

	}
	
	DescriptorDim = svm->getVarCount();//特征向量的维数，即HOG描述子的维数
	cout << "描述子的维数" << DescriptorDim << endl;

	cv::Mat svecsmat = svm->getSupportVectors();  //获取svecsmat，元素类型为float
	int svdim = svm->getVarCount();
	int numofsv = svecsmat.rows;

	
	cv::Mat alphamat = cv::Mat::zeros(numofsv, svdim, CV_32F);
	cv::Mat svindex = cv::Mat::zeros(1, numofsv, CV_64F);

	cv::Mat Result;
	double rho = svm->getDecisionFunction(0, alphamat, svindex);
	//将alphamat元素的数据类型重新转成CV_32F
	alphamat.convertTo(alphamat, CV_32F);
	Result = -1 * alphamat * svecsmat;

	std::vector<float> vec;
	for (int i = 0; i < svdim; ++i)
	{
		vec.push_back(Result.at<float>(0, i));
	}
	vec.push_back(rho);

	//saving HOGDetectorForOpenCV.txt
	std::ofstream fout("HOGDetectorForOpenCV.txt");
	for (int i = 0; i < vec.size(); ++i)
	{
		fout << vec[i] << std::endl;
	}
	cout << vec.size() << endl;
}