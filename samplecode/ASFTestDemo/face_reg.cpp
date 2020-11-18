#include "arcsoft_face_sdk.h"
#include "amcomdef.h"
#include "asvloffscreen.h"
#include "merror.h"
#include <iostream>  
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<time.h>

using namespace cv;
using namespace std;

//从开发者中心获取APPID/SDKKEY(以下均为假数据，请替换)
#define APPID "6FdXUSDAUCAJi5KiF23fNKvU6V4idckFyLLvnw4idE66"
#define SDKKEY "EyWVcTzti9EwPAWVv43GoW9aM6anPMYLFpvEvKsAnGLd"

#define NSCALE 16 
#define FACENUM	5

#define SafeFree(p) { if ((p)) free(p); (p) = NULL; }
#define SafeArrayDelete(p) { if ((p)) delete [] (p); (p) = NULL; } 
#define SafeDelete(p) { if ((p)) delete (p); (p) = NULL; } 

struct RESULT
{
	int flag=-1; // 1:正常，-1:人脸检测或人脸特征提取错误
	/*
	10000000 默认值，不代表任何错误
	-1: 人脸数不为1
	-2: 图片读取失败
	-3: 图片尺度过小
	*/
	long error_code=10000000; // 
};

//图像颜色格式转换
int ColorSpaceConversion(MInt32 width, MInt32 height, MInt32 format, MUInt8* imgData, ASVLOFFSCREEN& offscreen)
{
	offscreen.u32PixelArrayFormat = (unsigned int)format;
	offscreen.i32Width = width;
	offscreen.i32Height = height;
	
	switch (offscreen.u32PixelArrayFormat)
	{
	case ASVL_PAF_RGB24_B8G8R8:
		offscreen.pi32Pitch[0] = offscreen.i32Width * 3;
		offscreen.ppu8Plane[0] = imgData;
		break;
	case ASVL_PAF_I420:
		offscreen.pi32Pitch[0] = width;
		offscreen.pi32Pitch[1] = width >> 1;
		offscreen.pi32Pitch[2] = width >> 1;
		offscreen.ppu8Plane[0] = imgData;
		offscreen.ppu8Plane[1] = offscreen.ppu8Plane[0] + offscreen.i32Height*offscreen.i32Width;
		offscreen.ppu8Plane[2] = offscreen.ppu8Plane[0] + offscreen.i32Height*offscreen.i32Width * 5 / 4;
		break;
	case ASVL_PAF_NV12:
	case ASVL_PAF_NV21:
		offscreen.pi32Pitch[0] = offscreen.i32Width;
		offscreen.pi32Pitch[1] = offscreen.pi32Pitch[0];
		offscreen.ppu8Plane[0] = imgData;
		offscreen.ppu8Plane[1] = offscreen.ppu8Plane[0] + offscreen.pi32Pitch[0] * offscreen.i32Height;
		break;
	case ASVL_PAF_YUYV:
	case ASVL_PAF_DEPTH_U16:
		offscreen.pi32Pitch[0] = offscreen.i32Width * 2;
		offscreen.ppu8Plane[0] = imgData;
		break;
	case ASVL_PAF_GRAY:
		offscreen.pi32Pitch[0] = offscreen.i32Width;
		offscreen.ppu8Plane[0] = imgData;
		break;
	default:
		return 0;
	}
	return 1;
}

void opencvRGB2NV21(Mat Img, unsigned  char* yuvbuff, int &width, int &height){
	int rows = height;
	int cols = width;

	int Yindex = 0;
	int UVindex = rows * cols;
	
	// unsigned char* yuvbuff = new unsigned char[rows * cols*  3 / 2];
 
	cv::Mat NV21(rows+rows/2, cols, CV_8UC1);
	cv::Mat OpencvYUV;
	cv::Mat OpencvImg;
	cv::cvtColor(Img, OpencvYUV, CV_BGR2YUV_YV12);
	
	int UVRow{ 0 };
	for (int i=0;i<rows;i++)
	{
		for (int j=0;j<cols;j++)
		{
			uchar* YPointer = NV21.ptr<uchar>(i);
 
			int B = Img.at<cv::Vec3b>(i, j)[0];
			int G = Img.at<cv::Vec3b>(i, j)[1];
			int R = Img.at<cv::Vec3b>(i, j)[2];
 
			//计算Y的值
			int Y = (77 * R + 150 * G + 29 * B) >> 8;
			YPointer[j] = Y;
			yuvbuff[Yindex++] = (Y < 0) ? 0 : ((Y > 255) ? 255 : Y);
			uchar* UVPointer = NV21.ptr<uchar>(rows+i/2);
			//计算U、V的值，进行2x2的采样
			if (i%2==0&&(j)%2==0)
			{
				int U = ((-44 * R - 87 * G + 131 * B) >> 8) + 128;
				int V = ((131 * R - 110 * G - 21 * B) >> 8) + 128;
				UVPointer[j] = V;
				UVPointer[j+1] = U;
				yuvbuff[UVindex++] = (V < 0) ? 0 : ((V > 255) ? 255 : V);
				yuvbuff[UVindex++] = (U < 0) ? 0 : ((U > 255) ? 255 : U);
			}
		}
	}
}

RESULT detecFace(string img_path, MHandle handle, MRESULT res, int pad=0)
{
	RESULT result;
	result.flag=-1;

	cv::Mat Img = cv::imread(img_path), pad_img, resized_img;
	if (Img.empty())
	{
		std::cout << "empty! check your image: "<<img_path<<endl;
		result.error_code = -2;
		return result;
	}
	
	// resize 至4的倍数
	copyMakeBorder(Img, pad_img, pad, pad, pad, pad, BORDER_CONSTANT, 0);
	Img = pad_img;
	int cols = Img.cols;
	int rows = Img.rows;

	if (cols%4!=0||rows%4!=0)
	{
		cols = (cols/4)*4;
		rows = (rows/4)*4;
		if (rows<10||cols<10)
		{
			cout<<"image size is too small"<<endl;
			result.error_code = -3;
			return result;
		}
		else
		{
			Size dsize(cols,rows);
			resize(pad_img,resized_img,dsize,0,0,INTER_AREA);
			Img = resized_img;
		}
		
	}

	int width = Img.cols;
	int height = Img.rows;

	MUInt8* img_data_1 = (MUInt8*)malloc(width*height*3/2);
	opencvRGB2NV21(Img,img_data_1, width, height);
	
	// cout<<"img size: "<<width<<" "<<height<<endl;

	ASVLOFFSCREEN offscreen1 = { 0 };
	ColorSpaceConversion(width, height, ASVL_PAF_NV21, img_data_1, offscreen1);
	
	ASF_MultiFaceInfo detectedFaces1 = { 0 };
	ASF_SingleFaceInfo SingleDetectedFaces = { 0 };
	ASF_FaceFeature feature1 = { 0 };
	
	res = ASFDetectFacesEx(handle, &offscreen1, &detectedFaces1);
	// cout<<img_path<<" detect faces: "<<detectedFaces1.faceNum<<", res "<<res<<endl;
	SafeArrayDelete(img_data_1);

	if (res!=MOK)
	{
		result.error_code = res;
		return result;
	}
	if (detectedFaces1.faceNum!=1)
	{	
		// 检测到人脸不为1，返回错误码-1
		result.error_code = -1;
		return result;
	}

	// extract face feature
	SingleDetectedFaces.faceRect.left = detectedFaces1.faceRect[0].left;
	SingleDetectedFaces.faceRect.top = detectedFaces1.faceRect[0].top;
	SingleDetectedFaces.faceRect.right = detectedFaces1.faceRect[0].right;
	SingleDetectedFaces.faceRect.bottom = detectedFaces1.faceRect[0].bottom;
	SingleDetectedFaces.faceOrient = detectedFaces1.faceOrient[0];
	res = ASFFaceFeatureExtractEx(handle, &offscreen1, &SingleDetectedFaces, &feature1);
	if (res != MOK)
	{
		result.error_code = res;
		return result;
	}
	else
	{
		result.flag=1;
		return result;
	}

}

int CountLines(char *filename)
{
    ifstream ReadFile;
    int n=0;
    string tmp;
    ReadFile.open(filename,ios::in);//ios::in 表示以只读的方式读取文件
    if(ReadFile.fail())//文件打开失败:返回0
    {
        return 0;
    }
    else//文件存在
    {
        while(getline(ReadFile,tmp,'\n'))
        {
            n++;
        }
        ReadFile.close();
        return n;
    }
}
// ./face_reg ../files/data_path1.txt ../files/results1.txt 1
int main(int argc, char** argv)
{

	MRESULT res = MOK;
	res = ASFOnlineActivation(APPID, SDKKEY);
	if (MOK != res && MERR_ASF_ALREADY_ACTIVATED != res)
		printf("ASFOnlineActivation fail: %d\n", res);
	else
		printf("ASFOnlineActivation sucess: %d\n", res);

	//初始化引擎
	MHandle handle = NULL;
	MInt32 mask = ASF_FACE_DETECT | ASF_FACERECOGNITION | ASF_AGE | ASF_GENDER | ASF_FACE3DANGLE | ASF_LIVENESS | ASF_IR_LIVENESS;
	res = ASFInitEngine(ASF_DETECT_MODE_IMAGE, ASF_OP_0_ONLY, NSCALE, FACENUM, mask, &handle);
	if (res != MOK)
		printf("ASFInitEngine fail: %d\n", res);
	else
		printf("ASFInitEngine sucess: %d\n", res);

	
	string infile_path(argv[1]), outfile_path(argv[2]);
	int pad = atoi(argv[3]);
	// cout<<infile_path<<endl;
	// cout<<outfile_path<<endl;

	ifstream img_paths;
	img_paths.open(infile_path);
	ofstream outfile;
	outfile.open(outfile_path);
	string line;

	int num_lines = CountLines(argv[1]),complet=0;
	cout<<num_lines<<" imgs"<<endl;

	while(getline(img_paths,line))
	{
		string img_path = line.substr(0,line.length());
		RESULT result = detecFace(img_path, handle, res, pad);
		if (result.flag==-1)
		{
			outfile<<img_path<<" "<<result.error_code<<endl;
		}
		complet++;
		if (complet%100==0) cout<<complet<<"/"<<num_lines<<endl;
	}
	img_paths.close();
	outfile.close();

    return 0;
}

