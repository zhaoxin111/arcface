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
#include <set>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pthread.h>
#include <unistd.h>

using namespace cv;
using namespace std;

//从开发者中心获取APPID/SDKKEY(以下均为假数据，请替换)
#define APPID "6FdXUSDAUCAJi5KiF23fNKvU6V4idckFyLLvnw4idE66"
#define SDKKEY "EyWVcTzti9EwPAWVv43GoW9aM6anPMYLFpvEvKsAnGLd"

#define NSCALE 16 
#define FACENUM	5
#define PAD 5

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
	float score=-1;
	ASF_FaceFeature feature={0};
	int ASFProcessEx=-1; // 是否成功检测到人脸属性，检测到：1，没有：-1
	int genderInfo = -1;
	ASF_Face3DAngle angleInfo = { 0 };
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

int detectGender(MHandle handle,MRESULT res)
{
	int gender=-1;
	ASF_GenderInfo genderInfo = { 0 };
	res = ASFGetGender(handle, &genderInfo);
	if (res != MOK)
	{
		cout<<"detect gender fail"<<endl;
		return gender;
	}
	gender=genderInfo.genderArray[0];
	return gender;
}

ASF_Face3DAngle detect3Dangle(MHandle handle,MRESULT res)
{
	ASF_Face3DAngle angleInfo = { 0 };
	res = ASFGetFace3DAngle(handle, &angleInfo);
	if (res != MOK)
	{
		cout<<"detect 3Dangle fail"<<endl;
	}
	
	return angleInfo;
}

void detecFace(string img_path, MHandle handle, MRESULT res, int pad,RESULT*& result)
{
	result->flag=-1;

	cv::Mat Img = cv::imread(img_path), pad_img, resized_img;
	// cout<<Img.size()<<endl;
	if (Img.empty())
	{
		std::cout << "empty! check your image: "<<img_path<<endl;
		result->error_code = -2;
		return ;
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
			result->error_code = -3;
			return ;
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
		result->error_code = res;
		return ;
	}
	if (detectedFaces1.faceNum!=1)
	{	
		// 检测到人脸不为1，返回错误码-1
		result->error_code = -1;
		return ;
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
		result->error_code = res;
		return ;
	}
	else
	{
		result->flag=1;
		result->feature=feature1;

		// gender and 3D angle
		MInt32 processMask = ASF_AGE | ASF_GENDER | ASF_FACE3DANGLE | ASF_LIVENESS;
		res = ASFProcessEx(handle, &offscreen1, &detectedFaces1, processMask);
		if (res != MOK)
		{
			// printf("ASFProcessEx fail: %d\n", res);
			result->ASFProcessEx=-1;
		}
		else
		{
			// 	printf("ASFProcessEx sucess: %d\n", res);
			int gender = detectGender(handle,res);
			ASF_Face3DAngle angleInfo = detect3Dangle(handle,res);
			result->ASFProcessEx=1;
			result-> genderInfo = gender;
			result->angleInfo = angleInfo;
		}
		// cout<<img_path<<" gender "<<gender<<endl;
		// printf("face 3dAngle: roll: %lf yaw: %lf pitch: %lf\n", angleInfo.roll[0], angleInfo.yaw[0], angleInfo.pitch[0]);
		return ;
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

void init(MRESULT& res,MHandle& handle)
{
	
	res = ASFOnlineActivation(APPID, SDKKEY);
	if (MOK != res && MERR_ASF_ALREADY_ACTIVATED != res)
		printf("ASFOnlineActivation fail: %d\n", res);
	else
		printf("ASFOnlineActivation sucess: %d\n", res);

	//初始化引擎
	
	MInt32 mask = ASF_FACE_DETECT | ASF_FACERECOGNITION | ASF_AGE | ASF_GENDER | ASF_FACE3DANGLE | ASF_LIVENESS | ASF_IR_LIVENESS;
	res = ASFInitEngine(ASF_DETECT_MODE_IMAGE, ASF_OP_0_ONLY, NSCALE, FACENUM, mask, &handle);
	if (res != MOK)
		printf("ASFInitEngine fail: %d\n", res);
	else
		printf("ASFInitEngine sucess: %d\n", res);
}

RESULT* copyfeature(RESULT* result1)
{
	RESULT* copy_res=new RESULT;
	copy_res->flag = result1->flag;
	copy_res->genderInfo = result1->genderInfo;
	copy_res->angleInfo = result1->angleInfo;
	copy_res->error_code = result1->error_code;
	copy_res->score = result1->score;

	copy_res->feature.featureSize = result1->feature.featureSize;
	copy_res->feature.feature = (MByte *)malloc(result1->feature.featureSize);
	memset(copy_res->feature.feature, 0, result1->feature.featureSize);
	memcpy(copy_res->feature.feature, result1->feature.feature, result1->feature.featureSize);
	return copy_res;
}

vector<cv::String> get_all_imgs(cv::String folder_path)
{
	// cout<<folder_path;
	std::vector<cv::String> file_names;
    cv::glob(folder_path, file_names,true);
	// cout<<file_names.size();
	return file_names;
}

float person2person_score(cv::String subject1,cv::String subject2,MHandle handle,MRESULT res)
{
	float mean_socre = -1;
	subject1 = subject1+"/*.jpg";
	subject2 = subject2+"/*.jpg";
	std::vector<cv::String> imgs1,imgs2;
	imgs1 = get_all_imgs(subject1);
	imgs2 = get_all_imgs(subject2);
	
	if(imgs1.size()==0 || imgs2.size()==0) 
	{
		cout<<subject1<<" or "<<subject2<<" is empty"<<endl;
		return mean_socre;
	}
	
	int valid_num_compare=0,VALID_TRY=1,MAX_TRY=3,num_imgs1=imgs1.size(),num_imgs2=imgs2.size();
	vector<float> scores;

	// cout<<"find "<<num_imgs1<<" "<<num_imgs2<<endl;
	int MAX_VALID = 1;
	vector<RESULT*> valid_results1;
	vector<string> valid_paths1;
	int valid_num1 = 0;
	vector<RESULT*> valid_results2;
	vector<string> valid_paths2;
	int valid_num2 = 0;
	// find valid faces
	for (int i=0;i<num_imgs1;i++)
	{
		if (valid_num1>=MAX_VALID) break;
		cv::String img1_path = imgs1[i];
		RESULT *result1=new RESULT;
		detecFace(img1_path, handle, res, PAD,result1);
		if (result1->flag==-1 || (result1->ASFProcessEx==-1)) continue;
		float roll = fabs(result1->angleInfo.roll[0]);
		float pitch = fabs(result1->angleInfo.pitch[0]);
		float yaw = fabs(result1->angleInfo.yaw[0]);
		if((roll<=12)&&(yaw<=15)&&(pitch<=18))
		{
			RESULT* res=copyfeature(result1);
			valid_results1.push_back(res);
			valid_paths1.push_back(imgs1[i]);
			valid_num1++;
		}
		
	}
	for (int i=0;i<num_imgs2;i++)
	{
		if (valid_num2>=MAX_VALID) break;
		cv::String img2_path = imgs2[i];
		RESULT *result2=new RESULT;
		detecFace(img2_path, handle, res, PAD,result2);
		if (result2->flag==-1 || (result2->ASFProcessEx==-1)) continue;
		float roll = fabs(result2->angleInfo.roll[0]);
		float pitch = fabs(result2->angleInfo.pitch[0]);
		float yaw = fabs(result2->angleInfo.yaw[0]);
		if((roll<=12)&&(yaw<=15)&&(pitch<=18))
		{
			RESULT* res=copyfeature(result2);
			valid_results2.push_back(res);
			valid_paths2.push_back(imgs2[i]);
			valid_num2++;
		}
		
	}
	if (valid_paths1.empty()||valid_paths2.empty()) return mean_socre;
	
	std::set<float> scores_set;
	std::vector<float> scores_vec;

	srand((unsigned)time(NULL));
	for (int i=0;i<MAX_TRY;i++)
	{
		if (valid_num_compare>=VALID_TRY) break;
		int idx1 = (rand()%valid_num1),idx2 = (rand()%valid_num2);
		cv::String img1_path = valid_paths1[idx1],img2_path=valid_paths2[idx2];
		MFloat confidenceLevel=-1;
		res = ASFFaceFeatureCompare(handle, &valid_results1[idx1]->feature, &valid_results2[idx2]->feature, &confidenceLevel);
		// if (res != MOK)
			// printf("ASFFaceFeatureCompare fail: %d\n", res);
			
		if (res == MOK)
		{
			if (scores_set.count(confidenceLevel)==0)
			{
				scores_set.insert(confidenceLevel);
				valid_num_compare++;
			}
		}		
	}
	if (scores_set.empty()) return mean_socre;
	scores_vec.assign(scores_set.begin(),scores_set.end());
	
	// if (scores_vec.size()<=2) return mean_socre;
	// sort(scores_vec.begin(),scores_vec.end());
	float sum=0;
	for (int i=0;i<scores_vec.size();i++)
	{
		// cout<<scores_vec[i]<<" ";
		sum+=scores_vec[i];
	}
	mean_socre = sum/(scores_vec.size());
	// cout<<endl<<mean_socre<<endl;
	return mean_socre;
}

vector<string> split(string str, string delim) {
	vector<string> res;
	size_t pos = 0;
	std::string token;
	while ((pos = str.find(delim)) != std::string::npos) {
		token = str.substr(0, pos);
		// std::cout << token<<" ";
		res.push_back(token);
		// cout<<res[0]<<endl;
		str.erase(0, pos + delim.length());
	}
	// std::cout << str << std::endl;
	res.push_back(str);
 
	return res;
}

class thread_data
{
public:
	int thread_id=0;
	char* infile;
	char* outfile;
	int start = 0;
	int end = 0;
	float min_model_score=0;
	float max_model_score=0;
};


void *process(void *param)
{
	thread_data *data=(thread_data *)param;

	float THRESHOLD_MODEL_MIN=data->min_model_score,THRESHOLD_MODEL_MAX=data->max_model_score;
	// cout<<THRESHOLD_MODEL_MIN<<THRESHOLD_MODEL_MAX<<endl;
	MRESULT res = MOK;
	MHandle handle = NULL;
	init(res,handle);

	// string infile_path(data->infile);
	// cout<<"infile_path "<<data->infile<<endl;
	// cout<<"outfile_path "<<data->outfile<<endl;
	// string outfile_path(data->outfile);
	ifstream img_paths;
	img_paths.open(data->infile);
	// cout<<data->outfile<<endl;
	ofstream outfile;
	outfile.open(data->outfile);
	// cout<<data->infile<<endl;

	int num_lines = data->end-data->start;
	// cout<<num_lines<<" pairs"<<endl;

	
	int complet = 0,line_index=0;
	string line;
	while (getline(img_paths,line)&&line_index<data->start)
	{
		line_index++;
		/* code */
	}
	cout<<data->thread_id<<" index "<<line_index<<endl;

	// getline(img_paths,line);
	// cout<<line<<endl;
	while(getline(img_paths,line)&&complet<num_lines)
	{
		complet++;
		line = line.substr(0,line.length());
		// cout<<"line: "<<line<<endl;
		// cout<<data->thread_id<<","<<line<<endl;
		vector<string> parts = split(line," ");
		// printf("%s,%s,%s\n", parts[0].c_str(),parts[1].c_str(),parts[2].c_str());
		if (parts.size()!=3)
		{
			cout<<"split error: "<<line<<endl;
			sleep(1);
			continue;
		}
		// 过滤model预测相似度小于阈值的id对
		float model_score = stof(parts[2]);
		if (model_score>THRESHOLD_MODEL_MAX||model_score<THRESHOLD_MODEL_MIN) continue;
		float score = person2person_score(parts[0],parts[1],handle,res);
		// printf("embedding score: %s, sdk score %2.3f \n",parts[2].substr(0,4).c_str(),score);
		line=line+" "+to_string(score);
		outfile<<line<<endl;
		// endTime = clock();
		
		// printf("thread id: %d, start %d end %d, current %d \n",data->thread_id,line_index,data->end,complet);
		if (complet%100==0) cout<<data->thread_id<<" "<<complet<<"/"<<num_lines<<endl;
	}
	cout<<data->thread_id<<" pthread_exit"<<endl;
	img_paths.close();
	outfile.close();
	pthread_exit(NULL);
}

int main(int argc, char** argv)
{
	int NUM_THREADS=atoi(argv[1]);
	float min_model_score=atof(argv[2]), max_model_score=atof(argv[3]);
	int start=atoi(argv[4]),step=atoi(argv[5]);
	char* infile = argv[6];

	printf("num_threads:%d,min_score:%f,max_score:%f,start:%d,step:%d,%s\n",NUM_THREADS,min_model_score,max_model_score,start,step,infile);
	string root_outfile = "/data/zhaoxin_data/renren_filtered_data_v3/features/sdk/test_thread_"+\
	to_string(min_model_score)+"_"+to_string(max_model_score)+"_threadID_";
	pthread_t threads[NUM_THREADS];
	thread_data td[NUM_THREADS];
	int rc=0;
	
	for(int i=0; i < NUM_THREADS; i++){
		sleep(1);
		cout <<"main() : creating thread, " << i << endl;
		td[i].thread_id = i;
		td[i].infile = infile;
		
		string tmp = root_outfile+to_string(i)+".txt";
		td[i].outfile = strdup(tmp.c_str());  // string to char*
		// cout<<td[i].outfile<<endl;
		td[i].start = start;
		td[i].end = start+step;
		start = start+step;

		td[i].min_model_score= min_model_score;
		td[i].max_model_score= max_model_score;
		rc = pthread_create(&threads[i], NULL,
							process, (void *)&td[i]);
		if (rc){
			cout << "Error:unable to create thread," << rc << endl;
			exit(-1);
		}
	}
	pthread_exit(NULL);



	// float THRESHOLD_MODEL_MIN=stof(argv[3]),THRESHOLD_MODEL_MAX=stof(argv[4]);
	// cout<<THRESHOLD_MODEL_MIN<<THRESHOLD_MODEL_MAX<<endl;
	// MRESULT res = MOK;
	// MHandle handle = NULL;
	// init(res,handle);

	// string infile_path(argv[1]);
	// string outfile_path(argv[2]);
	// ifstream img_paths;
	// img_paths.open(infile_path);
	// ofstream outfile;
	// outfile.open(outfile_path);

	// int num_lines = CountLines(argv[1]),complet=0;
	// cout<<num_lines<<" pairs"<<endl;

	// clock_t startTime,endTime;
	// startTime = clock();
	// int num_iter = 0;
	// string line;
	// while(getline(img_paths,line))
	// {
	// 	num_iter++;
	// 	line = line.substr(0,line.length());
	// 	vector<string> parts = split(line," ");
	// 	// 过滤model预测相似度小于阈值的id对
	// 	float model_score = stof(parts[2]);
	// 	if (model_score>THRESHOLD_MODEL_MAX||model_score<THRESHOLD_MODEL_MIN) continue;
	// 	float score = person2person_score(parts[0],parts[1],handle,res);
	// 	// printf("embedding score: %s, sdk score %2.3f \n",parts[2].substr(0,4).c_str(),score);
	// 	line=line+" "+to_string(score);
	// 	outfile<<line<<endl;
	// 	endTime = clock();
	// 	cout << "time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC / num_iter << "s" << endl;
	// 	complet++;
		
	// 	if (complet%100==0) cout<<complet<<"/"<<num_lines<<endl;
	// }
	cout<<"done!"<<endl;
    return 0;
}

