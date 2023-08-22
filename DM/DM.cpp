#include <opencv2/core/core.hpp>  
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>   
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>   
#include <fstream>   
#include<time.h>

using namespace std;
using namespace cv;

inline bool XYCheck(const Mat& img, const Point2i& p)
{
	return p.x >= 0 && p.x < img.cols&& p.y >= 0 && p.y < img.rows;
}
inline bool XYCheck(const Mat& img, int c, int r)
{
	return c >= 0 && c < img.cols&& r >= 0 && r < img.rows;
}

//SAD matching cost
double SAD_MC(const Mat& img1, int c1, int r1, const Mat& img2, int c2, int r2, int size)
{
	if (img1.type() != CV_8UC1 || !XYCheck(img1, c1, r1) || img2.type() != CV_8UC1 || !XYCheck(img2, c2, r2) || size <= 0)
	{
		return FLT_MAX;
	}
	double diff = 0;
	int num = 0;
	for (int i = -size; i <= size; i++)
	{
		for (int j = -size; j <= size; j++)
		{
			if (XYCheck(img1, c1 + j, r1 + i) && XYCheck(img2, c2 + j, r2 + i))
			{
				diff += fabs(img1.ptr<uchar>(r1 + i)[c1 + j] - img2.ptr<uchar>(r2 + i)[c2 + j]);
				++num;
			}
		}
	}
	return num != 0 ? diff / num : FLT_MAX;
}

//caculate disparity map by dynamic programming  
void DispDP(const Mat& img1, const Mat& img2, int min_disp, int max_disp, int P1, int P2, int m_size, Mat& disp)
{
	if (img1.empty() || img1.type() != CV_8UC1 || img2.empty() || img2.type() != CV_8UC1 || img1.size() != img2.size() || m_size < 1)
		return;

	int disp_range = max_disp - min_disp + 1;   // range of disparity
	disp = Mat(img1.size(), CV_8UC1, Scalar::all(0));   // disparity map
	for (int r = 0; r < img1.rows; ++r)
	{
		Mat diff(Size(disp_range, img1.cols), CV_64FC1, Scalar::all(0));   // disparity space
		Mat p_mat(Size(disp_range, img1.cols), CV_8UC1, Scalar::all(0));   // direction of the path
		for (int c1 = 0; c1 < img1.cols; ++c1)
		{
			for (int d = 0; d <= disp_range; ++d)
			{
				int c2 = c1 - d - min_disp;   //right pixel coordinate(x) under the current disparity
				if (c2 >= 0)
				{
					diff.ptr<double>(c1)[d] = SAD_MC(img1, c1, r, img2, c2, r, m_size);   //caculate SAD matching cost
				}
				else
				{
					diff.ptr<double>(c1)[d] = c1 < min_disp ? 0 : diff.ptr<double>(c1)[d - 1];    //preventing exceeding the boundary
				}

			}
		}

		Mat e_mat_cur(Size(1, disp_range), CV_64FC1, Scalar::all(0));   // energy space
		for (int c = 1; c < img1.cols; ++c)
		{
			Mat e_mat_pre = e_mat_cur.clone();
			for (int cur = 0; cur < disp_range; ++cur)
			{
				double cost_min = FLT_MAX;   //path cost under current state
				int p_min = 0;   //direction of the current path
				double e_cur = diff.ptr<double>(c)[cur];   //data item
				for (int pre = 0; pre < disp_range; pre++)
				{
					int deta_d = abs(cur - pre);
					double e_smooth = deta_d > 1 ? P2 : (deta_d == 0 ? 0 : P1);   //different disparity with different smooth penalty
					double e_path = e_cur + e_smooth + e_mat_pre.ptr<double>(pre)[0];   //path energy
					if (e_path < cost_min)
					{
						cost_min = e_path;
						p_min = pre;
					}
				}
				e_mat_cur.ptr<double>(cur)[0] = cost_min;
				p_mat.ptr<uchar>(c)[cur] = p_min;
			}
		}

		int p_min = 0;
		double e_min = e_mat_cur.ptr<double>(0)[0];
		for (int i = 0; i < disp_range; ++i)
		{
			if (e_mat_cur.ptr<double>(i)[0] < e_min)
			{
				p_min = i;
				e_min = e_mat_cur.ptr<double>(i)[0];
			}
		}

		//get disparity
		for (int c = img1.cols - 1; c >= 0; --c)
		{
			int d = p_mat.ptr<uchar>(c)[p_min];
			p_min = d;
			disp.ptr<uchar>(r)[c] = d + min_disp;
		}

		printf("Dynamic Programming...%d%%\r", (int)((r + 1) * 100.0 / (img1.rows)));
	}
}

//caculate disparity map by semi-global block matching
void DispSGBM(Mat left, Mat right, Mat& disp)
{
	int mindisparity = 0;
	int ndisparities = 64;
	int SADWindowSize = 7;
	//SGBM
	Ptr<StereoSGBM> sgbm = StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);
	int P1 = 8 * left.channels() * SADWindowSize * SADWindowSize;
	int P2 = 32 * left.channels() * SADWindowSize * SADWindowSize;
	sgbm->setP1(P1);
	sgbm->setP2(P2);
	sgbm->setPreFilterCap(63);//63
	sgbm->setUniquenessRatio(5);//5
	sgbm->setSpeckleRange(300);//100
	sgbm->setSpeckleWindowSize(10);//30
	sgbm->setDisp12MaxDiff(1);//1
	sgbm->setMode(StereoSGBM::MODE_HH);
	sgbm->compute(left, right, disp);
	disp.convertTo(disp, CV_32F, 1.0 / 16);              //除以16得到真实视差值

}

//caculate disparity map by block matching
void DispBM(Mat left, Mat right, Mat& disp) {

	int mindisparity = 0;
	int ndisparities = 64;
	int SADWindowSize = 7;

	Ptr<StereoBM> bm = StereoBM::create(ndisparities, SADWindowSize);
	// setter
	bm->setBlockSize(SADWindowSize);
	bm->setMinDisparity(mindisparity);
	bm->setNumDisparities(ndisparities);
	bm->setPreFilterSize(33);//15\21
	bm->setPreFilterCap(63);//31
	bm->setTextureThreshold(30);//15
	bm->setUniquenessRatio(10);//10\5
	bm->setDisp12MaxDiff(1);//1

	copyMakeBorder(left, left, 0, 0, 80, 0, IPL_BORDER_REPLICATE);  //防止黑边
	copyMakeBorder(right, right, 0, 0, 80, 0, IPL_BORDER_REPLICATE);
	bm->compute(left, right, disp);

	disp.convertTo(disp, CV_32F, 1.0 / 16); //除以16得到真实视差值
	disp = disp.colRange(80, disp.cols);

}

//disparity map hole filling
void insertDisp32f(Mat& disp)
{
	const int width = disp.cols;
	const int height = disp.rows;
	float* data = (float*)disp.data;
	Mat integralMap = Mat::zeros(height, width, CV_64F);
	Mat ptsMap = Mat::zeros(height, width, CV_32S);
	double* integral = (double*)integralMap.data;
	int* ptsIntegral = (int*)ptsMap.data;
	memset(integral, 0, sizeof(double) * width * height);
	memset(ptsIntegral, 0, sizeof(int) * width * height);
	for (int i = 0; i < height; ++i)
	{
		int id1 = i * width;
		for (int j = 0; j < width; ++j)
		{
			int id2 = id1 + j;
			if (data[id2] > 1e-3)
			{
				integral[id2] = data[id2];
				ptsIntegral[id2] = 1;
			}
		}
	}
	// integration interval
	for (int i = 0; i < height; ++i)
	{
		int id1 = i * width;
		for (int j = 1; j < width; ++j)
		{
			int id2 = id1 + j;
			integral[id2] += integral[id2 - 1];
			ptsIntegral[id2] += ptsIntegral[id2 - 1];
		}
	}
	for (int i = 1; i < height; ++i)
	{
		int id1 = i * width;
		for (int j = 0; j < width; ++j)
		{
			int id2 = id1 + j;
			integral[id2] += integral[id2 - width];
			ptsIntegral[id2] += ptsIntegral[id2 - width];
		}
	}
	int wnd;
	double dWnd = 2;
	while (dWnd > 1)
	{
		wnd = int(dWnd);
		dWnd /= 2;
		for (int i = 0; i < height; ++i)
		{
			int id1 = i * width;
			for (int j = 0; j < width; ++j)
			{
				int id2 = id1 + j;
				int left = j - wnd - 1;
				int right = j + wnd;
				int top = i - wnd - 1;
				int bot = i + wnd;
				left = max(0, left);
				right = min(right, width - 1);
				top = max(0, top);
				bot = min(bot, height - 1);
				int dx = right - left;
				int dy = (bot - top) * width;
				int idLeftTop = top * width + left;
				int idRightTop = idLeftTop + dx;
				int idLeftBot = idLeftTop + dy;
				int idRightBot = idLeftBot + dx;
				int ptsCnt = ptsIntegral[idRightBot] + ptsIntegral[idLeftTop] - (ptsIntegral[idLeftBot] + ptsIntegral[idRightTop]);
				double sumGray = integral[idRightBot] + integral[idLeftTop] - (integral[idLeftBot] + integral[idRightTop]);
				if (ptsCnt <= 0)
				{
					continue;
				}
				data[id2] = float(sumGray / ptsCnt);
			}
		}
		int s = wnd / 2 * 2 + 1;
		if (s > 201)
		{
			s = 201;
		}
		GaussianBlur(disp, disp, Size(s, s), s, s);
	}
}

//convert disparity map to depth image
Mat GetDepth(Mat disp, double fk, double baseline) {
	for (int r = 0; r < disp.rows; r++) {
		for (int c = 0; c < disp.cols; c++) {
			if (disp.ptr<float>(r)[c] > 0) {  //remove disparity = 0
				disp.ptr<float>(r)[c] = (fk * baseline) / disp.ptr<float>(r)[c];
			}
		}
	}
	return disp;
}

//convert disparity map to 3D point cloud
void Dep2PCL(Mat dep, vector <Point3f>& PointCloud, double x0, double y0, double fk) {
	for (int r = 0; r < dep.rows; r++) {
		for (int c = 0; c < dep.cols; c++) {
			if (dep.ptr<float>(r)[c] <= 0 || isinf(dep.ptr<float>(r)[c])) {
				continue;
			}
			else {
				Point3f tempP;
				tempP.z = dep.ptr<float>(r)[c];
				tempP.x = (c - x0) / fk;
				tempP.y = (r - y0) / fk;
				PointCloud.push_back(tempP);
			}
		}
	}
	ofstream outputfile;
	outputfile.open("MatchPoints.xyztri");
	if (outputfile.is_open()) {
		outputfile << "xyztri" << endl << PointCloud.size() + 1 << endl;
		for (size_t i = 0; i < PointCloud.size(); i++) {
			outputfile << PointCloud.at(i).x << " " << PointCloud.at(i).y << " " << PointCloud.at(i).z << endl;
		}
	}
	else {
		return;
	}
	outputfile.close();
}

//SAD 
void SAD(string path_left, string path_right) 
{
	//set SAD parameters
	int winSize, DSR;
	cout << "Please set matching parameters:" << endl;
	cout << "Size of the window:";
	cin >> winSize;
	cout << "Moving range:";
	cin >> DSR;

	//Start clock
	clock_t start = clock();
	clock_t finish;

	Mat left = imread(path_left);
	Mat right = imread(path_right);

	namedWindow("leftimg");
	imshow("leftimg", left);

	namedWindow("rightimg");
	imshow("rightimg", right);

	int Height = left.rows;
	int Width = left.cols;
	//Initialize matrix
	Mat Kernel_L = Mat::zeros(winSize, winSize, CV_8U);
	Mat Kernel_R = Mat::zeros(winSize, winSize, CV_8U);
	Mat disp = Mat::zeros(Height, Width, CV_8U);//true disparity
	Mat disp_show = Mat::zeros(Height, Width, CV_8U);//for display

	//Calculate disparity map by SAD
	for (int i = 0; i < Width - winSize; ++i) {
		for (int j = 0; j < Height - winSize; ++j) {
			Kernel_L = left(Rect(i, j, winSize, winSize));//Extract the rectangle area at (i,j) from left.
			Mat MM = Mat::zeros(1, DSR, CV_32F);

			for (int k = 0; k < DSR; ++k) {
				int x = i - k;
				if (x >= 0) {
					Kernel_R = right(Rect(x, j, winSize, winSize));
					Mat Dif;
					absdiff(Kernel_L, Kernel_R, Dif);
					Scalar ADD = sum(Dif);
					float a = ADD[0];
					MM.at<float>(k) = a;
				}
				Point minLoc;
				minMaxLoc(MM, NULL, NULL, &minLoc, NULL);//Find location of minimum matching cost at x.

				int loc = minLoc.x;
				disp.at<char>(j, i) = loc;
			}
		}
		printf("SAD Matching...%d%%\r", (int)((i+1) * 100.0 / (Width - winSize)));
	}

	//get disparity map
	Mat disp_m, depth;
	medianBlur(disp, disp_m, 3);
	normalize(disp_m, disp_show, 0, 255, NORM_MINMAX, CV_8UC1);
	imshow("Disparity_SAD", disp_show);
	imwrite("Disparity_SAD.bmp", disp_show);

	//Stop clock
	finish = clock();
	cout << endl;
	cout << "Finish!" << endl;
	cout << "running time：" << (double)(finish - start) / CLOCKS_PER_SEC << "s" << endl;
	waitKey(0);
}

//Dynamic Programming
void DynamicProgram(string path_left, string path_right)
{
	//set dynamic programming parameters
	int P1, P2, size;
	cout << "Please set matching parameters:" << endl;
	cout << "smooth penalty:" << endl;
	cout << "P1: ";
	cin >> P1;
	cout << "P2: ";
	cin >> P2;
	cout << "half size of the window:";
	cin >> size;

	clock_t start = clock();
	clock_t finish;

	Mat img1 = imread(path_left, IMREAD_COLOR);
	Mat img2 = imread(path_right, IMREAD_COLOR);
	imshow("img1", img1);
	imshow("img2", img2);
	//convert RGB to gray
	Mat gray1, gray2;
	cvtColor(img1, gray1, COLOR_BGR2GRAY);
	cvtColor(img2, gray2, COLOR_BGR2GRAY);

	//get disparity map
	Mat disp, disp_m, disp_show;
	DispDP(gray1, gray2, 0, 100, P1, P2, size, disp);
	medianBlur(disp, disp_m, 3);
	normalize(disp_m, disp_show, 0, 255, NORM_MINMAX, CV_8UC1);
	imshow("Disparity_DP", disp_show);
	imwrite("Disparity_DP.bmp", disp_show);

	finish = clock();
	cout << endl;
	cout << "Finish!" << endl;
	cout << "running time：" << (double)(finish - start) / CLOCKS_PER_SEC << "s" << endl;
	waitKey(0);
}

//Semi-Global Block Matching in opencv
void SGBM(string path_left, string path_right) 
{
	//set camera parameters
	double fk, baseline;
	cout << "Please set camera parameters:" << endl;
	cout << "fk:";
	cin >> fk;
	cout << "baseline:";
	cin >> baseline;

	//start clock
	clock_t start = clock();
	clock_t finish;

	Mat left = imread(path_left, IMREAD_GRAYSCALE);
	Mat right = imread(path_right, IMREAD_GRAYSCALE);

	//get disparity map
	Mat disp, disp_m, disp_show, depth;
	DispSGBM(left, right, disp);
	medianBlur(disp, disp_m, 3);
	insertDisp32f(disp_m);
	normalize(disp_m, disp_show, 0, 255, NORM_MINMAX, CV_8UC1);
	imshow("Disparity_SGBM", disp_show);
	imwrite("Disparity_SGBM.bmp", disp_show);

	//get depth map
	depth = GetDepth(disp_m, fk, baseline);   //f and baseline are unknown

	//convert depth map to 3D point cloud 
	vector <Point3f> PointCloud;
	Dep2PCL(depth, PointCloud, depth.rows / 2, depth.cols / 2, fk);

	//stop clock
	finish = clock();
	cout << endl;
	cout << "Finish!" << endl;
	cout << "running time：" << (double)(finish - start) / CLOCKS_PER_SEC << "s" << endl;
	waitKey(0);
}

//Block Matching in opencv
void BM(string path_left, string path_right) 
{
	//set camera parameters
	double fk, baseline;
	cout << "Please set camera parameters:" << endl;
	cout << "fk:";
	cin >> fk;
	cout << "baseline:";
	cin >> baseline;

	clock_t start = clock();
	clock_t finish;

	Mat left = imread(path_left, IMREAD_GRAYSCALE);
	Mat right = imread(path_right, IMREAD_GRAYSCALE);

	//get disparity map
	Mat disp, disp_m, disp_show, depth;
	DispBM(left, right, disp);
	medianBlur(disp, disp_m, 3);
	insertDisp32f(disp_m);
	normalize(disp_m, disp_show, 0, 255, NORM_MINMAX, CV_8UC1);
	imshow("Disparity_BM", disp_show);
	imwrite("Disparity_BM.bmp", disp_show);

	//get depth map
	depth = GetDepth(disp_m, fk, baseline);   //f and baseline are unknown

	//convert depth map to 3D point cloud 
	vector <Point3f> PointCloud;
	Dep2PCL(depth, PointCloud, depth.rows / 2, depth.cols / 2, 50);

	finish = clock();
	cout << endl;
	cout << "Finish!" << endl;
	cout << "running time：" << (double)(finish - start) / CLOCKS_PER_SEC << "s" << endl;
	waitKey(0);
}

int main()
{
	string path_left = "1-2.lei.bmp";
	string path_right = "1-2.rei.bmp";

	cout << "Please select a method: ( 1:SAD 2:BM 3:DP 4:SGBM )" << endl;
	int i;
	cin >> i;

	//local match method
	//SAD
	if (i == 1) {
		SAD(path_left, path_right);
	}
	//BM
	else if (i == 2) {
		BM(path_left, path_right);
	}
	//global match method
	//Dynamic programming(DP)
	else if (i == 3) {
		DynamicProgram(path_left, path_right);
	}
	//semi- global matching
	//SGBM
	else if (i == 4) {
		SGBM(path_left, path_right);
	}
	else {
		cout << "No such solution!" << endl;
		return 0;
	}
	return 0;
}