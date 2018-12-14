#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;
void Kmeans_Points()
{
	const int MAX_CLUSTERS = 50;
	Scalar colorTab[] =
	{
		Scalar(0, 0, 255),
		Scalar(0,255,0),
		Scalar(255,100,100),
		Scalar(255,0,255),
		Scalar(0,255,255)
	};
	Mat img(500, 500, CV_8UC3);
	RNG rng(12345);
	for (;;)
	{
		int k, clusterCount = rng.uniform(2, MAX_CLUSTERS + 1);
		int i, sampleCount = rng.uniform(1, 1001);
		Mat points(sampleCount, 1, CV_32FC2), labels;
		clusterCount = MIN(clusterCount, sampleCount);
		Mat centers;
		/* generate random sample from multigaussian distribution */
		for (k = 0; k < clusterCount; k++)
		{
			Point center;
			center.x = rng.uniform(0, img.cols);
			center.y = rng.uniform(0, img.rows);
			Mat pointChunk = points.rowRange(k*sampleCount / clusterCount,
				k == clusterCount - 1 ? sampleCount :
				(k + 1)*sampleCount / clusterCount);
			rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(img.cols*0.05, img.rows*0.05));
		}
		randShuffle(points, 1, &rng);
		kmeans(points, clusterCount, labels,
			TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
			3, KMEANS_PP_CENTERS, centers);
		img = Scalar::all(0);
		for (i = 0; i < sampleCount; i++)
		{
			int clusterIdx = labels.at<int>(i);
			Point ipt = points.at<Point2f>(i);
			circle(img, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA);
		}
		imshow("clusters", img);
		char key = (char)waitKey();
		if (key == 27 || key == 'q' || key == 'Q') // 'ESC'
			break;
	}
}
void Kmeans_Image(string imgName, int clusterNum) {
	//Đọc ảnh cần phân cụm màu
	Mat src = imread(imgName);
	//imshow("original", src);

	//Khởi tạo ảnh dùng cho Kmeans
	//Tạo ma trận đặc trưng MxN,3
	Mat samples(src.rows * src.cols, 3, CV_32F);

	int clusterCount = clusterNum;//Khởi tạo số cụm ban đầu
	Mat labels; //Lưu trữ chỉ mục cụm cho mỗi mẫu
	int attempts = 5; //Giới hạn vòng lặp
	Mat centers;

	//Copy các giá trị điểm ảnh từ ảnh gốc
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x * src.rows, z) = src.at<Vec3b>(y, x)[z];
	//Gọi thuật toán Kmeans của OpenCV
	kmeans(samples, clusterCount, labels,
		TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001)
		, attempts, KMEANS_PP_CENTERS, centers);

	//TermCriteria(Type,max_cluster_count,epsilon)
	//KMEANS_PP_CENTERS : Khởi tạo trung tâm các cụm theo thuật toán của Arthur and Vassilvitskii [Arthur2007]

	//Đổi ma trận đặc trưng ban đầu thành ảnh 3 thành phần màu
	Mat new_image(src.size(), src.type());
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x * src.rows, 0);
			new_image.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			new_image.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			new_image.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		}
	std::string k_val = "K = " + std::to_string(clusterCount);
	putText(new_image, k_val, Point(50, 50), 1, 2, Scalar(0, 0, 255), 2);
	imshow("clustered image" + std::to_string(clusterCount), new_image);
}
void GomMauAnh() {
	Mat src = imread("birdofparadise.jpg");
	imshow("original", src);
	Mat dst = src.clone();
	int thres[2] = { 80,160 };

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			float val = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2]) / 3;
			if (val < thres[0])
			{
				dst.at<Vec3b>(i, j)[0] = 40;
				dst.at<Vec3b>(i, j)[1] = 80;
				dst.at<Vec3b>(i, j)[2] = 0;
			}

			else if (val > thres[1])
			{
				dst.at<Vec3b>(i, j)[0] = 80;
				dst.at<Vec3b>(i, j)[1] = 160;
				dst.at<Vec3b>(i, j)[2] = 0;
			}
			else
			{
				dst.at<Vec3b>(i, j)[0] = 128;
				dst.at<Vec3b>(i, j)[1] = 255;
				dst.at<Vec3b>(i, j)[2] = 0;


			}
		}
	}
	imshow("dst", dst);
}
int main(int /*argc*/, char** /*argv*/)
{
	//Kmeans_Points();
	string imgName = "birdofparadise.jpg";
	Mat img = imread(imgName);
	imshow("org", img);
	Kmeans_Image(imgName, 3);
	waitKey();
	return 0;
}