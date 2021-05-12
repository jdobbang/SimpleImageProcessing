#include<opencv2/opencv.hpp>
#include<stdio.h>

using namespace cv;

void RGB2YUV(Mat InpImg, int height, int width)
{
	unsigned char** Y;
	unsigned char** U, **V;
	
	if (InpImg.data != NULL)
	{
		Y = new unsigned char*[InpImg.rows];
		U = new unsigned char*[InpImg.rows];
		V = new unsigned char*[InpImg.rows];
		for (int h = 0; h < InpImg.rows; h++) {
			Y[h] = new unsigned char[InpImg.cols];
			U[h] = new unsigned char[InpImg.cols];
			V[h] = new unsigned char[InpImg.cols];
		}
		
		for (int h = 0; h < InpImg.rows; h++)
		{
			for (int w = 0; w < InpImg.cols; w++)
			{
				unsigned char r, g, b;
				r = InpImg.at<unsigned char>(h, w * 3 + 2);
				g = InpImg.at<unsigned char>(h, w * 3 + 1);
				b = InpImg.at<unsigned char>(h, w * 3); 
				Y[h][w] = 0.299 *r + 0.587*g + 0.114*b;
				U[h][w] = -0.169*r - 0.331*g + 0.499*b + 128;
				V[h][w] = 0.499*r - 0.418*g - 0.081*b + 128;
			}
		}
		Mat y(height, width, CV_8UC1);

		for (int h = 0; h < InpImg.rows; h++)
		{
			for (int w = 0; w < InpImg.cols; w++)
			{
				y.at<uchar>(h, w) = Y[h][w];
			}
		}


		Mat u(height, width, CV_8UC1);

		for (int h = 0; h < InpImg.rows; h++)
		{
			for (int w = 0; w < InpImg.cols; w++)
			{
				u.at<uchar>(h, w) = U[h][w];
			}
		}


		Mat v(height, width, CV_8UC1);

		for (int h = 0; h < InpImg.rows; h++)
		{
			for (int w = 0; w < InpImg.cols; w++)
			{
				v.at<uchar>(h, w) = V[h][w];
			}
		}


		Mat sumyuv(height, width, CV_8UC3);

		for (int h = 0; h < InpImg.rows; h++)
		{
			for (int w = 0; w < InpImg.cols; w++)
			{
				sumyuv.at<Vec3b>(h, w)[0] = y.at<uchar>(h, w);
				sumyuv.at<Vec3b>(h, w)[1] = v.at<uchar>(h, w);
				sumyuv.at<Vec3b>(h, w)[2] = u.at<uchar>(h, w);
			}
		}

		imshow("y", y);
		imshow("u", u);
		imshow("v", v);
		imshow("sumyuv", sumyuv);
		waitKey(0);

	}
}

void r2ycvtColor(Mat img_color, int height, int width)
{
	Mat yuvImg(height, width, CV_8UC3);
	Mat yuv2rgb(height, width, CV_8UC3);
	cvtColor(img_color, yuvImg, CV_BGR2YUV);


	imshow("rgb", img_color);
	imshow("yuv", yuvImg);
	waitKey(0);

}

void main()
{
	Mat img_color = imread("test.jpg", IMREAD_COLOR);
	int height = img_color.rows;
	int width = img_color.cols;

	RGB2YUV(img_color, height, width);

	r2ycvtColor(img_color, height, width);
	waitKey(0);
	return;
}
