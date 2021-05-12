#include <stdio.h>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;


Mat padding(Mat input, Mat padding1, Mat padding2,int n) {

	int plus = n / 2; 

	if (input.channels() == 3) 
	{
		for (int h = 0; h < input.rows; h++) {
			for (int w = 0; w < input.cols; w++) {
				padding1.at<Vec3b>(h+plus, w+plus)[2] = input.at < Vec3b >(h,w) [2];
				padding1.at<Vec3b>(h+plus, w+plus)[1] = input.at < Vec3b >(h, w)[1];
				padding1.at<Vec3b>(h+plus, w+plus)[0] = input.at < Vec3b >(h, w)[0];
			}
		}

		for (int h = 0; h < plus ; h++) {
			for (int w = 0; w < input.cols; w++) {
				padding1.at<Vec3b>(h, w+plus)[2] = input.at < Vec3b >(0, w)[2];
				padding1.at<Vec3b>(h, w+plus)[1] = input.at < Vec3b >(0, w)[1];
				padding1.at<Vec3b>(h, w+plus)[0] = input.at < Vec3b >(0, w)[0];

				padding1.at<Vec3b>(h + (input.rows - 1), w + plus)[2] = input.at < Vec3b >(input.rows-1,w)[2];
				padding1.at<Vec3b>(h + (input.rows - 1), w + plus)[1] = input.at < Vec3b >(input.rows - 1, w)[1];
				padding1.at<Vec3b>(h + (input.rows - 1), w + plus)[0] = input.at < Vec3b >(input.rows - 1, w)[0];
			}
		}

		for (int h = 0; h < input.rows; h++) {
			for (int w = 0; w < plus; w++) {
				padding1.at<Vec3b>(h + plus, w)[2] = input.at < Vec3b >(h, 0)[2];
				padding1.at<Vec3b>(h + plus, w)[1] = input.at < Vec3b >(h, 0)[1];
				padding1.at<Vec3b>(h + plus, w)[0] = input.at < Vec3b >(h, 0)[0];

				padding1.at<Vec3b>(h + plus, w+input.cols-1)[2] = input.at < Vec3b >(h, input.cols - 1)[2];
				padding1.at<Vec3b>(h + plus, w+input.cols - 1)[1] = input.at < Vec3b >(h, input.cols - 1)[1];
				padding1.at<Vec3b>(h + plus, w+input.cols - 1)[0] = input.at < Vec3b >(h, input.cols - 1)[0];
			}
		}

		for (int h = 0; h < plus; h++) {
			for (int w = 0; w < plus; w++) {
				padding1.at<Vec3b>(h, w)[2] = input.at < Vec3b >(0, 0)[2];
				padding1.at<Vec3b>(h, w)[1] = input.at < Vec3b >(0, 0)[1];
				padding1.at<Vec3b>(h, w)[0] = input.at < Vec3b >(0, 0)[0];

				padding1.at<Vec3b>(h + input.rows - 1, w)[2] = input.at < Vec3b >(input.rows - 1, 0)[2];
				padding1.at<Vec3b>(h + input.rows - 1, w)[1] = input.at < Vec3b >(input.rows - 1, 0)[1];
				padding1.at<Vec3b>(h + input.rows - 1, w)[0] = input.at < Vec3b >(input.rows - 1, 0)[0];

				padding1.at<Vec3b>(h, w+input.cols - 1)[2] = input.at < Vec3b >(0, input.cols - 1)[2];
				padding1.at<Vec3b>(h, w+ input.cols - 1)[1] = input.at < Vec3b >(0, input.cols - 1)[1];
				padding1.at<Vec3b>(h, w+ input.cols - 1)[0] = input.at < Vec3b >(0, input.cols - 1)[0];

				padding1.at<Vec3b>(h + input.rows - 1, w+ input.cols - 1)[2] = input.at < Vec3b >(input.rows - 1, input.cols - 1)[2];
				padding1.at<Vec3b>(h + input.rows - 1, w+ input.cols - 1)[1] = input.at < Vec3b >(input.rows - 1, input.cols - 1)[1];
				padding1.at<Vec3b>(h + input.rows - 1, w+ input.cols - 1)[0] = input.at < Vec3b >(input.rows - 1, input.cols - 1)[0];
			}
		}
		return padding1;
	}


	if (input.channels() == 1)
	{
		for (int h = 0; h < input.rows; h++) {
			for (int w = 0; w < input.cols; w++) {
				padding2.at<uchar>(h+plus, w+plus) = input.at <uchar >(h, w);
			}
		}
		for (int h = 0; h < plus; h++) {
			for (int w = 0; w < input.cols; w++) {
				padding2.at<uchar>(h, w + plus) = input.at < uchar >(0, w);
				
				padding2.at<uchar>(h + input.rows - 1, w + plus) = input.at <uchar>(input.rows - 1, w);
			}
		}
		for (int h = 0; h < input.rows; h++) {
			for (int w = 0; w < plus; w++) {
				padding2.at<uchar>(h + plus, w) = input.at < uchar >(h, 0);

				padding2.at<uchar>(h + plus, w + input.cols - 1) = input.at < uchar>(h, input.cols - 1);

			}
		}
		for (int h = 0; h <  plus; h++) {
			for (int w = 0; w < plus; w++) {
				padding2.at<uchar>(h, w) = input.at <uchar >(0, 0);

				padding2.at<uchar>(h + input.rows - 1, w) = input.at < uchar >(input.rows - 1, 0);

				padding2.at<uchar>(h, w + input.cols - 1) = input.at < uchar>(0, input.cols - 1);

				padding2.at<uchar>(h + input.rows - 1, w + input.cols - 1) = input.at <uchar >(input.rows - 1, input.cols - 1);
			}
		}
		return padding2;
	}
}



void lp(Mat input, int height, int width, int n)
{
	int **mask;
	mask = (int**)malloc(sizeof(int*)*n);
	for (int i = 0; i < n; i++)
	{
		mask[i] = (int*)malloc(sizeof(int)*n);
	}

	if (n == 3) {
	
		int mask3[3][3] = { {0,-1,0},{-1,4,-1},{0,-1,0} };

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {

				mask[i][j] = mask3[i][j];
			}

		}

	}
	if (n == 5) {
		int mask5[5][5] = { {0,0,-1,0,0},{0.-1,2,-1.0},{-1,-2,16,-2,-1},{0. - 1,2,-1.0},{0,0,-1,0,0} };
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				mask[i][j] = mask5[i][j];
			}
		}
	}
	

	int plus = n / 2;
	Mat output1(input.rows, input.cols, CV_8UC3);//rgb output
	Mat output2(input.rows, input.cols, CV_8UC1);//gray output
	Mat output3(input.rows, input.cols, CV_8UC1);//gray output


	Mat padding1(input.rows + 2 * plus, input.cols + 2 * plus, CV_8UC3);
	Mat padding2(input.rows + 2 * plus, input.cols + 2 * plus, CV_8UC1);

	padding(input, padding1, padding2, n);

	if (input.channels() == 3) 
	{

		for (int h = 0; h < height ; h++)
		{
			for (int w = 0; w < width ; w++)
			{
				int r = 0;
				int g = 0;
				int b =0;
				for (int mh = 0; mh < n; mh++)
				{
					for (int mw = 0; mw < n; mw++)
					{
						r += (int)padding1.at<Vec3b>(h+mh, w+mw)[2]* mask[mh][mw];
						g += (int)padding1.at<Vec3b>(h+mh, w+mw)[1]* mask[mh][mw];
						b += (int)padding1.at<Vec3b>(h+mh, w+mw)[0]* mask[mh][mw];
					}
				}
				if (n == 5) 
				{
					if ((r + g + b) > 4500) 
					{
						output1.at<Vec3b>(h, w)[2] = input.at<Vec3b>(h, w)[2];
						output1.at<Vec3b>(h, w)[1] = input.at<Vec3b>(h, w)[1];
						output1.at<Vec3b>(h, w)[0] = input.at<Vec3b>(h, w)[0];
					}
					else 
					{
						output1.at<Vec3b>(h, w)[2] = 0;
						output1.at<Vec3b>(h, w)[1] = 0;
						output1.at<Vec3b>(h, w)[0] = 0;

					}
				}
				if (n == 3)
				{
					if ((r + g + b) > 50)
					{
						output1.at<Vec3b>(h, w)[2] = input.at<Vec3b>(h, w)[2];
						output1.at<Vec3b>(h, w)[1] = input.at<Vec3b>(h, w)[1];
						output1.at<Vec3b>(h, w)[0] = input.at<Vec3b>(h, w)[0];
					}
					else
					{
						output1.at<Vec3b>(h, w)[2] = 0;
						output1.at<Vec3b>(h, w)[1] = 0;
						output1.at<Vec3b>(h, w)[0] = 0;
					}
				}
			}
		}
		for (int i = 0; i < n; i++) 
		{
			free(mask[i]);
		}
		free(mask);

		imshow("원본 칼라 이미지", input);
		imshow("라플라시안 필터 컬러", output1);
		waitKey(0);
	}


	if (input.channels() == 1) {

		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				int gray = 0;
				for (int mh = 0; mh < n; mh++)
				{
					for (int mw = 0; mw < n; mw++)
					{
						gray += (int)padding2.at<uchar>(h+mh, w+mw)*mask[mh][mw];
					}
				}
		
				if (n == 3) {
					if (gray > 30)
						output2.at<uchar>(h, w) = 255;
					else if (gray < 30)
						output2.at<uchar>(h, w) = 0;

				}
			
				if (n == 5) {
					if (gray > 1550)
						output2.at<uchar>(h, w) = 255;
					else if (gray < 1550)
						output2.at<uchar>(h, w) = 0;

				}
				
			}
		}
		
		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				output3.at<uchar>(h, w) = output2.at<uchar>(h, w);

			}
		}
		for (int i = 0; i < n; i++) {
			free(mask[i]);
		}
		free(mask);
		imshow("원본이미지", input);
		imshow("라플라시안 필터 흑백", output3);
		waitKey(0);
	}

}


//이동 평균 필터

void maf(Mat input,  int height, int width, int n)
{
	//Mat mask(n, n, CV_8UC1,Scalar(0));//filtermask
	int plus = n / 2;
	Mat output1(input.rows, input.cols, CV_8UC3);//rgb output
	Mat output2(input.rows, input.cols, CV_8UC1);//gray output

	Mat padding1(input.rows + 2 * plus, input.cols + 2 * plus, CV_8UC3);
	Mat padding2(input.rows + 2 * plus, input.cols + 2 * plus, CV_8UC1);

	padding(input, padding1, padding2, n);
	
	if (input.channels() == 3) 
	{
		for (int h = 0; h < height; h++)
		{
		
			for (int w = 0; w < width ; w++)
			{
				int r = 0;
				int g = 0;
				int b = 0;
				for (int mh = h; mh < h+n; mh++)
				{
					for (int mw = w; mw < w+n; mw++)
					{
						r += padding1.at<Vec3b>(mh, mw)[2];
						g += padding1.at<Vec3b>(mh, mw)[1];
						b += padding1.at<Vec3b>(mh, mw)[0];
					}
				} 
				//printf_s("%d		%d		%d\n\n\n", r,g,b);
				output1.at<Vec3b>(h, w)[2] = r / (n*n);
				output1.at<Vec3b>(h, w)[1] = g / (n*n);
				output1.at<Vec3b>(h, w)[0] = b / (n*n);

			}
			//printf_s("%d\n",h);
		}
		imshow("원본 이미지", input);
		imshow("평균 필터 컬러", output1);
		waitKey(0);
	}


	if (input.channels() == 1) {

		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				int gray = 0;
				for (int mh = h; mh < h+ n; mh++)
				{
					for (int mw = w; mw <w+ n; mw++)
					{
						gray += padding2.at<uchar>(mh, mw);
					}
				}
				output2.at<uchar>(h,w) = gray / (n * n);
			}
		
		}
		imshow("원본 이미지", input);
		imshow("평균필터 흑백", output2);
		waitKey(0);
	}
	
}

void main()
{
	Mat input1 = imread("lena128.jpg");
	Mat input2=imread("lena.jpg");

	Mat input_gray1,input_gray2;
	cvtColor(input1, input_gray1, COLOR_BGR2GRAY);
	cvtColor(input2, input_gray2, COLOR_BGR2GRAY);

	int n;//mask size
	printf_s("size of a filter mask:");
	scanf_s("%d", &n);

	//maf(input1, input1.rows, input1.cols, n);
	//maf(input_gray1, input1.rows, input1.cols, n);//moving average filter은 3x3만
	lp(input2, input2.rows, input2.cols, n);
	//lp(input_gray2, input2.rows, input2.cols, n);
	return;

}