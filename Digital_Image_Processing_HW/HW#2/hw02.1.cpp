#include<opencv2/opencv.hpp>
#include<stdio.h>

using namespace cv;
using namespace std;

void RGB(Mat img_color, int height, int width)
{

	Mat img_gray(height, width, CV_8UC1);

	uchar *data_input = img_color.data;

		for (int y = 0; y < height ; y++)
		{
			uchar *data_output = img_gray.data;

			for (int x = 0; x < width; x++)
			{
				uchar b = data_input[y*width * 3 + x * 3];
				uchar g = data_input[y*width * 3 + x * 3+1];
				uchar r = data_input[y*width * 3 + x * 3+2];

				data_output[width*y + x] = (r + g + b) / 3.0;

			}
		}
		imshow("color", img_color);
		imshow("grayscale", img_gray);
		waitKey(0);
		return;
	
}



void main()
{
	Mat img_color = imread("test.jpg", IMREAD_COLOR);
	int height = img_color.rows;
	int width = img_color.cols;

	RGB(img_color, height, width);

	return;
}
