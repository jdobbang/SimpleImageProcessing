#include<opencv2\opencv.hpp>
#include<stdio.h>
#include<math.h>
using namespace cv;

void YUV(Mat img_color, int height, int width, Mat f, Mat f2)
{

	for (int y = 0; y < height; y++)
	{

		for (int x = 0; x < width; x++)
		{

			uchar b = img_color.at<Vec3b>(y, x)[0];//r,g,b pixel
			uchar g = img_color.at<Vec3b>(y, x)[1];
			uchar r = img_color.at<Vec3b>(y, x)[2];

			uchar U = -0.169*r - 0.331*g + 0.499*b + 128;
			uchar V = 0.499*r - 0.418*g - 0.081*b + 128;

			if  ((105 <= U &&  U <= 125) &&  (130 <= V && V <= 160)) // (112.761 <= U &&  U <= 119.736) &&  (137.792 <= V && V <= 153.086)
			{
				f.at<Vec3b>(y, x)[2] = r;
				f.at<Vec3b>(y, x)[1] = g;
				f.at<Vec3b>(y, x)[0] = b;
			}
			else {
				f.at<Vec3b>(y, x)[2] = 0;
				f.at<Vec3b>(y, x)[1] = 0;
				f.at<Vec3b>(y, x)[0] = 0;
			}

			if ((105 <= U && U <= 125) && (130 <= V && V <= 160)) // (112.761 <= U &&  U <= 119.736) &&  (137.792 <= V && V <= 153.086)
			{
				f2.at<Vec3b>(y, x)[2] = 255;
				f2.at<Vec3b>(y, x)[1] = 255;
				f2.at<Vec3b>(y, x)[0] = 255;
			}
			else {
				f2.at<Vec3b>(y, x)[2] = 0;
				f2.at<Vec3b>(y, x)[1] = 0;
				f2.at<Vec3b>(y, x)[0] = 0;
			}

		}
	}



}



void RGB(Mat img_color, int height, int width, Mat f4, Mat f5)
{

	Mat f3(height, width, CV_8UC3);//face3



	for (int y = 0; y < height; y++)
	{

		for (int x = 0; x < width; x++)
		{
			uchar b = img_color.at<Vec3b>(y, x)[0];//r,g,b pixel
			uchar g = img_color.at<Vec3b>(y, x)[1];
			uchar r = img_color.at<Vec3b>(y, x)[2]; 

			double sum = (double)r + (double)g + (double)b;
			
			if (x == 50 && y == 50)
				printf("%f \n", sum);
			
			f3.at<Vec3b>(y, x)[0] = (uchar)(255*(double)b / sum );
			f3.at<Vec3b>(y, x)[1] = (uchar)(255*(double)g / sum );
			f3.at<Vec3b>(y, x)[2] = (uchar)(255*(double)r / sum );

			uchar b1 = f3.at<Vec3b>(y, x)[0];
			uchar g1 = f3.at<Vec3b>(y, x)[1];
			uchar r1 = f3.at<Vec3b>(y, x)[2];

			uchar l = (74 - b1)*(74 - b1) + (82 - g1)*(82 - g1) + (98 - r1)*(98 - r1);
			uchar rate = 20;

			if (l <= rate)
			{
				f5.at<Vec3b>(y, x)[0] = b;
				f5.at<Vec3b>(y, x)[1] = g;
				f5.at<Vec3b>(y, x)[2] = r;
			}
			 else if(l > rate) {
				f5.at<Vec3b>(y, x)[0] = 0;
				f5.at<Vec3b>(y, x)[1] = 0;
				f5.at<Vec3b>(y, x)[2] = 0;
			}
			
			if (l <= rate)
			{
				f4.at<uchar>(y, x) = 255;
				
			}
			else
				f4.at<uchar>(y, x) = 0;
			}
	}


}

void main() {

	Mat input = imread("face.jpg", IMREAD_COLOR);

	int height = input.rows;
	int width = input.cols;
	Mat f(height, width, CV_8UC3);//face
	Mat f2(height, width, CV_8UC3);//face2
	Mat f4(height, width, CV_8UC1);//face4
	Mat f5(height, width, CV_8UC3);//face5

	YUV(input, height, width,f,f2);
	RGB(input, height, width,f4,f5);

	imshow("original face", input);
	imshow("YUV face color", f);
	imshow("YUV face b/w", f2);
	imshow("RGB normalized face b/w", f4);
	imshow("RGB normalized face color", f5);
	waitKey(0);


}
