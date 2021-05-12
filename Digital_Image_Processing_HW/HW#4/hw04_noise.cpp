#include "stdio.h"
#include <opencv2\opencv.hpp>
#include <Windows.h>
#include <cstdlib>
#include <math.h>
#define PI 3.14
#define X_MAX 256
#define Y_MAX 256
using namespace std;
using namespace cv;

// PADDING �Լ�
Mat padding(Mat input, int height, int width, int n)
{
	int plus = n / 2;
	Mat padding(height + 2 * plus, width + 2 * plus, CV_8UC1);

	for (int h = 0; h < height; h++) {
		for (int w = 0; w < width; w++) {
			padding.at<uchar>(h + plus, w + plus) = input.at<uchar>(h, w);
		}
	}

	for (int h = 0; h < plus; h++) {
		for (int w = 0; w < width; w++) {
			padding.at<uchar>(h, w + plus) = input.at<uchar>(0, w);

			padding.at<uchar>(h + height - 1, w + plus) = input.at<uchar>(height - 1, w);

		}
	}
	for (int h = 0; h < height; h++) {
		for (int w = 0; w < plus; w++) {
			padding.at<uchar>(h + plus, w) = input.at<uchar>(h, 0);

			padding.at<uchar>(h + plus, w + width - 1) = input.at<uchar>(h, width - 1);
		}
	}

	for (int h = 0; h < plus; h++) {
		for (int w = 0; w < plus; w++) {
			padding.at<uchar>(h, w) = input.at<uchar>(0, 0);

			padding.at<uchar>(h + height - 1, w) = input.at<uchar>(height - 1, 0);

			padding.at<uchar>(h, w + width - 1) = input.at<uchar>(0, width - 1);

			padding.at<uchar>(h + height - 1, w + width - 1) = input.at<uchar>(height - 1, width - 1);
		}
	}
	return padding;
}

//psnr �����Լ�
void psnr(Mat original, Mat after, int height, int width) {
	double mse = 0;
	for (int h = 0; h < height; h++) {
		for (int w = 0; w < width; w++) {
			mse += pow(original.at<uchar>(h, w) - after.at<uchar>(h, w), 2) / (height*width);
		}
	}
	double temp = pow(255, 2) / mse;
	double psnr = 10 * log(temp);
	printf("%lf", psnr);
}

//����þ� ������

float GetNoise(float *PDF, int nLength) {
	int n;
	int Center = nLength / 2;
	float fRand, fComp, fTemp = 0;
	float x = 0, fDiff;
	float* CDF = new float[nLength];

	CDF[0] = 0;

	fRand = (float)rand() / (RAND_MAX + 1);

	for (n = 1; n < nLength; n++)
	{
		CDF[n] = (PDF[n] + PDF[n - 1]) / 2 + CDF[n - 1];
		fDiff = fRand - CDF[n];
		if (fDiff < 0)
		{
			x = ((float)n - Center);
			break;
		}
	}
	fComp = (fRand - CDF[n - 1]) / (CDF[n] - CDF[n - 1]);

	delete[] CDF;
	return x + fComp;
}

void GetGaussianPDF(float* EmptyPDF, int nLength, float fMean, float fStDev)
{//����þ� �������� Ȯ���е� �Լ�
	int n;
	int Center = nLength / 2;// 256/2 == 128
	float x;

	for (n = 0; n < nLength; n++) {
		x = (float)(n - Center);
		EmptyPDF[n] = (1 / ((float)sqrt(2 * PI)*fStDev)) * exp(-pow(x - fMean, 2) / (2 * fStDev*fStDev));//�Լ� ����
	}
}
void InputGaussianNoise(Mat input, Mat output, int height, int width, float fMean, float fStdev)
{
	float fTemp = 0, fPDF[256] = { 0.0f };
	GetGaussianPDF(fPDF, 256, fMean, fStdev);
	srand(GetTickCount());

	for (int h = 0; h < height; h++) {
		for (int w = 0; w < width; w++) {
			fTemp = (float)(input.at<uchar>(h, w)) + GetNoise(fPDF, 256);// ������ ���ϱ�
			output.at<uchar>(h, w) = static_cast<unsigned char>(fTemp);

		}
	}

}

//SAULT AND PEPPER NOISE
void InputSaltPepperNoise(Mat input, Mat output, int height, int width, float fSProb, float fPProb)
{
	float Low = fSProb; //���� ���� probability
	float High = 1.0f - fPProb;//�ְ� ���� probability
	float fRand;//����

	srand(GetTickCount());

	for (int h = 0; h < height; h++)
	{
		for (int w = 0; w < width; w++) {
			fRand = ((float)rand() / RAND_MAX);

			if (fRand < Low) {
				output.at<uchar>(h, w) = 255;

			}//������ ���������� ������ �Ͼ��
			else if (fRand > High)//������ �ְ������� ������ ������
			{
				output.at<uchar>(h, w) = 0;
			}
			else output.at<uchar>(h, w) = input.at<uchar>(h, w);//�� �ܿ��� input�̹��� �״�� ����
		}
	}
}

//MEAN FILTERS
void MeanFilter(Mat input, int height, int width, int filtersize, Mat original) {
	int nTemp, nTemp2;
	int padsize = (int)(filtersize / 2);
	Mat out1(height, width, CV_8UC1);
	Mat out2(height, width, CV_8UC1);
	Mat out3(height, width, CV_8UC1);
	Mat out4(height, width, CV_8UC1);
	Mat padding2(height + 2 * padsize, width + 2 * padsize, CV_8UC1);

	padding2 = padding(input, height, width, filtersize);

	//arithmetic
	for (int h = 0; h < height; h++)
	{
		for (int w = 0; w < width; w++)
		{
			nTemp = 0;
			for (int mh = h; mh < h + filtersize; mh++)
			{
				for (int mw = w; mw < w + filtersize; mw++)
				{
					nTemp += padding2.at<uchar>(mh, mw);
				}
			}
			out1.at<uchar>(h, w) = nTemp / (filtersize*filtersize);
		}
	}

	imshow("arithmetic mean filter", out1);
	printf("\narithmetic mean filter's PSNR:");
	psnr(original, out1, height, width);
	waitKey(0);

}

//median filter
void medainfilter(Mat input, int height, int width, int filtersize, Mat original)
{
	int nTemp, nTemp2;
	int padsize = (int)(filtersize / 2);
	Mat out(height, width, CV_8UC1);
	Mat padding2(height + 2 * padsize, width + 2 * padsize, CV_8UC1);

	padding2 = padding(input, height, width, filtersize);

	//median filter
	int *array, temp;
	array = (int*)malloc(sizeof(int)*(filtersize*filtersize));
	for (int h = 0; h < height; h++)
	{
		for (int w = 0; w < width; w++)
		{
			for (int mh = h; mh < h + filtersize; mh++)
			{
				for (int mw = w; mw < w + filtersize; mw++)
				{
					nTemp = padding2.at<uchar>(mh, mw);
					//printf("%d\n", filtersize*(mh - h) + (mw - w));
					array[filtersize*(mh - h) + (mw - w)] = nTemp;
				}
			}
			for (int i = 0; i < filtersize*filtersize; i++)    // ����� ������ŭ �ݺ�
			{
				for (int j = 0; j < filtersize*filtersize - 1; j++)   // ����� ���� - 1��ŭ �ݺ�
				{
					if (array[j] > array[j + 1])          // ���� ����� ���� ���� ����� ���� ���Ͽ�
					{                                 // ū ����
						temp = array[j];
						array[j] = array[j + 1];
						array[j + 1] = temp;            // ���� ��ҷ� ����
					}
				}
			}
			out.at<uchar>(h, w) = array[(filtersize)*(filtersize) / 2];// Ȧ������ �ڵ������� ��� �� ����
			//printf("%u\n", out.at<uchar>(h, w));
		}
	}
	imshow("median filter", out);
	printf("median filter to sault and pepper noise image : ");
	psnr(original, out, height, width);
	waitKey(0);
}


void main(void)
{
	int i, j;
	FILE *in, *out;//���� I/O ������
	Mat original(256, 256, CV_8UC1);// output Mat
	Mat output_g(256, 256, CV_8UC1);//����þ� output
	Mat output_sp(256, 256, CV_8UC1);
	char in_data[X_MAX][Y_MAX];
	char out_data[X_MAX][Y_MAX];

	fopen_s(&in, "lena.raw", "rb");//file open
	if (in == NULL)
	{
		printf("File not found!!\n");
		return;
	}
	fread(in_data, sizeof(uchar), X_MAX * Y_MAX, in);//file �б�
	fclose(in);

	for (i = 0; i < Y_MAX; i++)
	{
		for (j = 0; j < X_MAX; j++)

		{
			original.at<uchar>(i, j) = in_data[i][j];//raw ������ �Է��ϱ�
		}
	}

	//������ ÷�� �� return
	InputGaussianNoise(original, output_g, 256, 256, 15, 70);
	InputSaltPepperNoise(original,output_sp,256, 256, 0.25, 0.25);

	
	imshow("raw image", original);
	imshow("gaussain", output_g);
	
	imshow("saltpepper", output_sp);
	waitKey(0);

	//������ ÷�� �̹���
	MeanFilter(output_g, 256, 256, 5, original);

	medainfilter(output_sp, 256, 256, 3, original);
	
	return;
}
