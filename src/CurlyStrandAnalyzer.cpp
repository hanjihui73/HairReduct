#include "CurlyStrandAnalyzer.h"

//#define IMAGE_BASED 

#define BEFORE_HAIRS

CurlyStrandAnalyzer::CurlyStrandAnalyzer(void)
{
	init();
}

CurlyStrandAnalyzer::~CurlyStrandAnalyzer(void)
{
}

void CurlyStrandAnalyzer::init(void)
{
	int dy = 0;
	double dyRatio = 0.0;

#ifdef IMAGE_BASED
	FILE *fp0, *fp1;
	fopen_s(&fp0, "data\\Result1.txt", "r");
	fopen_s(&fp1, "data\\Result2.txt", "r");
	fscanf_s(fp0, "%d %d", &_beforeAttr._width, &_beforeAttr._height);
	fscanf_s(fp1, "%d %d", &_afterAttr._width, &_afterAttr._height);
	fclose(fp0);
	fclose(fp1);

	//int dx = _beforeAttr._width - _afterAttr._width;
#else
	_beforeAttr._height = (int)39.83;
	_afterAttr._height = (int)32.63;
#endif

	dy = _beforeAttr._height - _afterAttr._height;
	dyRatio = dy / (double)_beforeAttr._height;
	
#if (defined BEFORE)
	_strands._type = 0;
	_strands._heightRatio = 1.0;
#endif

#if (defined AFTER)
	_strands._type = 1;
	_strands._heightRatio = 1.0 - dyRatio;
#endif

	_strands.makeScene();
}

void CurlyStrandAnalyzer::thinningIteration(cv::Mat& im, int iter)
{
	cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);
	for (int i = 1; i < im.rows - 1; i++) {
		for (int j = 1; j < im.cols - 1; j++) {
			uchar p2 = im.at<uchar>(i - 1, j);
			uchar p3 = im.at<uchar>(i - 1, j + 1);
			uchar p4 = im.at<uchar>(i, j + 1);
			uchar p5 = im.at<uchar>(i + 1, j + 1);
			uchar p6 = im.at<uchar>(i + 1, j);
			uchar p7 = im.at<uchar>(i + 1, j - 1);
			uchar p8 = im.at<uchar>(i, j - 1);
			uchar p9 = im.at<uchar>(i - 1, j - 1);
			int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
				(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
				(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
				(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
			int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
			int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);
			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) {
				marker.at<uchar>(i, j) = 1;
			}
		}
	}
	im &= ~marker;
}

void CurlyStrandAnalyzer::thinning(cv::Mat& im)
{
	im /= 255;
	cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
	cv::Mat diff;
	do {
		thinningIteration(im, 0);
		thinningIteration(im, 1);
		cv::absdiff(im, prev, diff);
		im.copyTo(prev);
	} while (cv::countNonZero(diff) > 0);
	im *= 255;
}

void CurlyStrandAnalyzer::draw(void)
{
	_strands.draw();
}
