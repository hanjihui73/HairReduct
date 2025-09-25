#ifndef __CURLY_STRAND_ANALYZER_H__
#define __CURLY_STRAND_ANALYZER_H__

#pragma once
#include "CurlyStrandGenerator.h"
#include "include/opencv2/opencv.hpp"

struct AABB
{
	int _width;
	int _height;
};

class CurlyStrandAnalyzer
{
public:
	CurlyStrandGenerator	_strands;
	AABB					_afterAttr;
	AABB					_beforeAttr;
public:
	CurlyStrandAnalyzer(void);
	~CurlyStrandAnalyzer(void);
public:
	void		init(void);
	void		draw(void);
	void		thinning(cv::Mat& im);
	void		thinningIteration(cv::Mat& im, int iter);
};

#endif

