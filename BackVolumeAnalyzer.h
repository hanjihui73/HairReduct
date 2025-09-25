#ifndef __BACK_VOLUME_ANALYZER_H__
#define __BACK_VOLUME_ANALYZER_H__

#pragma once
#include "HairDataGenerator.h"

class BackVolumeAnalyzer
{
public:
	HairDataGenerator	_hair;
	double				_guideLine;
public:
	BackVolumeAnalyzer(void);
	~BackVolumeAnalyzer(void);
public:
	void	deform(void);
	double	smoothKernel(double r2, double h);
public:
	void	draw(void);
	void	drawGuideLine(void);
};

#endif