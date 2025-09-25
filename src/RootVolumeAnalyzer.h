#ifndef __ROOT_VOLUME_ANALYZER_H__
#define __ROOT_VOLUME_ANALYZER_H__

#pragma once
#include "HairDataGenerator.h"

class RootVolumeAnalyzer
{
public:
	RootVolumeAnalyzer(void);
	~RootVolumeAnalyzer(void);
public:
	HairDataGenerator	_hair;
	double				_offset;
	double				_guideLine;
public:
	void	deform(void);
	double	smoothKernel(double r2, double h);
public:
	void	draw(void);
	void	drawGuideLine(void);
};

#endif

