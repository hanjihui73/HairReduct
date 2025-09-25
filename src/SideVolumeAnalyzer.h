#ifndef __SIDE_VOLUME_ANALYZER_H__
#define __SIDE_VOLUME_ANALYZER_H__

#pragma once
#include "HairDataGenerator.h"

class SideVolumeAnalyzer
{
public:
	HairDataGenerator	_hair;
	double				_guideLine;
public:
	SideVolumeAnalyzer(void);
	~SideVolumeAnalyzer(void);
public:
	void	deform(void);
	double	smoothKernel(double r2, double h);
public:
	void	draw(void);
	void	drawGuideLine(void);
};

#endif
