#ifndef __TOP_VOLUME_ANALYZER_H__
#define __TOP_VOLUME_ANALYZER_H__

#pragma once
#include "HairDataGenerator.h"

class TopVolumeAnalyzer
{
public:
	TopVolumeAnalyzer(void);
	~TopVolumeAnalyzer(void);
public:
	HairDataGenerator	_hair;
	double				_guideLine;
public:
	void	deform(void);
public:
	void	draw(void);
	void	drawGuideLine(void);
};

#endif
