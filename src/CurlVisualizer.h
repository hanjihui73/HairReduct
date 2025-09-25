#ifndef __CURL_VISUALIZER_H__
#define __CURL_VISUALIZER_H__

#pragma once
#include "HairDataGenerator.h"
#include "HashTable.h"

class CurlVisualizer
{
public:
	CurlVisualizer(void);
	~CurlVisualizer(void);
public:
	HashTable			*_hashTable; // ´ÙÀ½ÁÖ¿¡ ¼³¸í
	HairDataGenerator	_hair;
public:
	vec3	scalarToColor(double val);
public:
	void	draw(void);
};

#endif
