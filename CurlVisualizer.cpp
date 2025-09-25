#include "CurlVisualizer.h"

CurlVisualizer::CurlVisualizer(void)
{
	_hashTable = new HashTable(64);

	_hair.open("data\\strands00512.txt");
	//_hair.open("data\\strands00001.txt");
	//_hair.open("data\\strands00501.txt"); 
}

CurlVisualizer::~CurlVisualizer(void)
{
}

vec3 CurlVisualizer::scalarToColor(double val)
{
	double map[5][3] = { { 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };	//Red->Blue
	auto v = val;
	if (val > 1.0) v = 1.0; if (val < 0.0) v = 0.0; v *= 4.0;
	int low = (int)floor(v), high = (int)ceil(v);
	auto t = v - low;
	vec3 color;
	color.x((map[low][0])*(1 - t) + (map[high][0])*t);
	color.y((map[low][1])*(1 - t) + (map[high][1])*t);
	color.z((map[low][2])*(1 - t) + (map[high][2])*t);
	return color;
}

void CurlVisualizer::draw(void)
{
	_hair.draw();
}