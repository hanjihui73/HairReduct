#ifndef __CURLY_STRAND_GENERATOR_H__
#define __CURLY_STRAND_GENERATOR_H__

#pragma once
#include "Vec3.h"
#include <vector>
#include <time.h> 
#include "Common.h"

using namespace std;

class CurlyStrandGenerator
{
public:
	int				_type;
	int				_numParticles; // number of particles
	vector<vec3>	_points;
	double			_heightRatio;
public:
	CurlyStrandGenerator(void);
	~CurlyStrandGenerator(void);
public:
	void	spiralStrand(double length, double radius, vec3 root);
	void	straightStrand(double length, double radius, vec3 root);
public:
	void	compute(void);
	void	makeScene(void);
public:
	void	draw(void);
	void	drawStrand(void);
	void	drawCylinder(void);
};

#endif