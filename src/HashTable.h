#ifndef __HASH_TABLE_H__
#define __HASH_TABLE_H__

#pragma once
#include "Vec3.h"
#include <vector>

using namespace std;

class Particle
{
public:
	int		_id; // strand id
	vec3	_pos;
public:
	Particle(void) {}
	Particle(int id, vec3 pos)
	{
		_id = id;
		pos = pos;
	}
	~Particle(void) {}
};


template <typename T> T ***Alloc3D(int w, int h, int d)
{
	T *** Buffer = new T **[w + 1];
	for (int i = 0; i < w; i++) {
		Buffer[i] = new T*[h + 1];
		for (int j = 0; j < h; j++) {
			Buffer[i][j] = new T[d];
		}
		Buffer[i][h] = NULL;
	}
	Buffer[w] = NULL;
	return Buffer;
}

template <class T> void Free3D(T ***ptr)
{
	for (int i = 0; ptr[i] != NULL; i++) {
		for (int j = 0; ptr[i][j] != NULL; j++) delete[] ptr[i][j];
		delete[] ptr[i];
	}
	delete[] ptr;
}

class HashTable
{
public:
	int						_res;
	int						_type;
	vector<Particle*>	***_particles;
public:
	HashTable();
	HashTable(int res);
	~HashTable();
public:
	void				sort(vector<vector<vec3>> &strands);
	vector<Particle*>	getNeigbors(int i, int j, int k, int w, int h, int d);
};

#endif
