#include "HashTable.h"

HashTable::HashTable()
{
}

HashTable::HashTable(int res)
{
	_res = res;
	_particles = Alloc3D<vector<Particle*>>(res, res, res);
}

HashTable::~HashTable()
{
	Free3D(_particles);
}

void HashTable::sort(vector<vector<vec3>> &strands)
{
	for (int i = 0; i < _res; i++) {
		for (int j = 0; j < _res; j++) {
			for (int k = 0; k < _res; k++) {
				_particles[i][j][k].clear();
			}
		}
	}

	int size = strands.size();
	for(int n = 0; n < size; n++) {
		auto &s = strands[n];
		auto p = s[0];
		int i = (int)fmax(0, fmin(_res - 1, _res*p[0]));
		int j = (int)fmax(0, fmin(_res - 1, _res*p[1]));
		int k = (int)fmax(0, fmin(_res - 1, _res*p[2]));
		_particles[i][j][k].push_back(new Particle(n, p));
	}
}

vector<Particle*> HashTable::getNeigbors(int i, int j, int k, int w, int h, int d)
{
	vector<Particle*> res;
	for (int si = i - w; si <= i + w; si++) {
		for (int sj = j - h; sj <= j + h; sj++) {
			for (int sk = k - d; sk <= k + d; sk++) {
				if (si < 0 || si > _res - 1 || sj < 0 || sj > _res - 1 || sk < 0 || sk > _res - 1) continue;
				for (int a = 0; a < (int)_particles[si][sj][sk].size(); a++) {
					auto p = _particles[si][sj][sk][a];
					res.push_back(p);
				}
			}
		}
	}
	return res;
}
