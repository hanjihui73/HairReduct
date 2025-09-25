#include "TopVolumeAnalyzer.h"
#include <iterator>
#include <random>
#include <algorithm>

TopVolumeAnalyzer::TopVolumeAnalyzer(void)
{
	_guideLine = 0.77;
	_hair.open("data\\strands00512.txt");

#if defined(BEFORE)
	deform();
#endif	
}

TopVolumeAnalyzer::~TopVolumeAnalyzer(void)
{
}

void TopVolumeAnalyzer::deform(void)
{
	vec3 cp(0.5, 1.0, 0.5);
	double before = 146.85;
	double after = 134.48;
	double scale = after / before;
	auto r = 0.25;
	printf("scale : %f\n", scale);
	vector<int> candidates;
	int size = 0;

	vector<vector<vec3>> strands;
	int numStrands = (int)_hair._strands.size();
	for(int i = 0; i < numStrands; i++) {
		auto &s = _hair._strands[i];
		auto p = s[0];
		//if (p(1) > _guideLine) 
		{
			size++;
		}
		auto sdf = pow(p.x() - cp.x(), 2.0) + pow(p.z() - cp.z(), 2.0) - (r*r);
		if (sdf < 0.0) {
			candidates.push_back(i);
		} else {
			strands.push_back(s);
		}
	}

	int reducedSize = (int)((double)size * scale);
	printf("%d -> %d\n", size, reducedSize);

	vector<int> samples;
	random_device rd;	
	shuffle(candidates.begin(), candidates.end(), default_random_engine(rd()));
	int remain = reducedSize - strands.size();

	for (int i = 0; i < remain; i++) {
		int id = candidates[i];
		strands.push_back(_hair._strands[id]);
	}
	_hair._strands.swap(strands);
	printf("%d\n", _hair._strands.size());
}

void TopVolumeAnalyzer::draw(void)
{
	_hair.drawCylinder(_guideLine);
	drawGuideLine();
}

void TopVolumeAnalyzer::drawGuideLine(void)
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glLineWidth(3.0);
	glColor3f(1.0f, 0.0f, 0.0f);
	glBegin(GL_LINES);
	glVertex3f(0.0, _guideLine, 0.0);
	glVertex3f(1.0, _guideLine, 0.0);
	glVertex3f(0.0, _guideLine, 1.0);
	glVertex3f(1.0, _guideLine, 1.0);
	glVertex3f(0.0, _guideLine, 0.0);
	glVertex3f(0.0, _guideLine, 1.0);
	glVertex3f(1.0, _guideLine, 0.0);
	glVertex3f(1.0, _guideLine, 1.0);
	glEnd();
	glLineWidth(1.0);
	glEnable(GL_LIGHTING);
	glPopMatrix();
}