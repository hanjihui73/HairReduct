#include "RootVolumeAnalyzer.h"

RootVolumeAnalyzer::RootVolumeAnalyzer(void)
{
	_guideLine = 0.77;
	_offset = 0.18;
	_hair.open("data\\strands00512.txt");

#if defined(AFTER)
	deform();
#endif
}

RootVolumeAnalyzer::~RootVolumeAnalyzer(void)
{
}

double RootVolumeAnalyzer::smoothKernel(double r2, double h)
{
	return fmax(1.0f - r2 / (h*h), 0.0f);
}

void RootVolumeAnalyzer::deform(void)
{
	double before = 2.33;
	double after = 3.04;
	double scale = after / before;
	printf("scale : %f\n", scale);
	
	int numStrands = (int)_hair._strands.size();
	for (auto &s : _hair._strands) {
		for (auto &p : s) {
			vec3 cp(0.5, _guideLine, 0.5);
			auto w = smoothKernel((p - cp).length(), 2.0);
			auto grad = p - cp;
			p = cp + (grad * scale * w);
		}
	}
}

void RootVolumeAnalyzer::draw(void)
{
	_hair.drawCylinder(_guideLine);
	drawGuideLine();
}

void RootVolumeAnalyzer::drawGuideLine(void)
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

	auto top = _guideLine + _offset;
	glVertex3f(0.0, top, 0.0);
	glVertex3f(1.0, top, 0.0);
	glVertex3f(0.0, top, 1.0);
	glVertex3f(1.0, top, 1.0);
	glVertex3f(0.0, top, 0.0);
	glVertex3f(0.0, top, 1.0);
	glVertex3f(1.0, top, 0.0);
	glVertex3f(1.0, top, 1.0);
	glEnd();
	glLineWidth(1.0);
	glEnable(GL_LIGHTING);
	glPopMatrix();
}
