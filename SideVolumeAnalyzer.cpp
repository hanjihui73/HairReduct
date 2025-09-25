#include "SideVolumeAnalyzer.h"

SideVolumeAnalyzer::SideVolumeAnalyzer(void)
{
	_guideLine = 0.4;
	_hair.open("data\\strands00014.txt");

#if defined(AFTER)
	deform();
#endif
}

SideVolumeAnalyzer::~SideVolumeAnalyzer(void)
{

}

double SideVolumeAnalyzer::smoothKernel(double r2, double h)
{
	return fmax(1.0f - r2 / (h*h), 0.0f);
}

void SideVolumeAnalyzer::deform(void)
{
	double before = 25.89;
	double after = 32.29;
	double scale = after / before;	
	printf("scale : %f\n", scale);

	int numStrands = (int)_hair._strands.size();
	for (auto &s : _hair._strands) {
		for (auto &p : s) {
			vec3 cp(0.5, p.y(), 0.25);
			//auto w = 1.0;
			auto w = smoothKernel((p - cp).length(), 2.0);
			auto grad = p - cp;
			p = cp + (grad * scale * w);
		}
	}
}

void SideVolumeAnalyzer::draw(void)
{
	_hair.draw();
	drawGuideLine();
}

void SideVolumeAnalyzer::drawGuideLine(void)
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glLineWidth(3.0);
	glColor3f(1.0f, 0.0f, 0.0f);
	glBegin(GL_LINES);
	glVertex3f(0.0, _guideLine, 0.0);
	glVertex3f(1.0, _guideLine, 0.0);
	glVertex3f(0.0, _guideLine, 0.5);
	glVertex3f(1.0, _guideLine, 0.5);
	glVertex3f(0.0, _guideLine, 0.0);
	glVertex3f(0.0, _guideLine, 0.5);
	glVertex3f(1.0, _guideLine, 0.0);
	glVertex3f(1.0, _guideLine, 0.5);
	glEnd();
	glLineWidth(1.0);
	glEnable(GL_LIGHTING);
	glPopMatrix();
}