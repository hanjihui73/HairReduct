#include "BackVolumeAnalyzer.h"

BackVolumeAnalyzer::BackVolumeAnalyzer(void)
{	
	_guideLine = 0.4;

	_hair.open("data\\strands00001.txt");

	//_hair.open("data\\strands00500.txt");
	//_hair.open("data\\strands00501.txt");

#if defined(AFTER)
	deform();
#endif
}

BackVolumeAnalyzer::~BackVolumeAnalyzer(void)
{
}

double BackVolumeAnalyzer::smoothKernel(double r2, double h)
{
	return fmax(1.0f - r2 / (h*h), 0.0f);
}

void BackVolumeAnalyzer::deform(void)
{
	double before = 25.89;
	double after = 32.29;
	double scale = after / before;
	printf("scale : %f\n", scale);
	
	for (auto &s : _hair._strands) {
		for (auto &p : s) {
			vec3 cp(0.5, p.y(), 0.5);
			auto w = smoothKernel((p - cp).length(), 2.0);
			//w = 1.0;
			auto grad = p - cp;
			p = cp + (grad * scale * w);
		}
	}
}

void BackVolumeAnalyzer::draw(void)
{
	_hair.draw();
	drawGuideLine();
}

void BackVolumeAnalyzer::drawGuideLine(void)
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
