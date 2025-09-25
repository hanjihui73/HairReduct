#include "CurlyStrandGenerator.h"

CurlyStrandGenerator::CurlyStrandGenerator(void)
{
	_type = 0;
	_heightRatio = 1.0;
	srand((unsigned int)time(NULL));
}

CurlyStrandGenerator::~CurlyStrandGenerator(void)
{
}

void CurlyStrandGenerator::straightStrand(double length, double radius, vec3 root)
{
	auto angle = length / ((_numParticles - 1) * radius);
	auto step = length / (_numParticles - 1);
	root(0) += radius;

	vec3 dirx(1, 0, 0);
	vec3 diry(0, -1, 0);
	vec3 dirz(0, 0, 1);

	auto tip = (diry * length) + root;
	auto jx = sqrt(((double)rand() / RAND_MAX)) * radius;

	double min = -0.1;
	double max = 0.1;
	auto w = (((double)rand() / RAND_MAX) - 0.5) * 0.03;
	angle += w;

	for (int i = 0; i < _numParticles; i++) {
		auto curl_p = (dirx * cos(i * angle) + dirz * sin(i * angle) - dirx) * radius + root;
		curl_p += diry * i * step;
		auto t = static_cast<double>(i) / (_numParticles - 1);
		auto straight_p = tip.lerp(root, t);
		auto pos = curl_p.lerp(straight_p, 0.1);
		_points.push_back(pos);
	}
}

void CurlyStrandGenerator::spiralStrand(double length, double radius, vec3 root)
{
	auto angle = length / ((_numParticles - 1) * radius);
	auto step = length / (_numParticles - 1);
	root(0) += radius;

	vec3 dirx(1, 0, 0);
	vec3 diry(0, -1, 0);
	vec3 dirz(0, 0, 1);

	auto tip = (diry * length) + root;	
	auto jx = sqrt(((double)rand() / RAND_MAX)) * radius;

	double min = -0.1;
	double max = 0.1;
	auto w = (((double)rand() / RAND_MAX) - 0.5) * 0.03;
	angle += w;

	for (int i = 0; i < _numParticles; i++) {
		auto curl_p = (dirx * cos(i * angle) + dirz * sin(i * angle) - dirx) * radius + root;
		curl_p += diry * i * step;
		const auto t = static_cast<double>(i) / (_numParticles - 1);
		auto straight_p = tip.lerp(root, t);
		auto pos = curl_p.lerp(straight_p, t);
		_points.push_back(pos);
	}
}

void CurlyStrandGenerator::makeScene(void)
{
	compute();
}

void CurlyStrandGenerator::compute(void)
{
	double radius = 0.01;
	double cx = 0.5;
	double cz = 0.5;
	_numParticles = 100;
	int numStrands = 100;
	_points.clear();

	for (int i = 0; i < numStrands; i++) {
		double angle = ((double)rand() / RAND_MAX) * 2 * PI;
		double distance = sqrt(((double)rand() / RAND_MAX)) * radius;
		double x = cx + distance * cos(angle);
		double z = cz + distance * sin(angle);
		if (_type == 0) {
			straightStrand(_heightRatio, 0.05, vec3(x, 1.0, z));
		}
		else if (_type == 1) {
			spiralStrand(_heightRatio, 0.05, vec3(x, 1.0, z));
		}
	}
}

void CurlyStrandGenerator::draw(void)
{
	//drawStrand();
	drawCylinder();
}

void CurlyStrandGenerator::drawStrand(void)
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glColor3f(1.0f, 1.0f, 1.0f);
	int numStrands = _points.size() / _numParticles;
	for (int i = 0; i < numStrands; i++) {
		glBegin(GL_LINE_STRIP);
		for (int j = 0; j < _numParticles; j++) {
			int id = i * _numParticles + j;
			glVertex3f(_points[id].x(), _points[id].y(), _points[id].z());
		}
		glEnd();
	}
	glEnable(GL_LIGHTING);
	glPopMatrix();
}

void CurlyStrandGenerator::drawCylinder(void)
{
	glPushMatrix();
	
	float diffuse[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
	
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glShadeModel(GL_SMOOTH);
	static double color[13][3] =
	{ { 0.4901960784313725, 0.6392156862745098, 0.9725490196078431 },
	{ 0.992156862745098, 0.5294117647058824, 0.8274509803921569 },
	{ 0.7176470588235294, 0.9450980392156863, 0.5411764705882353 },
	{ 0.9254901960784314, 0.8352941176470588, 0.5686274509803922 },
	{ 0.5372549019607843, 0.8235294117647059, 0.5450980392156863 },
	{ 0.5843137254901961, 0.7098039215686275, 0.6980392156862745 },
	{ 0.4705882352941176, 0.7490196078431373, 0.7254901960784314 },
	{ 0.5529411764705882, 0.8274509803921569, 0.7176470588235294 },
	{ 0.992156862745098, 0.9607843137254902, 0.4862745098039216 },
	{ 0.9333333333333333, 0.8509803921568627, 0.8392156862745098 },
	{ 0.7647058823529412, 0.6274509803921569, 0.603921568627451 },
	{ 0.5568627450980392, 0.5019607843137255, 0.4588235294117647 },
	{ 0.807843137254902, 0.3764705882352941, 0.4745098039215686 } };
	srand(123456);

	double r = 0.001;
	int numStrands = _points.size() / _numParticles;
	
	for (int i = 0; i < numStrands; i++) {
		int id = 13 * ((double)rand() / RAND_MAX);
		int mode = id % 13;
		//glColor3f(color[mode][0], color[mode][1], color[mode][2]);
		diffuse[0] = color[mode][0];
		diffuse[1] = color[mode][1];
		diffuse[2] = color[mode][2];
		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
		static int sides = 12;
		double max_radius = r;
		double min_radius = r * 0.5;
		for (int j = 0; j < _numParticles; j++) {
			int id = i * _numParticles + j;
			if (j < _numParticles - 1) {
				int id = i * _numParticles + j;
				int id1 = i * _numParticles + (j + 1);
				vec3 a(_points[id].x(), _points[id].y(), _points[id].z());
				vec3 b(_points[id1].x(), _points[id1].y(), _points[id1].z());
				int i1, i2;
				vec3 ortho1, ortho2;
				ortho1 = (b - a).ortho();
				ortho1.normalize();
				double weight = (double)j / (double)_numParticles;
				//double radius = (weight*min_radius) + (1.0 - weight)*max_radius;
				double radius = r;
				ortho1 *= radius;
				ortho2 = (b - a).cross(ortho1);
				ortho2.normalize();
				ortho2 *= radius;
				glBegin(GL_QUADS);
				for (i1 = 0; i1 < sides; i1++) {
					i2 = (i1 + 1) % sides;
					double theta1 = i1 * (2.0 * PI) / sides;
					double theta2 = i2 * (2.0 * PI) / sides;
					vec3 v[4];
					v[0] = a + ortho1 * cos(theta1) + ortho2 * sin(theta1);
					v[1] = b + ortho1 * cos(theta1) + ortho2 * sin(theta1);
					v[2] = b + ortho1 * cos(theta2) + ortho2 * sin(theta2);
					v[3] = a + ortho1 * cos(theta2) + ortho2 * sin(theta2);
					// compute normal
					vec3 v0, v1, v2;
					v0 = v[0];
					v1 = v[1];
					v2 = v[2];
					vec3 va, vb;
					va = v1 - v0;
					vb = v2 - v0;
					vec3 normal(va.cross(vb));
					normal.normalize();
					normal.inverse();
					glNormal3f((GLfloat)normal.x(), (GLfloat)normal.y(), (GLfloat)normal.z());
					for (int j = 3; j >= 0; j--) {
						glVertex3f((GLfloat)v[j].x(), (GLfloat)v[j].y(), (GLfloat)v[j].z());
					}
				}
				glEnd();
			}
		}
	}
	glEnable(GL_LIGHTING);
	diffuse[0] = 1.0f;
	diffuse[1] = 1.0f;
	diffuse[2] = 1.0f;
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
	glPopMatrix();
}