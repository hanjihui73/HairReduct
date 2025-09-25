#include <stdio.h>
#include <Windows.h> 
#include "GL\glut.h"
#include <math.h>
#include "CurlyStrandAnalyzer.h"
#include "BackVolumeAnalyzer.h"
#include "SideVolumeAnalyzer.h"
#include "RootVolumeAnalyzer.h"
#include "TopVolumeAnalyzer.h"
#include "CurlVisualizer.h"

int _width = 800;
int _height = 600;
float _zoom = 1.439998f;
float _rot_x = 0.0f;
float _rot_y = 0.001f;
float _trans_x = 0.0f;
float _trans_y = 0.0f;
int _last_x = 0;
int _last_y = 0;
unsigned char _buttons[3] = { 0 };

int _frame = 0;
int _renderMode = 1;
bool _simulation = false;

CurlVisualizer _curlViz; 

void Init(void)
{
	glEnable(GL_DEPTH_TEST);
}

void Domain(void)
{
	glDisable(GL_LIGHTING);
	glColor3f(1.0f, 1.0f, 1.0f);
	glPushMatrix();
	glTranslatef(0.5, 0.5, 0.5);
	glutWireCube(1.0);
	glPopMatrix();
	glEnable(GL_DEPTH_TEST);
}

void Draw(void)
{
	glDisable(GL_LIGHTING);
	
	switch (_renderMode)
	{
	case 1:
		_curlViz.draw();
		break;
		break;
	case 2:
		break;
	case 3:
		break;
	}
	//Domain();
	glDisable(GL_LIGHTING);
}

void Display(void)
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glDisable(GL_CULL_FACE);
	glTranslatef(0, 0, -_zoom);
	glTranslatef(_trans_x, _trans_y, 0);
	glRotatef(_rot_x, 1, 0, 0);
	glRotatef(_rot_y, 0, 1, 0);
	glTranslatef(-0.5, -0.5, -0.5);
	Draw();
	glutSwapBuffers();
}

void Reshape(int w, int h)
{
	if (w == 0) {
		h = 1;
	}
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (float)w / h, 0.1, 100);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void Keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'Q':
	case 'q':
		exit(0);
	case '1':
		_renderMode = 1;
		break;
	case '2':
		_renderMode = 2;
		break;
	case ' ':
		_simulation = !_simulation;
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void Mouse(int button, int state, int x, int y)
{
	_last_x = x;
	_last_y = y;

	switch (button)
	{
	case GLUT_LEFT_BUTTON:
		_buttons[0] = state == GLUT_DOWN ? 1 : 0;
		break;
	case GLUT_MIDDLE_BUTTON:
		_buttons[1] = state == GLUT_DOWN ? 1 : 0;
		break;
	case GLUT_RIGHT_BUTTON:
		_buttons[2] = state == GLUT_DOWN ? 1 : 0;
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void Motion(int x, int y)
{
	int diff_x = x - _last_x;
	int diff_y = y - _last_y;
	_last_x = x;
	_last_y = y;

	if (_buttons[2]) {
		_zoom -= (float)0.02f * diff_x;
	}
	else if (_buttons[1]) {
		_trans_x += (float)0.02f * diff_x;
		_trans_y -= (float)0.02f * diff_y;
	}
	else if (_buttons[0]) {
		_rot_x += (float)0.2f * diff_y;
		_rot_y += (float)0.2f * diff_x;
	}
	glutPostRedisplay();
}

void Capture(int endFrame)
{
	if (_frame == 0 || _frame % 2 == 0) {
		static int index = 0;
		char filename[100];
		sprintf_s(filename, "capture\\capture-%d.bmp", index);
		BITMAPFILEHEADER bf;
		BITMAPINFOHEADER bi;
		unsigned char *image = (unsigned char*)malloc(sizeof(unsigned char)*_width*_height * 3);
		FILE *file;
		fopen_s(&file, filename, "wb");
		if (image != NULL) {
			if (file != NULL) {
				glReadPixels(0, 0, _width, _height, 0x80E0, GL_UNSIGNED_BYTE, image);
				memset(&bf, 0, sizeof(bf));
				memset(&bi, 0, sizeof(bi));
				bf.bfType = 'MB';
				bf.bfSize = sizeof(bf) + sizeof(bi) + _width * _height * 3;
				bf.bfOffBits = sizeof(bf) + sizeof(bi);
				bi.biSize = sizeof(bi);
				bi.biWidth = _width;
				bi.biHeight = _height;
				bi.biPlanes = 1;
				bi.biBitCount = 24;
				bi.biSizeImage = _width * _height * 3;
				fwrite(&bf, sizeof(bf), 1, file);
				fwrite(&bi, sizeof(bi), 1, file);
				fwrite(image, sizeof(unsigned char), _height*_width * 3, file);
				fclose(file);
			}
			free(image);
		}
		index++;
		if (index == endFrame) {
			//	exit(0);
		}
	}
}

void Idle(void)
{
	if (_simulation) {
		//Capture(1000);
		//printf("frame : %d\n", _frame);
		_frame++;
	}
	glutPostRedisplay();
}


int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(_width, _height);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Strand Analyzer");
	glutDisplayFunc(Display);
	glutReshapeFunc(Reshape);
	glutMouseFunc(Mouse);
	glutMotionFunc(Motion);
	glutKeyboardFunc(Keyboard);
	glutIdleFunc(Idle);
	Init();
	glutMainLoop();
	return 0;
}