#ifndef __HAIR_DATA_GENERATOR_H__
#define __HAIR_DATA_GENERATOR_H__

#pragma once
#include "Vec3.h"
#include <iostream>
#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include "Common.h"

using namespace std;



enum class HairGroup : int { Front = 0, Back = 1, Left = 2, Right = 3 };
// ���� ȸ�� ǥ��
enum class TurnSign : int { Straight = 0, CW = 1, CCW = 2 };
// Back ���� ��/�� ���� ǥ��
enum class TailSide : int { None = 0, Left = 1, Right = 2 };

class HairDataGenerator
{
public:
	vec3					_minB;
	vec3					_maxB;
	vec3					_offset;
	float					_scaling;
	vector<vector<vec3>>	_strands; // �Ӹ�ī�� -> ��� ����
	vector<vec3>			_colors;
	vector<int>          _groupIds;  // per-strand group id (HairGroup as int)
	float                _zFront = 0.5f + 0.30f * 0.5f; // �⺻: ���� 30%
	float                _zBack = 0.5f - 0.30f * 0.5f; // �⺻: ���� 30%
	float                _alphaFB = 0.30f;              // ��/�� �� ����

	float _longAxisWorld = 1.0f;                         // ����ȭ�� cm�� ȯ���� �� ���� ����
	vector<vector<int>> _segCuts;                        // ��� �������(0,15,30 �̷�������..)
	vector<vector<uint8_t>> _segLabel;                   // ���׸�Ʈ ��(Front/Back/Left/Right) = �� ������ ���� ����
	vector<vector<vec3>> _segColor;                      // ���׸�Ʈ ��
	vector<vector<array<float, 4>>> _segHist;             //{pF, pB, pL, pR} ����(���� ���ؼ� ������ ũ�� �������� �Ϸ���)
	vector<int> _finalGroupId;                                //���� ���� �׷�Ű(�������̵� ���)

	std::vector<int> _destLabelPerStrand;   // [si] = ���� ������(F/B/L/R �� 0/1/2/3)
	std::vector<int> _turnSignPerStrand;    // [si] = TurnSign (0/1/2) : ȸ��
	std::vector<int> _tailSidePerStrand;    // [si] = TailSide (0/1/2) : back,��,��
public:
	HairDataGenerator(void);
	~HairDataGenerator(void);
public:
	void	open(const char *filename);
	void	normalizePos(void);
public:
	void	draw(void);
	void	drawPoint(void);
	void	drawStrand(void);

	void    computeFrontBackThreshold(float alpha = 0.30f);	//����ȭ ��ǥ�� �������� ��/�� �Ӱ谪�� �ѹ��� ����ϴ°�
	int  classifyPoint4(vec3 p) const;  //�� ���� ��/��/��/�� �� �ϳ��� �з��ϴ°�
	void    buildGroupsAndColors();	//�׷� -> �� �ȷ�Ʈ ����

	void buildSegmentsByLength(float target_cm = 1.8f, int minSamples = 8);		//�߶� -_segmentcut����°�(���� ��� �ܰ谡 seg������ ���� ����)
	void profileSegmentsWithClassify4();                       // ���� ������׷� & �ʱ� ��(_segHist[si][seg] = {pF, pB, pL, pR} �̰� make)
	void detectAndCommitSplits(float diffTh = 0.40f, float persist_cm = 1.5f);		//�б� �ĺ��� �����ϴ°�
	void decideDestinationAndTurn(float tailRatio = 0.40f,
		float tailAvgTh = 0.55f,
		float runLenCm = 1.5f,
		float thetaTurnDeg = 45.0f);		//�� �κ��� ���� ������, turn���������ؼ� �Ӽ� ���� �̾Ƴ��� �Լ�
	void buildFinalGroupKeyPerStrand();                       // ���� ���� �׷�Ű ���� + �������̵�
	void bakeFinalGroupColors();                               // ���� �׷�Ű �� �ȷ�Ʈ ��
	void smoothSegments(float minLen_cm = 1.0f);
	void logFinalGroupColorSummary(const std::vector<int>& finalKeys,
		const std::vector<vec3>& colors);
	//void collapseRareSubgroups(int minCount = 17); //��� ���� �׷��� ������ �°�/���
	void absorbSmallGroupsByKNN_TipAnchored(
		int   minDestCount = 12,	//�� �� �̸��� ���� ���ڸ� �ĺ�
		float radius = 0.06f,	//�� ��Ŀ �ֺ� �ݰ� r(����ȭ���� �� 0.06��)
		int   K = 24,	//�̿� ��(�ݰ� �� 24��)
		float voteTh = 0.65f,	//�ټ��� �ּ� ��ǥ���� 0.65�̻�ǥ�� �޾ƾ� �缱(?)
		float crossFBTh = 0.70f,	//Front/Back ��� �ѱ� �� �̿��� �ݴ� �� ����(��: 0.70f) �̻� ���
		float tailRatio = 0.40f		//�� ��Ŀ�� ���� tail ���� ����(��: 0.40f)
	);
	// �׷캰 ��ǥ��(=�޸��̵� ����)
	int extractRepresentativeStrand(
		const std::vector<int>& strandIndices,
		float tailRatio = 0.40f,
		int   S = 5,
		float w_pos = 0.2f,
		float w_dir = 0.3f,
		float w_shape = 0.5f
	);
	// ���� ���
	void setRepOnly(bool on) { _repOnly = on; }
	void toggleRepOnly() { _repOnly = !_repOnly; }
	// ��ǥ��(�޵��̵�) 1���� �̱�: �׷�Ű -> ��ǥ ���� �ε���
	std::unordered_map<int, int> computeRepresentatives(
		int   minCount = 50,   // �� ���� �̻��� �׷츸 ��ǥ�� ����
		float tailRatio = 0.40f,
		int   S = 5,
		float w_pos = 0.2f,
		float w_dir = 0.3f,
		float w_shape = 0.5f
	);

	std::unordered_map<int, std::vector<int>> computeRepresentativesMulti(
		int   kPerGroup = 4,
		int   minCount = 50,
		float tailRatio = 0.40f,
		int   S = 5,
		float w_pos = 0.2f,
		float w_dir = 0.3f,
		float w_shape = 0.5f
	);

	// ��ǥ��/���� ���� �񱳿� ���� ��� �����
	struct RepDesc {
		std::vector<vec3> pts;  // tail �������� ������ S�� ����Ʈ
		vec3 centroid;          // pts�� �߽�
		vec3 tipDir;            // �� ����(������ �� ������ ��������)
	};

	// ���� �׷��� ��ǥ�� ���絵(��ġ+������+����)�� "��°" ����
	void mergeSmallGroupsByRepSimilarity(
		int   minLargeCount = 50,   // �� �� �̻��̸� "ū �׷�"
		float tailRatio = 0.40f,// tail ���� ����(��ǥ��/Desc ���� ��)
		int   S = 5,    // tail �������� � ���� ����
		float w_pos = 0.2f, // ��ġ ����ġ
		float w_dir = 0.3f, // ������ ����ġ
		float w_shape = 0.5f, // ���� ����ġ
		float dirGateDeg = 45.0f,// 1�� ������(������ �ִ� ��)
		float Dmax = 1.0f, // ���� �Ÿ� �Ӱ�(���� ������ġ)
		float margin = 0.15f,// 1��� 2�� �Ÿ� �� ����(��� ������ġ)
		bool  strictFB = true  // Front��Back ���� �� �� ����������
	);


private:
	bool _repOnly = false;                        // ON�̸� ��ǥ���� �׸�
	std::unordered_set<int> _repStrands;          // ��ǥ ���� �ε��� ����
};

#endif
