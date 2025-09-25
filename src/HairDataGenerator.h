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
// 방향 회전 표기
enum class TurnSign : int { Straight = 0, CW = 1, CCW = 2 };
// Back 내부 좌/우 꼬리 표기
enum class TailSide : int { None = 0, Left = 1, Right = 2 };

class HairDataGenerator
{
public:
	vec3					_minB;
	vec3					_maxB;
	vec3					_offset;
	float					_scaling;
	vector<vector<vec3>>	_strands; // 머리카락 -> 헤어 입자
	vector<vec3>			_colors;
	vector<int>          _groupIds;  // per-strand group id (HairGroup as int)
	float                _zFront = 0.5f + 0.30f * 0.5f; // 기본: 앞쪽 30%
	float                _zBack = 0.5f - 0.30f * 0.5f; // 기본: 뒤쪽 30%
	float                _alphaFB = 0.30f;              // 앞/뒤 폭 비율

	float _longAxisWorld = 1.0f;                         // 정규화거 cm로 환산할 때 쓰일 예정
	vector<vector<int>> _segCuts;                        // 어디서 끊겼는지(0,15,30 이런식으로..)
	vector<vector<uint8_t>> _segLabel;                   // 세그먼트 라벨(Front/Back/Left/Right) = 각 구간의 방향 비율
	vector<vector<vec3>> _segColor;                      // 세그먼트 색
	vector<vector<array<float, 4>>> _segHist;             //{pF, pB, pL, pR} 비율(전후 비교해서 분포가 크면 갈라졌다 하려고)
	vector<int> _finalGroupId;                                //가닥 최종 그룹키(오버라이드 결과)

	std::vector<int> _destLabelPerStrand;   // [si] = 최종 목적지(F/B/L/R → 0/1/2/3)
	std::vector<int> _turnSignPerStrand;    // [si] = TurnSign (0/1/2) : 회전
	std::vector<int> _tailSidePerStrand;    // [si] = TailSide (0/1/2) : back,좌,우
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

	void    computeFrontBackThreshold(float alpha = 0.30f);	//정규화 좌표를 기준으로 앞/뒤 임계값을 한번에 계산하는거
	int  classifyPoint4(vec3 p) const;  //한 점을 앞/뒤/좌/우 중 하나로 분류하는거
	void    buildGroupsAndColors();	//그룹 -> 색 팔레트 매핑

	void buildSegmentsByLength(float target_cm = 1.8f, int minSamples = 8);		//잘라서 -_segmentcut만드는거(이후 모든 단계가 seg단위로 동작 가능)
	void profileSegmentsWithClassify4();                       // 세그 히스토그램 & 초기 라벨(_segHist[si][seg] = {pF, pB, pL, pR} 이거 make)
	void detectAndCommitSplits(float diffTh = 0.40f, float persist_cm = 1.5f);		//분기 후보로 측정하는거
	void decideDestinationAndTurn(float tailRatio = 0.40f,
		float tailAvgTh = 0.55f,
		float runLenCm = 1.5f,
		float thetaTurnDeg = 45.0f);		//끝 부분이 어디로 갔는지, turn방향판정해서 속성 정보 뽑아내는 함수
	void buildFinalGroupKeyPerStrand();                       // 가닥 최종 그룹키 산출 + 오버라이드
	void bakeFinalGroupColors();                               // 최종 그룹키 → 팔레트 색
	void smoothSegments(float minLen_cm = 1.0f);
	void logFinalGroupColorSummary(const std::vector<int>& finalKeys,
		const std::vector<vec3>& colors);
	//void collapseRareSubgroups(int minCount = 17); //희귀 하위 그룹을 상위로 승격/흡수
	void absorbSmallGroupsByKNN_TipAnchored(
		int   minDestCount = 12,	//이 값 미만에 속한 가닥만 후보
		float radius = 0.06f,	//팁 앵커 주변 반경 r(정규화햇을 때 0.06임)
		int   K = 24,	//이웃 수(반경 내 24개)
		float voteTh = 0.65f,	//다수결 최소 득표율로 0.65이상표를 받아야 당선(?)
		float crossFBTh = 0.70f,	//Front/Back 경계 넘길 때 이웃의 반대 면 비율(예: 0.70f) 이상만 허용
		float tailRatio = 0.40f		//팁 앵커를 잡을 tail 구간 비율(예: 0.40f)
	);
	// 그룹별 대표선(=메모이드 추출)
	int extractRepresentativeStrand(
		const std::vector<int>& strandIndices,
		float tailRatio = 0.40f,
		int   S = 5,
		float w_pos = 0.2f,
		float w_dir = 0.3f,
		float w_shape = 0.5f
	);
	// 렌더 토글
	void setRepOnly(bool on) { _repOnly = on; }
	void toggleRepOnly() { _repOnly = !_repOnly; }
	// 대표선(메도이드) 1개씩 뽑기: 그룹키 -> 대표 가닥 인덱스
	std::unordered_map<int, int> computeRepresentatives(
		int   minCount = 50,   // 이 개수 이상인 그룹만 대표선 추출
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

	// 대표선/가닥 형태 비교에 쓰는 요약 기술자
	struct RepDesc {
		std::vector<vec3> pts;  // tail 구간에서 샘플한 S개 포인트
		vec3 centroid;          // pts의 중심
		vec3 tipDir;            // 끝 방향(마지막 두 점으로 단위벡터)
	};

	// 작은 그룹을 대표선 유사도(위치+끝방향+곡선모양)로 "통째" 병합
	void mergeSmallGroupsByRepSimilarity(
		int   minLargeCount = 50,   // 이 수 이상이면 "큰 그룹"
		float tailRatio = 0.40f,// tail 구간 비율(대표선/Desc 뽑을 때)
		int   S = 5,    // tail 구간에서 곡선 샘플 개수
		float w_pos = 0.2f, // 위치 가중치
		float w_dir = 0.3f, // 끝방향 가중치
		float w_shape = 0.5f, // 곡선모양 가중치
		float dirGateDeg = 45.0f,// 1차 게이팅(끝방향 최대 각)
		float Dmax = 1.0f, // 절대 거리 임계(최종 안전장치)
		float margin = 0.15f,// 1등과 2등 거리 차 여유(상대 안전장치)
		bool  strictFB = true  // Front↔Back 교차 시 더 보수적으로
	);


private:
	bool _repOnly = false;                        // ON이면 대표선만 그림
	std::unordered_set<int> _repStrands;          // 대표 가닥 인덱스 모음
};

#endif
