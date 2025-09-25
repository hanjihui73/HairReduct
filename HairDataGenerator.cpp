#include "HairDataGenerator.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>

HairDataGenerator::HairDataGenerator(void)
{
}

HairDataGenerator::~HairDataGenerator(void)
{
}

void HairDataGenerator::open(const char *filename)
{
	_scaling = 100.0f;
	_offset.set(0.0, -185.0, -25.0);

	FILE *fp;
	fopen_s(&fp, filename, "r");
	if (!fp) {
		fprintf(stderr, "Couldn't open %s\n", filename);
		return;
	}

	int numStrands = 0;
	if (!fscanf_s(fp, "%d", &numStrands)) {
		fprintf(stderr, "Couldn't read number of strands\n");
		fclose(fp);
		return;
	}

	//numStrands = 1000;

	_minB.set(std::numeric_limits<float>::infinity());
	_maxB.set(-std::numeric_limits<float>::infinity());

	for (int i = 0; i < numStrands; i++) {
		int nverts = 0;
		float length = 0;
		fscanf_s(fp, "%d", &nverts);
		vector<vec3> verts;
		for (int j = 0; j < nverts; j++) {
			vec3 vert, pre_vert;
			float px, py, pz;
			if (!fscanf_s(fp, "%f %f %f", &px, &py, &pz)) {
				fprintf(stderr, "Couldn't read %d-th vertex in strand %d\n", j, i);
				fclose(fp);
				return;
			}
			if (nverts != 100) continue;
			vert.set(px, py, pz);
			vert *= _scaling;
			vert += _offset;
			if (j != 0) {
				vec3 edge = pre_vert - vert;
				length += edge.length();
			}
			verts.push_back(vert);
			pre_vert = vert;
			_minB.x(fmin(_minB.x(), vert.x()));
			_minB.y(fmin(_minB.y(), vert.y()));
			_minB.z(fmin(_minB.z(), vert.z()));
			_maxB.x(fmax(_maxB.x(), vert.x()));
			_maxB.y(fmax(_maxB.y(), vert.y()));
			_maxB.z(fmax(_maxB.z(), vert.z()));
		}
		if (nverts != 100) continue;

		// 파티클간 평균 길이가 0.2 보다 작은 strand 제외
		if ((length / (float)nverts) < 0.2f) {
			continue;
		}
		_strands.push_back(verts);
	}
	fprintf(stderr, "num. strands : %d\n", _strands.size());
	fclose(fp);
	normalizePos();
	computeFrontBackThreshold(0.20f); //앞/뒤 폭을 좁게 해서 중앙 영역 늘어나게해줌

	// 세그먼트 자르기 + 프로파일 생성
	buildSegmentsByLength(1.2f, 5); //세그먼트를 더 잘게(길이를 줄여서 더 많이 비교하게 되는거)
	profileSegmentsWithClassify4();

	detectAndCommitSplits(0.45f, 1.0f);	//작은 변화도 새로운 그룹으로 인정
	smoothSegments(1.0f);

	decideDestinationAndTurn(0.40f, 0.55f,   // 5단계: 목적지/회전/꼬리좌우 확정
		1.5f, 45.0f);
	buildFinalGroupKeyPerStrand();

	//collapseRareSubgroups(50);          // 후처리: 자잘 그룹 상위로 흡수
	// 이 for문을 주석 처리하거나 it=0으로 유지
	for (int it = 0; it < 1; ++it) {
		mergeSmallGroupsByRepSimilarity(/*minLargeCount=*/100, /*tailRatio=*/0.40f, /*S=*/5,
			/*w_pos=*/0.2f, /*w_dir=*/0.3f, /*w_shape=*/0.5f,
			/*dirGateDeg=*/45.0f, /*Dmax=*/1.2f, /*margin=*/0.20f,
			/*strictFB=*/true);
	}
	bakeFinalGroupColors();
	logFinalGroupColorSummary(_finalGroupId, _colors);
	//buildGroupsAndColors();           // 라벨/색 버퍼 생성
	auto repsK = computeRepresentativesMulti(
		/*kPerGroup=*/7,
		/*minCount=*/10,
		/*tailRatio=*/0.40f,
		/*S=*/5,
		/*w_pos=*/0.2f,
		/*w_dir=*/0.3f,
		/*w_shape=*/0.5f
	);
	setRepOnly(true);
	 
}

// --- group key encode/decode helpers (4bit씩 안전하게 사용)
static inline int encodeGroupKey(int root, int dest, int turn, int tailSide) {
	// root,dest: 0~3 (F/B/L/R), turn: 0~2 (Straight/CW/CCW), tailSide: 0~2 (None/Left/Right)
	return ((tailSide & 0xF) << 12) | ((turn & 0xF) << 8) | ((dest & 0xF) << 4) | (root & 0xF);
}
// 필요하면 디버그용으로 decode도 만들어 쓸 수 있음
static inline int keyRoot(int key) { return  key & 0xF; }
static inline int keyDest(int key) { return (key >> 4) & 0xF; }
static inline int keyTurn(int key) { return (key >> 8) & 0xF; }
static inline int keyTailLR(int key) { return (key >> 12) & 0xF; }

// 대표선/임의 가닥 1개에 대한 Desc(pts, centroid, tipDir) 만들기
// 대표선/임의 가닥 1개에 대한 Desc(pts, centroid, tipDir) 만들기
static HairDataGenerator::RepDesc buildDescForStrand(
	const HairDataGenerator* self, int si, float tailRatio, int S);

// Desc 간 pos+dir+shape 가중합 거리 (dir 각도 라디안도 함께 리턴)
static std::pair<float, float> distPosDirShape(
	const HairDataGenerator::RepDesc& A, const HairDataGenerator::RepDesc& B,
	float w_pos, float w_dir, float w_shape);

static HairDataGenerator::RepDesc buildDescForStrand(
	const HairDataGenerator* self, int si, float tailRatio, int S)
{
	HairDataGenerator::RepDesc d;
	if (!self) return d;
	if (si < 0 || si >= (int)self->_strands.size()) return d;

	const auto& P = self->_strands[si];
	if (P.size() < 2) return d;

	// 세그 경계 필요
	if (si >= (int)self->_segCuts.size()) return d;
	const auto& cuts = self->_segCuts[si];
	if (cuts.size() < 2) return d;

	// 1) tailRatio 구간 시작 세그 찾기 (길이 기준)
	const int segN = (int)cuts.size() - 1;
	std::vector<float> segLenCm(segN, 0.0f);
	float totalLen = 0.0f;
	for (int k = 0; k < segN; ++k) {
		int a = cuts[k], b = cuts[k + 1];
		float acc = 0.0f;
		for (int j = a; j < b; ++j) {
			vec3 dv = P[j + 1] - P[j];
			acc += std::sqrt(dv.x() * dv.x() + dv.y() * dv.y() + dv.z() * dv.z());
		}
		segLenCm[k] = acc * self->_longAxisWorld;
		totalLen += segLenCm[k];
	}
	if (totalLen <= 0) totalLen = 1.0f;

	float need = totalLen * std::max(0.1f, std::min(0.9f, tailRatio));
	int tStart = segN - 1;
	float accTail = 0.0f;
	for (int k = segN - 1; k >= 0; --k) {
		accTail += segLenCm[k];
		tStart = k;
		if (accTail >= need) break;
	}

	// 2) tail 쪽에서부터 S개 세그의 중점 샘플
	int grabbed = 0;
	for (int k = segN - 1; k >= tStart && grabbed < S; --k, ++grabbed) {
		int a = cuts[k], b = cuts[k + 1];
		vec3 mid = (P[a] + P[b]) * 0.5f;
		d.pts.push_back(mid);
	}
	std::reverse(d.pts.begin(), d.pts.end()); // 아래→위 순서

	// 3) centroid
	vec3 c(0, 0, 0);
	for (auto& p : d.pts) c += p;
	if (!d.pts.empty()) c /= (float)d.pts.size();
	d.centroid = c;

	// 4) tipDir
	if (d.pts.size() >= 2) {
		vec3 v = d.pts.back() - d.pts[d.pts.size() - 2];
		double L = std::sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
		if (L > 1e-8) v /= (float)L; else v.set(0.0, 0.0, 1.0);
		d.tipDir = v;
	}
	else {
		d.tipDir.set(0.0, 0.0, 1.0);
	}

	return d;
}

static std::pair<float, float> distPosDirShape(
	const HairDataGenerator::RepDesc& A, const HairDataGenerator::RepDesc& B,
	float w_pos, float w_dir, float w_shape)
{
	// 위치
	float pos = (A.centroid - B.centroid).length();

	// 방향(라디안)
	vec3 ta = A.tipDir, tb = B.tipDir;
	double dotv = ta.x() * tb.x() + ta.y() * tb.y() + ta.z() * tb.z();
	if (dotv > 1.0) dotv = 1.0;
	if (dotv < -1.0) dotv = -1.0;
	float ang = (float)std::acos(dotv);

	// 모양(L2 평균)
	int L = (int)std::min(A.pts.size(), B.pts.size());
	float shape = 0.0f;
	for (int i = 0; i < L; ++i) shape += (A.pts[i] - B.pts[i]).length();
	if (L > 0) shape /= L;

	// 가중합 거리(스케일 정규화가 필요하면 곱/나눗값 추가)
	float D = w_pos * pos + w_dir * ang + w_shape * shape;
	return { D, ang };
}


static inline vec3 hsv2rgb(float h, float s, float v)
{
	// h in [0,1), s,v in [0,1]
	float r = v, g = v, b = v;
	if (s <= 1e-6f) return vec3(r, g, b);
	h = fmodf(h, 1.0f); if (h < 0) h += 1.0f;
	float hf = h * 6.0f;
	int   i = (int)floorf(hf);
	float f = hf - i;
	float p = v * (1.0f - s);
	float q = v * (1.0f - s * f);
	float t = v * (1.0f - s * (1.0f - f));
	switch (i % 6) {
	case 0: r = v; g = t; b = p; break;
	case 1: r = q; g = v; b = p; break;
	case 2: r = p; g = v; b = t; break;
	case 3: r = p; g = q; b = v; break;
	case 4: r = t; g = p; b = v; break;
	case 5: r = v; g = p; b = q; break;
	}
	return vec3(r, g, b);
}

static const char* grpName4(int v) {
	switch (v) {
	case 0: return "Front"; case 1: return "Back";
	case 2: return "Left";  case 3: return "Right"; default: return "?";
	}
}
static const char* turnName(int v) {
    switch (v) { case 0: return "Straight"; case 1: return "CW"; case 2: return "CCW"; default: return "?"; }
}
static const char* tailName(int v) {
    switch (v) { case 0: return "None"; case 1: return "Left"; case 2: return "Right"; default: return "?"; }
}

void HairDataGenerator::normalizePos(void)
{
	auto longAxis = fmax(fmax(fabs(_maxB.x() - _minB.x()), fabs(_maxB.y() - _minB.y())), fabs(_maxB.z() - _minB.z()));
	auto cp = (_maxB + _minB) / 2.0;
	_longAxisWorld = longAxis;	//정규화를 cm환산 스케일로 사용할 떄


	int index = 0;
	int numStrands = (int)_strands.size();
	for (int i = 0; i < numStrands; i++) {
		int numParticles = (int)_strands[i].size();
		for (int j = 0; j < numParticles; j++) {
			// strand
			vec3 pos = _strands[i][j];			
			auto v1 = pos - cp;
			v1 /= longAxis;
			v1 *= 1.0; // scaling
			pos.set(0.5); // new center
			pos += v1;
			_strands[i][j] = vec3((float)pos.x(), (float)pos.y(), (float)pos.z());
		}
	}
}

// === 앞/뒤 임계값: 퍼센타일 기반으로 산출 ===
void HairDataGenerator::computeFrontBackThreshold(float alpha)
{
	_alphaFB = alpha;
	const int N = (int)_strands.size();
	std::vector<float> zroots; zroots.reserve(N);

	// 각 스트랜드의 root z만 수집
	for (auto& s : _strands) {
		if (!s.empty()) {
			zroots.push_back((float)s.front().z());
		}
	}
	if (zroots.empty()) {
		// fallback: 이전 고정식
		const float center = 0.5f, half = 0.5f;
		_zFront = center + half * _alphaFB;
		_zBack = center - half * _alphaFB;
		fprintf(stderr, "[FB] (fallback) alpha=%.2f → zFront=%.3f, zBack=%.3f\n",
			_alphaFB, _zFront, _zBack);
		return;
	}

	std::sort(zroots.begin(), zroots.end());
	auto q = [&](float p)->float {
		// p in [0,1], 선형보간 퍼센타일
		float idx = p * (zroots.size() - 1);
		int i0 = (int)floorf(idx);
		int i1 = std::min((int)zroots.size() - 1, i0 + 1);
		float t = idx - i0;
		return zroots[i0] * (1.0f - t) + zroots[i1] * t;
		};

	// 앞: 상위 alpha → 1-alpha 퍼센타일 / 뒤: 하위 alpha → alpha 퍼센타일
	float pFront = 1.0f - _alphaFB; // 0.70 if alpha=0.30
	float pBack = _alphaFB;        // 0.30 if alpha=0.30
	_zFront = q(pFront);
	_zBack = q(pBack);

	fprintf(stderr,
		"[FB-quantile] alpha=%.2f → pFront=%.2f zFront=%.4f, pBack=%.2f zBack=%.4f (N=%zu)\n",
		_alphaFB, pFront, _zFront, pBack, _zBack, zroots.size());
}


// === 한 점을 4개 그룹(앞/뒤/좌/우) 중 하나로 분류 ===
int HairDataGenerator::classifyPoint4(vec3 p) const
{
	// 우선순위 규칙:
	// 1) 앞머리 최우선: z > _zFront
	// 2) 뒷머리 그다음: z < _zBack
	// 3) 나머지(중앙 띠)는 좌/우: x >= 0.5 → Right, x < 0.5 → Left
	const float pad = 0.03f; // 정규화 좌표에서 여유폭(필요시 0.02~0.05 사이로 튜닝)

	if (p.z() > _zFront - pad) return (int)HairGroup::Front;
	if (p.z() < _zBack + pad) return (int)HairGroup::Back;
	return (p.x() >= 0.5f) ? (int)HairGroup::Right : (int)HairGroup::Left;
}

// === 전 스트랜드 라벨링 + 기본색 매핑 ===
void HairDataGenerator::buildGroupsAndColors()
{
	const int N = (int)_strands.size();
	_groupIds.assign(N, 0);
	_colors.assign(N, vec3(1, 1, 1)); // default: white

	// 기본 팔레트: Front=Red, Back=Blue, Left=Green, Right=Yellow
	auto colorOf = [](int g)->vec3 {
		switch ((HairGroup)g) {
		case HairGroup::Front: return vec3(1, 0, 0); // 빨강
		case HairGroup::Back:  return vec3(0, 0, 1); // 파랑
		case HairGroup::Left:  return vec3(0, 1, 0); // 초록
		case HairGroup::Right: return vec3(1, 1, 0); // 노랑
		}
		return vec3(1, 1, 1);
		};

	int cnt[4] = { 0,0,0,0 };

	for (int si = 0; si < N; ++si)
	{
		const auto& s = _strands[si];
		if (s.empty()) { _groupIds[si] = (int)HairGroup::Back; _colors[si] = colorOf(_groupIds[si]); continue; }

		// root(첫 점)으로 스트랜드 그룹 결정
		vec3 root = s.front();
		int g = classifyPoint4(root);

		_groupIds[si] = g;
		_colors[si] = colorOf(g);
		cnt[g]++;
	}

	fprintf(stderr, "[GROUP] Front=%d, Back=%d, Left=%d, Right=%d (total=%d)\n",
		cnt[(int)HairGroup::Front], cnt[(int)HairGroup::Back],
		cnt[(int)HairGroup::Left], cnt[(int)HairGroup::Right], N);
}

void HairDataGenerator::buildSegmentsByLength(float target_cm, int minSamples)
{
	// 안전장치
	if (_strands.empty()) return;
	if (target_cm <= 0.0f) target_cm = 1.8f;
	if (minSamples < 2)    minSamples = 2;
	if (_longAxisWorld <= 0.0f) _longAxisWorld = 1.0f; // normalizePos에서 세팅되는 값

	// 컨테이너 크기 맞추기/초기화
	const size_t S = _strands.size();
	_segCuts.assign(S, {});      // [si] -> {start0=0, start1, ..., lastEnd=N-1}
	_segHist.assign(S, {});      // 다음 단계에서 채움
	_segLabel.assign(S, {});     // 다음 단계에서 채움

	for (size_t si = 0; si < S; ++si)
	{
		const auto& P = _strands[si];
		const int N = static_cast<int>(P.size());
		if (N < 2) {
			// 점이 1개 이하인 가닥은 단일 세그먼트로 처리
			_segCuts[si] = { 0, std::max(0, N - 1) };
			continue;
		}

		std::vector<int> cuts;
		cuts.reserve(std::max(2, N / 8));
		cuts.push_back(0);               // 첫 세그먼트 시작은 항상 0

		float acc_norm = 0.0f;           // 정규화 좌표에서 누적 길이
		int   lastCut = 0;

		for (int j = 0; j < N - 1; ++j)
		{
			const vec3 d = P[j + 1] - P[j];
			const float seg_len_norm = (P[j + 1] - P[j]).length();
			acc_norm += seg_len_norm;

			const float acc_world = acc_norm * _longAxisWorld;  // cm 환산

			const int segSamples = (j + 1) - lastCut + 1; // [lastCut .. j+1] 포함
			const bool enoughSamples = (segSamples >= minSamples);

			if (acc_world >= target_cm && enoughSamples)
			{
				// j+1을 새로운 세그먼트 시작으로 잡는다
				cuts.push_back(j + 1);
				lastCut = j + 1;
				acc_norm = 0.0f; // 다음 세그 누적 재시작
			}
		}

		// 마지막 포인트(N-1)를 "종료 경계"로 추가 (세그 정의: [cuts[i], cuts[i+1]])
		if (cuts.back() != N - 1)
			cuts.push_back(N - 1);

		// 너무 짧은 꼬리 세그먼트 처리: 마지막 세그 샘플 수가 minSamples 미만이면 병합
		if (cuts.size() >= 2)
		{
			int a = cuts[cuts.size() - 2];  // 마지막 직전 시작
			int b = cuts[cuts.size() - 1];  // 마지막 종료 = N-1
			int lastSegSamples = (b - a + 1);
			if (lastSegSamples < minSamples && cuts.size() > 2)
			{
				// 마지막 시작을 제거하여 바로 앞 세그먼트와 병합
				cuts.erase(cuts.end() - 2);
			}
		}

		// 최소 보장: 적어도 [0, N-1] 두 경계
		if (cuts.size() < 2)
		{
			cuts.clear();
			cuts.push_back(0);
			cuts.push_back(N - 1);
		}

		_segCuts[si] = std::move(cuts);
	}
}

//세그먼트별로 f/b/r/l 비율 히스토그램과 초기 라벨을 만들기
void HairDataGenerator::profileSegmentsWithClassify4()
{
	if (_strands.empty() || _segCuts.empty()) return;

	const size_t S = _strands.size();
	_segHist.assign(S, {});
	_segLabel.assign(S, {});

	for (size_t si = 0; si < S; ++si)
	{
		const auto& P = _strands[si];
		const auto& cuts = _segCuts[si];
		if (P.size() < 1 || cuts.size() < 2) {
			_segHist[si].clear();
			_segLabel[si].clear();
			continue;
		}

		std::vector<std::array<float, 4>> hists;
		std::vector<uint8_t> labels;
		hists.reserve(cuts.size() - 1);
		labels.reserve(cuts.size() - 1);

		for (size_t k = 0; k + 1 < cuts.size(); ++k)
		{
			const int a = cuts[k];
			const int b = cuts[k + 1];
			// [a .. b] (양끝 포함) 범위의 점들로 히스토그램 계산
			int cnt[4] = { 0,0,0,0 };
			int n = 0;

			for (int j = a; j <= b; ++j) {
				const int g = classifyPoint4(P[j]); // z/x 임계 기반 분류
				cnt[g]++; n++;
			}
			// 정규화 (합=1.0)
			std::array<float, 4> hist = { 0,0,0,0 };
			if (n > 0) {
				hist[0] = (float)cnt[0] / n; // Front
				hist[1] = (float)cnt[1] / n; // Back
				hist[2] = (float)cnt[2] / n; // Left
				hist[3] = (float)cnt[3] / n; // Right
			}
			// 초기 라벨 = 다수결 (동률이면 끝점(b) 라벨을 우선)
			int argmax = 0;
			float vmax = hist[0];
			for (int t = 1; t < 4; ++t) {
				if (hist[t] > vmax) { vmax = hist[t]; argmax = t; }
			}
			// 동률 처리: 다수값이 충분히 크지 않거나 여러 개면 끝점 라벨로 보정
			int endLabel = classifyPoint4(P[b]);
			if (vmax < 0.34f) argmax = endLabel;

			hists.push_back(hist);
			labels.push_back((uint8_t)argmax);
		}

		_segHist[si] = std::move(hists);
		_segLabel[si] = std::move(labels);
	}
	for (size_t si = 0; si < std::min((size_t)3, _strands.size()); ++si) {
		printf("[DEBUG] strand %zu: segs=%zu\n", si, _segCuts[si].size() - 1);
		for (size_t k = 0; k < std::min((size_t)3, _segHist[si].size()); ++k) {
			auto h = _segHist[si][k];
			printf("   seg%zu hist F=%.2f B=%.2f L=%.2f R=%.2f  label=%d\n",
				k, h[0], h[1], h[2], h[3], _segLabel[si][k]);
		}
	}
}

static inline int argmax4(const std::array<float, 4>& h) {
	int a = 0; float v = h[0];
	for (int i = 1; i < 4; ++i) { if (h[i] > v) { v = h[i]; a = i; } }
	return a;
}


//연속 세그먼트이 방향 분포차이가 diffTh이상이면 분기후보 => persist_cm이상 연속되면 진짜 분기 확정
void HairDataGenerator::detectAndCommitSplits(float diffTh, float persist_cm)
{
	if (_strands.empty() || _segCuts.empty() || _segHist.empty() || _segLabel.empty()) return;
	if (diffTh <= 0.0f)   diffTh = 0.30f;
	if (persist_cm <= 0)  persist_cm = 1.5f;

	const size_t S = _strands.size();
	int totalSplitsCommitted = 0;

	for (size_t si = 0; si < S; ++si)
	{
		const auto& P = _strands[si];
		const auto& cuts = _segCuts[si];
		auto& hists = _segHist[si];
		auto  labels = _segLabel[si]; // 작업용 복사(끝에 되돌려 씀)

		const int segN = (int)hists.size();
		if (segN <= 1 || (int)labels.size() != segN || (int)cuts.size() != segN + 1) {
			_segLabel[si] = labels; // 그대로
			continue;
		}

		// 세그먼트 길이(cm) 미리 계산
		std::vector<float> segLenCm(segN, 0.0f);
		for (int k = 0; k < segN; ++k) {
			int a = cuts[k], b = cuts[k + 1];
			float acc_norm = 0.0f;
			for (int j = a; j < b; ++j) {
				vec3 d = P[j + 1] - P[j];
				// vec3는 x(),y(),z() 접근. length()가 있으면 그것 사용해도 됨.
				float dn = std::sqrt(d.x() * d.x() + d.y() * d.y() + d.z() * d.z());
				acc_norm += dn;
			}
			segLenCm[k] = acc_norm * _longAxisWorld;
		}

		int commitsHere = 0;

		for (int k = 0; k < segN - 1; ++k)
		{
			// L1 차이 계산 (최대 2.0)
			const auto& h0 = hists[k];
			const auto& h1 = hists[k + 1];
			float diff = std::fabs(h1[0] - h0[0]) + std::fabs(h1[1] - h0[1]) +
				std::fabs(h1[2] - h0[2]) + std::fabs(h1[3] - h0[3]);

			int newLabel = argmax4(h1);
			if (diff < diffTh) continue;                // 변화가 약함 → 무시
			if (newLabel == labels[k]) continue;        // 라벨이 그대로면 분기 아님

			// 새 라벨이 충분히 '지속'되는지 확인
			float runLen = 0.0f;
			int    j = k + 1;
			while (j < segN)
			{
				const auto& hj = hists[j];
				int maj = argmax4(hj);
				// 다수 라벨이 newLabel 이거나, 해당 라벨 비율이 0.5 이상이면 같은 흐름으로 인정
				if (maj == newLabel || hj[newLabel] >= 0.50f) {
					runLen += segLenCm[j];
					++j;
				}
				else {
					break;
				}
			}

			if (runLen >= persist_cm)
			{
				// k+1 ~ j-1 구간은 새 라벨로 '커밋'
				for (int t = k + 1; t < j; ++t) labels[t] = (uint8_t)newLabel;
				commitsHere++;
				// 건너뛰기: 이미 커밋한 구간은 스킵
				k = j - 1;
			}
		}

		_segLabel[si] = std::move(labels);
		totalSplitsCommitted += commitsHere;

		// (선택) per-strand 디버그
		// if (commitsHere>0) printf("[SPLIT] strand %zu committed=%d\n", si, commitsHere);
	}

	// 전체 디버그
	printf("[SPLIT] committed total=%d\n", totalSplitsCommitted);
}

// 보조: 각도(deg) 언랩
static inline float unwrapDelta(float prev, float curr) {
	// prev, curr in degrees (-180..180], 최소 회전 경로로 보정
	float d = curr - prev;
	while (d > 180.0f) d -= 360.0f;
	while (d <= -180.0f) d += 360.0f;
	return d;
}

// 보조: 점의 극각(머리 중심 0.5,0.5 기준), deg 반환
static inline float thetaDeg(vec3 p)      // 값으로 받기(비-const) → x()/y() 호출 가능
{
	const double dx = (double)p.x() - 0.5;
	const double dy = (double)p.y() - 0.5;
	const double th = std::atan2(dy, dx) * (180.0 / 3.14159265358979323846);
	return (float)th;
}

//하단 40% 세그만 모아 목적지 후보를 뽑고 tip일치, 평균 임계, 연속 우세 길이 이 4가지중 2개 이상 만족하면 목적지 확정
void HairDataGenerator::decideDestinationAndTurn(float tailRatio, float tailAvgTh, float runLenCm, float thetaTurnDeg)
{
	if (_strands.empty() || _segCuts.empty() || _segHist.empty() || _segLabel.empty())
		return;

	const size_t S = _strands.size();
	_destLabelPerStrand.assign(S, 0);
	_turnSignPerStrand.assign(S, (int)TurnSign::Straight);
	_tailSidePerStrand.assign(S, (int)TailSide::None);

	// 안전 파라미터
	if (tailRatio <= 0.1f) tailRatio = 0.40f;
	if (tailRatio > 0.9f)  tailRatio = 0.90f;
	if (tailAvgTh <= 0.0f) tailAvgTh = 0.55f;
	if (runLenCm <= 0.0f) runLenCm = 1.5f;
	if (thetaTurnDeg <= 0.0f) thetaTurnDeg = 45.0f;

	for (size_t si = 0; si < S; ++si)
	{
		const auto& P = _strands[si];
		const auto& cuts = _segCuts[si];
		const auto& H = _segHist[si];
		const auto& L = _segLabel[si];

		const int segN = (int)H.size();
		if (P.size() < 2 || segN < 1 || (int)cuts.size() != segN + 1) {
			_destLabelPerStrand[si] = (int)(L.empty() ? 0 : L.back());
			_turnSignPerStrand[si] = (int)TurnSign::Straight;
			_tailSidePerStrand[si] = (int)TailSide::None;
			continue;
		}

		// --- 세그 길이(cm) & 전체 길이
		std::vector<float> segLenCm(segN, 0.0f);
		float totalLen = 0.0f;
		for (int k = 0; k < segN; ++k) {
			int a = cuts[k], b = cuts[k + 1];
			float acc = 0.0f;
			for (int j = a; j < b; ++j) {
				vec3 d = P[j + 1] - P[j];
				acc += std::sqrt(d.x() * d.x() + d.y() * d.y() + d.z() * d.z());
			}
			segLenCm[k] = acc * _longAxisWorld;
			totalLen += segLenCm[k];
		}
		if (totalLen <= 0.0f) totalLen = 1.0f;

		// --- tail(하단) 윈도우 선택: 뒤에서부터 tailRatio 만큼 커버되도록
		float need = totalLen * tailRatio;
		int   tailStartSeg = segN - 1;
		float accTail = 0.0f;
		for (int k = segN - 1; k >= 0; --k) {
			accTail += segLenCm[k];
			tailStartSeg = k;
			if (accTail >= need) break;
		}

		// --- tail 평균 비율(길이 가중)
		float sumW = 0.0f;
		float avg[4] = { 0,0,0,0 };
		for (int k = tailStartSeg; k < segN; ++k) {
			const float w = std::max(1e-6f, segLenCm[k]);
			sumW += w;
			avg[0] += H[k][0] * w; // F
			avg[1] += H[k][1] * w; // B
			avg[2] += H[k][2] * w; // L
			avg[3] += H[k][3] * w; // R
		}
		if (sumW > 0.0f) {
			for (int t = 0; t < 4; ++t) avg[t] /= sumW;
		}
		int destByAvg = 0; float vmax = avg[0];
		for (int t = 1; t < 4; ++t) { if (avg[t] > vmax) { vmax = avg[t]; destByAvg = t; } }

		// --- tip(마지막 세그) 라벨
		const int tipLabel = (int)L.back();

		// --- run 조건(pX>=0.60 연속 길이 ≥ runLenCm)
		int   candidate = destByAvg; // 우선 avg로 뽑힌 섹터를 시험
		float runLen = 0.0f;
		for (int k = segN - 1; k >= tailStartSeg; --k) {
			const float pX = H[k][candidate];
			if (pX >= 0.60f) runLen += segLenCm[k];
			else break;
		}

		// --- 조건 충족 개수 계산: [tip일치], [tailAvg], [run]
		int votes = 0;
		if (tipLabel == candidate) votes++;
		if (avg[candidate] >= tailAvgTh) votes++;
		if (runLen >= runLenCm) votes++;

		// 목적지 확정
		int dest = candidate;
		if (votes < 2) {
			// 보강: tip 기준으로 재평가
			int cand2 = tipLabel;
			float run2 = 0.0f;
			for (int k = segN - 1; k >= tailStartSeg; --k) {
				const float pX = H[k][cand2];
				if (pX >= 0.60f) run2 += segLenCm[k];
				else break;
			}
			int votes2 = 0;
			if (tipLabel == cand2) votes2++;
			if (avg[cand2] >= tailAvgTh) votes2++;
			if (run2 >= runLenCm) votes2++;
			if (votes2 >= 2) dest = cand2;
		}

		const float frontAvgGate = 0.45f;
		const float zPad = 0.02f;

		vec3 tip = P.back();  // ← 복사해서 비-const 객체로 만든다
		if (avg[0] >= frontAvgGate && tip.z() > _zFront - zPad) {
			dest = (int)HairGroup::Front;
		}

		// --- 회전량 Δθ 계산(언랩 누적)
		// 시작/끝 각도 직접 차이보다, 누적 언랩 합으로 총 회전량을 측정
		float th0 = thetaDeg(P.front());
		float accRot = 0.0f;
		float prev = th0;
		for (size_t j = 1; j < P.size(); ++j) {
			float curr = thetaDeg(P[j]);
			accRot += unwrapDelta(prev, curr);
			prev = curr;
		}
		// 분류
		TurnSign turn = TurnSign::Straight;
		if (accRot >= thetaTurnDeg)      turn = TurnSign::CW;     // 시계방향(오른쪽으로 휨)
		else if (accRot <= -thetaTurnDeg) turn = TurnSign::CCW;   // 반시계(왼쪽으로 흐름)

		// --- Back 내부 좌/우 꼬리 판정(선택)
		TailSide tailSide = TailSide::None;
		if (dest == (int)HairGroup::Back) {
			// 끝쪽 몇 점 평균 각으로 좌/우 판단(>0: Right, <0: Left)
			const int tailPts = std::max(2, (int)(P.size() * 0.1)); // 마지막 10%
			float meanTh = 0.0f;
			int cnt = 0;
			for (int j = (int)P.size() - tailPts; j < (int)P.size(); ++j) {
				meanTh += thetaDeg(P[j]);
				cnt++;
			}
			if (cnt > 0) meanTh /= cnt;
			if (meanTh > 0.0f) tailSide = TailSide::Right;
			else if (meanTh < 0.0f) tailSide = TailSide::Left;
			else tailSide = TailSide::None;
		}

		// --- 저장
		_destLabelPerStrand[si] = dest;
		_turnSignPerStrand[si] = (int)turn;
		_tailSidePerStrand[si] = (int)tailSide;

		//디버그
		if (si < 3) {
			printf("[DEST] si=%zu tip=%d avgDest=%d avg=%.2f run=%.2fcm -> dest=%d, turn=%d, tailSide=%d\n",
				si, tipLabel, destByAvg, avg[destByAvg], runLen, dest, (int)turn, (int)tailSide);
		}
	}
}

void HairDataGenerator::buildFinalGroupKeyPerStrand()
{
	if (_strands.empty()) return;

	const size_t S = _strands.size();
	if (_finalGroupId.size() != S) _finalGroupId.assign(S, 0);

	// 목적지/회전/꼬리좌우가 선행 단계에서 채워졌는지 점검
	if (_destLabelPerStrand.size() != S) _destLabelPerStrand.assign(S, 0);
	if (_turnSignPerStrand.size() != S) _turnSignPerStrand.assign(S, 0);
	if (_tailSidePerStrand.size() != S) _tailSidePerStrand.assign(S, 0);

	// 세그 라벨 버퍼 존재여부 점검(없으면 비워둠)
	const bool haveSeg = (_segLabel.size() == S && _segCuts.size() == S);

	for (size_t si = 0; si < S; ++si)
	{
		// root: 루트 포인트 라벨(기존 분류기 사용)
		int root = 0;
		if (!_strands[si].empty())
			root = classifyPoint4(_strands[si].front()); // 0:F,1:B,2:L,3:R

		// dest/turn/tailSide: 5단계에서 계산한 결과 사용
		const int dest = _destLabelPerStrand[si];
		const int turnSign = _turnSignPerStrand[si];      // 0/1/2
		const int tailSide = _tailSidePerStrand[si];      // 0/1/2

		// 최종 그룹키 생성
		const int gkey = encodeGroupKey(root, dest, turnSign, tailSide);
		_finalGroupId[si] = gkey;

		// 갈라짐이 있어도, 최종 목적지/흐름이 정해지면 가닥 전체를 새 그룹으로”
		// → 시각화/후속 단계 일관성을 위해 세그 라벨도 '목적지(dest)'로 전체 덮어쓰기
		if (haveSeg && !_segLabel[si].empty()) {
			for (size_t k = 0; k < _segLabel[si].size(); ++k)
				_segLabel[si][k] = (uint8_t)dest;  // 세그 라벨을 최종 목적지로 통일
		}
	}

	//통계/디버그
	size_t cntBack = 0; for (auto d : _destLabelPerStrand) if (d==1) cntBack++;
	printf("[FINAL] strands=%zu, Back-dest=%zu\n", S, cntBack);
}

void HairDataGenerator::bakeFinalGroupColors()
{
	const size_t S = _strands.size();
	if (_finalGroupId.size() != S) return;
	_colors.resize(S, vec3(1, 1, 1));

	// 1) 현재 등장하는 "고유 그룹키"를 순서대로 인덱싱
	std::unordered_map<int, int> key2idx;
	key2idx.reserve(S * 2);
	int nextIdx = 0;
	for (size_t si = 0; si < S; ++si) {
		int k = _finalGroupId[si];
		if (!key2idx.count(k)) key2idx[k] = nextIdx++;
	}

	// 2) 고대비 팔레트(먼저 20색 고정, 넘치면 해시 HSV)
	auto clamp01 = [](float x) { return x < 0 ? 0.f : (x > 1 ? 1.f : x); };

	static const vec3 TAB20[] = {
		// Tableau 20 비슷한 고대비(0~1 스케일)
		vec3(0.121f,0.466f,0.705f), vec3(1.000f,0.498f,0.054f),
		vec3(0.172f,0.627f,0.172f), vec3(0.839f,0.152f,0.156f),
		vec3(0.580f,0.404f,0.741f), vec3(0.549f,0.337f,0.294f),
		vec3(0.890f,0.467f,0.761f), vec3(0.498f,0.498f,0.498f),
		vec3(0.737f,0.741f,0.133f), vec3(0.090f,0.745f,0.811f),
		vec3(0.682f,0.780f,0.911f), vec3(1.000f,0.733f,0.470f),
		vec3(0.596f,0.874f,0.541f), vec3(1.000f,0.596f,0.588f),
		vec3(0.773f,0.690f,0.835f), vec3(0.768f,0.611f,0.580f),
		vec3(0.969f,0.714f,0.824f), vec3(0.780f,0.780f,0.780f),
		vec3(0.859f,0.859f,0.553f), vec3(0.619f,0.854f,0.898f)
	};
	const int P = (int)(sizeof(TAB20) / sizeof(TAB20[0]));

	auto hsv2rgb_local = [](float h, float s, float v)->vec3 {
		if (s <= 1e-6f) return vec3(v, v, v);
		h = fmodf(h, 1.0f); if (h < 0) h += 1.0f;
		float hf = h * 6.0f; int i = (int)floorf(hf); float f = hf - i;
		float p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
		switch (i % 6) {
		case 0: return vec3(v, t, p);
		case 1: return vec3(q, v, p);
		case 2: return vec3(p, v, t);
		case 3: return vec3(p, q, v);
		case 4: return vec3(t, p, v);
		default:return vec3(v, p, q);
		}
		};

	auto colorFromIndex = [&](int idx)->vec3 {
		if (idx < P) return TAB20[idx];
		// 20색을 넘어가면 해시 기반 골든앵글 HSV로 충분히 떨어뜨리기
		// hue: 황금비 간격, sat/value: 번갈아 변주(가시성↑)
		const float golden = 0.61803398875f;
		float h = fmodf(0.13f + idx * golden, 1.0f);
		float s = 0.70f + 0.25f * ((idx * 37) % 2); // 0.70 또는 0.95
		float v = 0.92f - 0.10f * ((idx * 53) % 2); // 0.92 또는 0.82
		return hsv2rgb_local(h, clamp01(s), clamp01(v));
		};

	// 3) 각 가닥을 "그룹 인덱스 색"으로 칠하기
	for (size_t si = 0; si < S; ++si) {
		int k = _finalGroupId[si];
		int idx = key2idx[k];
		_colors[si] = colorFromIndex(idx);
	}
}

void HairDataGenerator::smoothSegments(float minLen_cm)
{
	if (_strands.empty() || _segCuts.empty() || _segLabel.empty()) return;
	if (minLen_cm <= 0.0f) minLen_cm = 1.0f;

	const size_t S = _strands.size();

	for (size_t si = 0; si < S; ++si)
	{
		auto& labels = _segLabel[si];
		const auto& cuts = _segCuts[si];
		const auto& P = _strands[si];

		const int segN = (int)labels.size();
		if (segN <= 2 || (int)cuts.size() != segN + 1) continue;

		// 세그 길이(cm) 미리 계산
		std::vector<float> segLenCm(segN, 0.0f);
		for (int k = 0; k < segN; ++k) {
			int a = cuts[k], b = cuts[k + 1];
			float acc = 0.0f;
			for (int j = a; j < b; ++j) {
				vec3 d = P[j + 1] - P[j];
				acc += std::sqrt(d.x() * d.x() + d.y() * d.y() + d.z() * d.z());
			}
			segLenCm[k] = acc * _longAxisWorld;
		}

		// 1) 패턴 A-B-A: 가운데 B 섬이 짧으면 양옆 A로 흡수
		for (int k = 1; k < segN - 1; ++k) {
			int L = labels[k - 1], C = labels[k], R = labels[k + 1];
			if (L == R && C != L) {
				if (segLenCm[k] < minLen_cm) {
					labels[k] = (uint8_t)L; // 가운데 섬 병합
				}
			}
		}

		// 2) 그 외 짧은 조각: 더 긴 이웃으로 흡수
		for (int k = 0; k < segN; ++k) {
			float len = segLenCm[k];
			if (len >= minLen_cm) continue;

			// 왼쪽 run 길이
			int li = k - 1; float Lrun = 0.0f; int Llab = -1;
			if (li >= 0) {
				Llab = labels[li];
				int t = li;
				while (t >= 0 && labels[t] == Llab) { Lrun += segLenCm[t]; --t; }
			}

			// 오른쪽 run 길이
			int ri = k + 1; float Rrun = 0.0f; int Rlab = -1;
			if (ri < segN) {
				Rlab = labels[ri];
				int t = ri;
				while (t < segN && labels[t] == Rlab) { Rrun += segLenCm[t]; ++t; }
			}

			// 더 긴 쪽(동률이면 아래쪽/끝쪽 우선)으로 합치기
			if (Lrun > Rrun && Llab >= 0)      labels[k] = (uint8_t)Llab;
			else if (Rlab >= 0)                labels[k] = (uint8_t)Rlab;
		}
	}

	//로그
	printf("[SMOOTH] done (minLen=%.2fcm)\n", minLen_cm);
}

void HairDataGenerator::logFinalGroupColorSummary(const std::vector<int>& finalKeys, const std::vector<vec3>& colors)
{
	struct Acc { int cnt = 0; double r = 0, g = 0, b = 0; };
	std::unordered_map<int, Acc> acc;

	const size_t S = finalKeys.size();
	for (size_t i = 0; i < S; ++i) {
		int key = finalKeys[i];
		auto& a = acc[key];
		a.cnt += 1;

		if (i < colors.size()) {
			vec3 c = colors[i];
			a.r += c.x();
			a.g += c.y();
			a.b += c.z();
		}
	}

	// 요약 헤더
	printf("[SUMMARY] groups=%zu\n", acc.size());
	printf(" key(hex)  Root->Dest  Turn       Tail   Count   AvgColor(R,G,B)\n");

	// 보기 좋게 정렬(가장 많은 그룹부터)
	std::vector<std::pair<int, Acc>> rows(acc.begin(), acc.end());
	std::sort(rows.begin(), rows.end(),
		[](const auto& A, const auto& B) { return A.second.cnt > B.second.cnt; });

	for (const auto& kv : rows) {
		int key = kv.first;
		const Acc& a = kv.second;
		double inv = (a.cnt > 0) ? 1.0 / a.cnt : 0.0;
		double R = a.r * inv, G = a.g * inv, B = a.b * inv;

		int root = keyRoot(key), dest = keyDest(key), turn = keyTurn(key), tail = keyTailLR(key);
		printf(" 0x%04X   %s->%s  %-9s  %-5s %6d   (%.2f, %.2f, %.2f)\n",
			key, grpName4(root), grpName4(dest),
			turnName(turn), tailName(tail),
			a.cnt, R, G, B);
	}
}

void HairDataGenerator::absorbSmallGroupsByKNN_TipAnchored(
	int   minDestCount,
	float radius,
	int   K,
	float voteTh,
	float crossFBTh,
	float tailRatio
)
{
	// 안전 가드
	if (_strands.empty() || _finalGroupId.size() != _strands.size()) return;
	const int S = (int)_strands.size();
	if (S <= 1) return;
	if (radius <= 0.0f) radius = 0.06f;
	if (K < 4) K = 12;
	if (voteTh < 0.50f) voteTh = 0.60f;
	if (crossFBTh < 0.50f) crossFBTh = 0.70f;
	if (tailRatio <= 0.1f || tailRatio > 0.9f) tailRatio = 0.40f;

	// --- decode helpers (이미 위에 static inline들 있음)
	auto keyRoot = [](int k) { return  k & 0xF; };
	auto keyDest = [](int k) { return (k >> 4) & 0xF; };
	auto keyTurn = [](int k) { return (k >> 8) & 0xF; };
	auto keyTailLR = [](int k) { return (k >> 12) & 0xF; };
	auto makeKey = [](int r, int d, int t, int tl) { return ((tl & 0xF) << 12) | ((t & 0xF) << 8) | ((d & 0xF) << 4) | (r & 0xF); };

	// --- 1) dest별 전역 개수 집계 (작은 dest만 후보로 삼기 위함)
	int destCount[4] = { 0,0,0,0 };
	for (int i = 0; i < S; ++i) {
		int d = keyDest(_finalGroupId[i]);
		if (0 <= d && d < 4) destCount[d]++;
	}

	// --- 2) 각 가닥의 "팁 앵커" 좌표 계산 (tailRatio 구간의 길이 가중 중심)
	// decideDestinationAndTurn()에서 쓰는 tailRatio 논리와 정합. :contentReference[oaicite:2]{index=2}
	std::vector<std::array<float, 2>> tipAnchor(S, { 0.5f,0.5f });
	std::vector<int> tailStartSeg(S, 0);

	// 세그 길이(cm)나 히스토그램은 decideDestinationAndTurn 로직과 동일하게 접근. :contentReference[oaicite:3]{index=3}
	for (int si = 0; si < S; ++si) {
		const auto& P = _strands[si];
		const auto& cuts = _segCuts[si];
		const auto& H = _segHist[si];
		const int segN = (int)H.size();
		if ((int)cuts.size() != segN + 1 || segN == 0 || (int)P.size() < 2) {
			vec3 tip = P.empty() ? vec3(0.5f, 0.5f, 0.5f) : P.back();
			tipAnchor[si] = { (float)tip.x(), (float)tip.y() };
			tailStartSeg[si] = std::max(0, segN - 1);
			continue;
		}

		// 세그 길이(cm)
		std::vector<float> segLenCm(segN, 0.0f);
		float totalLen = 0.0f;
		for (int k = 0; k < segN; ++k) {
			int a = cuts[k], b = cuts[k + 1];
			float acc = 0.0f;
			for (int j = a; j < b; ++j) {
				vec3 d = P[j + 1] - P[j];
				acc += std::sqrt(d.x() * d.x() + d.y() * d.y() + d.z() * d.z());
			}
			segLenCm[k] = acc * _longAxisWorld;
			totalLen += segLenCm[k];
		}
		if (totalLen <= 0) totalLen = 1.0f;

		// tailRatio 구간 시작 세그 찾기 (뒤에서부터 누적)  :contentReference[oaicite:4]{index=4}
		float need = totalLen * tailRatio;
		int   tStart = segN - 1;
		float accTail = 0.0f;
		for (int k = segN - 1; k >= 0; --k) {
			accTail += segLenCm[k];
			tStart = k;
			if (accTail >= need) break;
		}
		tailStartSeg[si] = tStart;

		// 해당 구간의 길이 가중 중심(팁 앵커)
		double wsum = 0.0, cx = 0.0, cy = 0.0;
		for (int k = tStart; k < segN; ++k) {
			int a = cuts[k], b = cuts[k + 1];
			const float w = std::max(1e-6f, segLenCm[k]);
			// 세그 내부의 점들도 길이 가중으로 평균(간단히 끝점 평균도 무방)
			// 여기서는 간단히 세그 끝점 평균을 사용
			vec3 pa = P[a], pb = P[b];
			cx += ((double)pa.x() + (double)pb.x()) * 0.5 * w;
			cy += ((double)pa.y() + (double)pb.y()) * 0.5 * w;
			wsum += w;
		}
		if (wsum > 0.0) {
			cx /= wsum; cy /= wsum;
			tipAnchor[si] = { (float)cx, (float)cy };
		}
		else {
			vec3 tip = P.back();
			tipAnchor[si] = { (float)tip.x(), (float)tip.y() };
		}
	}

	// --- 3) 후보 리스트: 전역 dest 개수가 작을 때만 (작은 그룹만 흡수)
	std::vector<int> candidates;
	candidates.reserve(S / 2);
	for (int i = 0; i < S; ++i) {
		int d = keyDest(_finalGroupId[i]);
		if (0 <= d && d < 4 && destCount[d] < minDestCount) {
			candidates.push_back(i);
		}
	}
	if (candidates.empty()) return;

	// --- 4) 각 후보에 대해 반경 r 내 KNN 다수결
	const float r2 = radius * radius;
	std::vector<int> newDest(S, -1);

	for (int idx : candidates) {
		float xi = tipAnchor[idx][0], yi = tipAnchor[idx][1];

		// 반경 내 이웃 수집 (전수, S 수천이면 충분히 빠름)
		std::vector<std::pair<float, int>> nbr; nbr.reserve(64);
		for (int j = 0; j < S; ++j) {
			if (j == idx) continue;
			float dx = tipAnchor[j][0] - xi;
			float dy = tipAnchor[j][1] - yi;
			float d2 = dx * dx + dy * dy;
			if (d2 <= r2) nbr.emplace_back(d2, j);
		}
		if (nbr.empty()) continue;
		std::sort(nbr.begin(), nbr.end(),
			[](const auto& a, const auto& b) { return a.first < b.first; });
		if ((int)nbr.size() > K) nbr.resize(K);

		// 득표(거리 가중 선택: 1/(eps + d))
		double vote[4] = { 0,0,0,0 };
		double invAvgDist = 0.0;
		const double eps = 1e-6;
		for (auto& pr : nbr) {
			float d2 = pr.first;
			double d = std::sqrt(std::max(0.0f, d2));
			double w = 1.0 / (eps + d);
			int j = pr.second;
			int dlab = keyDest(_finalGroupId[j]);
			if (0 <= dlab && dlab < 4) vote[dlab] += w;
			invAvgDist += d;
		}
		invAvgDist = (nbr.empty() ? 1.0 : invAvgDist / (double)nbr.size());

		// 최다 득표 dest & 득표율
		int bestD = 0; double bestV = vote[0];
		double sumV = vote[0] + vote[1] + vote[2] + vote[3];
		for (int d = 1; d < 4; ++d) {
			if (vote[d] > bestV) { bestV = vote[d]; bestD = d; }
		}
		double ratio = (sumV > 0.0) ? bestV / sumV : 0.0;

		// Front/Back 경계 보호: 반대 면으로 넘길 때는 반대 면 표 비율이 높을 때만
		int curD = keyDest(_finalGroupId[idx]);
		bool crossFB = ((curD == 0 || curD == 1) && (bestD == 2 || bestD == 3)) ||
			((curD == 2 || curD == 3) && (bestD == 0 || bestD == 1)) ? false :
			// 위 조건은 좌우/전후 섞였음. 정확히는 Front<->Back 교차만 막고 싶으면:
			((curD == 0 && bestD == 1) || (curD == 1 && bestD == 0));
		// 반대 면 득표율(Front/Back만 따짐)
		double fbOpp = 0.0;
		if (curD == 0 || curD == 1) { // 현재 F/B면
			fbOpp = vote[1 - curD] / std::max(1e-6, sumV); // 반대(F<->B) 비율
		}

		// 신뢰조건: 득표율 + (경계 보호)
		bool ok = (ratio >= voteTh);
		if (ok && crossFB) {
			ok = (fbOpp >= crossFBTh);
		}

		if (ok) {
			// 최종 dest만 교체 (root/turn/tail 속성은 그대로 유지)
			int r = keyRoot(_finalGroupId[idx]);
			int t = keyTurn(_finalGroupId[idx]);
			int tl = keyTailLR(_finalGroupId[idx]);
			newDest[idx] = makeKey(r, bestD, t, tl);
		}
	}

	// --- 5) 적용
	int changed = 0;
	for (int i = 0; i < S; ++i) {
		if (newDest[i] != -1) {
			_finalGroupId[i] = newDest[i];
			changed++;
		}
	}
	if (changed > 0) {
		// 변경된 dest에 맞춰 색상을 다시 입힘
		// (최종 색상은 bakeFinalGroupColors에서 담당) :contentReference[oaicite:5]{index=5}
		fprintf(stderr, "[KNN-absorb] changed=%d (r=%.3f, K=%d, voteTh=%.2f, crossFBTh=%.2f, tail=%.2f)\n",
			changed, radius, K, voteTh, crossFBTh, tailRatio);
	}
}

int HairDataGenerator::extractRepresentativeStrand(const std::vector<int>& strandIndices, float tailRatio, int S, float w_pos, float w_dir, float w_shape)
{
	if (strandIndices.empty()) return -1;

	// --- 각 가닥 요약 정보 저장용
	struct Desc {
		std::vector<vec3> pts; // tail 구간에서 뽑은 점들 (최대 S개)
		vec3 centroid;
		vec3 tipDir;
	};
	std::vector<Desc> descs;
	descs.reserve(strandIndices.size());

	for (int si : strandIndices) {
		const auto& P = _strands[si];
		const auto& cuts = _segCuts[si];
		if (P.size() < 2 || cuts.size() < 2) { descs.push_back({}); continue; }

		// tailRatio만큼 길이 커버되도록 tail 시작 세그 찾기
		const int segN = (int)cuts.size() - 1;
		std::vector<float> segLenCm(segN, 0.0f);
		float totalLen = 0.0f;
		for (int k = 0; k < segN; ++k) {
			int a = cuts[k], b = cuts[k + 1];
			float acc = 0.0f;
			for (int j = a; j < b; ++j) {
				vec3 d = P[j + 1] - P[j];
				acc += std::sqrt(d.x() * d.x() + d.y() * d.y() + d.z() * d.z());
			}
			segLenCm[k] = acc * _longAxisWorld;
			totalLen += segLenCm[k];
		}
		if (totalLen <= 0) totalLen = 1.0f;
		float need = totalLen * tailRatio;
		int tStart = segN - 1;
		float accTail = 0;
		for (int k = segN - 1; k >= 0; --k) {
			accTail += segLenCm[k];
			tStart = k;
			if (accTail >= need) break;
		}// 밑에서부터 S개 세그먼트 포인트 추출 (중점)
		Desc d;
		int segCount = 0;
		for (int k = segN - 1; k >= tStart && segCount < S; --k, ++segCount) {
			int a = cuts[k], b = cuts[k + 1];
			vec3 mid = (P[a] + P[b]) * 0.5f;
			d.pts.push_back(mid);
		}
		std::reverse(d.pts.begin(), d.pts.end()); // 아래→위 순서 정렬

		// centroid
		vec3 c(0, 0, 0);
		for (auto& p : d.pts) c += p;
		if (!d.pts.empty()) c /= (float)d.pts.size();
		d.centroid = c;

		// tip 방향
		if (d.pts.size() >= 2) {
			vec3 v = d.pts.back() - d.pts[d.pts.size() - 2];
			double L = std::sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
			if (L > 1e-8) {
				v /= (float)L;                 // 길이로 나눠 단위벡터
			}
			else {
				v.set(0.0, 0.0, 1.0);          // 길이가 0이면 안전한 기본값
			}
			d.tipDir = v;
		}
		else {
			d.tipDir.set(0.0, 0.0, 1.0);
		}

		descs.push_back(d);
	}
	// --- 두 가닥 거리 함수
	auto dist = [&](int i, int j) {
		const auto& A = descs[i];
		const auto& B = descs[j];
		if (A.pts.empty() || B.pts.empty()) return 1e9f;

		// 위치 거리
		float pos = (A.centroid - B.centroid).length();

		// 끝 방향 각도  ---- 여기 수정 ----
		vec3 ta = A.tipDir;  // 비-const 복사본
		vec3 tb = B.tipDir;  // 비-const 복사본
		double dotv = ta.x() * tb.x() + ta.y() * tb.y() + ta.z() * tb.z();
		if (dotv > 1.0) dotv = 1.0;
		if (dotv < -1.0) dotv = -1.0;
		float dir = (float)std::acos(dotv); // 라디안

		// 모양 거리 (간단: 대응 점 L2 평균)
		int L = (int)std::min(A.pts.size(), B.pts.size()); // size_t → int 캐스팅
		float shape = 0.f;
		for (int k = 0; k < L; ++k) {
			shape += (A.pts[k] - B.pts[k]).length();
		}
		if (L > 0) shape /= L;

		return w_pos * pos + w_dir * dir + w_shape * shape;
		};

	// --- 메도이드 선정
	int bestIdx = -1;
	float bestScore = 1e9f;
	for (int ii = 0; ii < (int)strandIndices.size(); ++ii) {
		float s = 0.f;
		for (int jj = 0; jj < (int)strandIndices.size(); ++jj) {
			if (ii == jj) continue;
			s += dist(ii, jj);
		}
		if (s < bestScore) { bestScore = s; bestIdx = ii; }
	}

	return (bestIdx == -1) ? -1 : strandIndices[bestIdx];
}

std::unordered_map<int, int> HairDataGenerator::computeRepresentatives(int minCount, float tailRatio, int S, float w_pos, float w_dir, float w_shape)
{
	std::unordered_map<int, int> result;
	_repStrands.clear();

	const int N = (int)_strands.size();
	if (N == 0 || (int)_finalGroupId.size() != N) return result;

	// 1) 그룹키 -> 가닥 인덱스 목록 버킷 만들기
	std::unordered_map<int, std::vector<int>> buckets;
	buckets.reserve(N / 4);
	for (int si = 0; si < N; ++si) {
		buckets[_finalGroupId[si]].push_back(si);
	}

	// 2) 버킷마다 (minCount 이상일 때만) 대표선 뽑기
	for (auto& kv : buckets) {
		const int key = kv.first;
		const auto& idxs = kv.second;
		if ((int)idxs.size() < minCount) continue; // 50 미만은 패스

		// 가중치는 extractRepresentativeStrand 안에서 쓰이므로 전달
		int rep = extractRepresentativeStrand(idxs, tailRatio, S, w_pos, w_dir, w_shape);
		if (rep >= 0) {
			result[key] = rep;
			_repStrands.insert(rep); // (선택) 시각화용
		}
	}

	// (선택) 로그
	fprintf(stderr, "[rep] picked %zu representatives (min=%d, tail=%.2f, S=%d)\n",
		result.size(), minCount, tailRatio, S);

	return result;
}

std::unordered_map<int, std::vector<int>> HairDataGenerator::computeRepresentativesMulti(int kPerGroup, int minCount, float tailRatio, int S, float w_pos, float w_dir, float w_shape)
{
	std::unordered_map<int, std::vector<int>> result;
	_repStrands.clear();

	const int N = (int)_strands.size();
	if (N == 0 || (int)_finalGroupId.size() != N) return result;

	// 1) 그룹 버킷 만들기 (기존 로직과 동일)
	std::unordered_map<int, std::vector<int>> buckets;
	buckets.reserve(N / 4);
	for (int si = 0; si < N; ++si) {
		buckets[_finalGroupId[si]].push_back(si);
	}

	// 2) 각 그룹에서 K개 대표선 뽑기
	for (auto& kv : buckets) {
		const int key = kv.first;
		const auto& idxs = kv.second;
		if ((int)idxs.size() < std::max(1, minCount)) continue;

		// 2-1) 전체 후보에 대해 Desc 미리 계산(거리 계산용 캐시)
		std::vector<RepDesc> descs;
		descs.reserve(idxs.size());
		for (int si : idxs) {
			descs.push_back(buildDescForStrand(this, si, tailRatio, S));
		}

		// 2-2) 첫 대표선은 "메도이드"로 선정(기존 함수 재사용)
		std::vector<int> chosen; chosen.reserve(kPerGroup);
		int firstRep = extractRepresentativeStrand(idxs, tailRatio, S, w_pos, w_dir, w_shape);
		if (firstRep < 0) continue;
		chosen.push_back(firstRep);

		auto idxPos = [&](int si)->int {
			// si를 idxs에서의 로컬 인덱스로 변환
			return (int)(std::find(idxs.begin(), idxs.end(), si) - idxs.begin());
			};

		// 2-3) 남은 (K-1)개는 "가장 기존 대표들과 멀리 떨어진" 후보를 그리디로 선택
		while ((int)chosen.size() < kPerGroup) {
			int bestSi = -1;
			float bestMinD = -1.0f;

			for (int c : idxs) {
				// 이미 선택된 것은 스킵
				if (std::find(chosen.begin(), chosen.end(), c) != chosen.end()) continue;

				// 현재 후보 c와 "이미 선택된 대표선들" 사이의 최소거리
				float minD = 1e9f;
				int cpos = idxPos(c);
				const auto& Dc = descs[cpos];
				if (Dc.pts.empty()) continue;

				for (int s : chosen) {
					int spos = idxPos(s);
					const auto& Ds = descs[spos];
					if (Ds.pts.empty()) continue;
					float D = distPosDirShape(Dc, Ds, w_pos, w_dir, w_shape).first;
					if (D < minD) minD = D;
				}

				// 최소거리가 최대가 되도록(=가장 멀리 있는 후보)
				if (minD > bestMinD) { bestMinD = minD; bestSi = c; }
			}

			if (bestSi < 0) break; // 더 이상 뽑을 후보 없음
			chosen.push_back(bestSi);
		}

		// 2-4) 결과 등록 + 시각화 세트 채우기
		result[key] = chosen;
		for (int si : chosen) _repStrands.insert(si);
	}

	fprintf(stderr, "[repK] picked K reps per group: K=%d, groups=%zu\n",
		kPerGroup, result.size());

	return result;
}

void HairDataGenerator::mergeSmallGroupsByRepSimilarity(
	int minLargeCount, float tailRatio, int S,
	float w_pos, float w_dir, float w_shape,
	float dirGateDeg, float Dmax, float margin, bool strictFB)
{
	if (_strands.empty() || (int)_finalGroupId.size() != (int)_strands.size()) return;

	// 1) 그룹 버킷
	unordered_map<int, vector<int>> buckets;
	for (int si = 0; si < (int)_strands.size(); ++si) buckets[_finalGroupId[si]].push_back(si);

	// 2) 큰/작은 그룹 분리
	vector<int> bigKeys, smallKeys;
	bigKeys.reserve(buckets.size()); smallKeys.reserve(buckets.size());
	for (auto& kv : buckets) {
		((int)kv.second.size() >= minLargeCount ? bigKeys : smallKeys).push_back(kv.first);
	}
	if (bigKeys.empty() || smallKeys.empty()) return; // 병합할 게 없음

	// 3) 대표선(메도이드) 뽑기 — 네 기존 함수 그대로 활용
	//    큰 그룹 대표: computeRepresentatives(minLargeCount, ...)
	auto bigRep = computeRepresentatives(minLargeCount, tailRatio, S, w_pos, w_dir, w_shape);

	//    작은 그룹 대표: 임계 1로 완화해서 같은 로직 재사용
	unordered_map<int, int> smallRep;
	for (int k : smallKeys) {
		auto it = buckets.find(k);
		if (it == buckets.end()) continue;
		int rep = extractRepresentativeStrand(it->second, tailRatio, S, w_pos, w_dir, w_shape);
		if (rep >= 0) smallRep[k] = rep;
	}

	// 4) Desc 캐싱
	struct Pack { int key; int si; RepDesc d; };
	vector<Pack> big, sml; big.reserve(bigRep.size()); sml.reserve(smallRep.size());
	for (auto& kv : bigRep) {
		Pack p{ kv.first, kv.second, buildDescForStrand(this, kv.second, tailRatio, S) };
		big.push_back(std::move(p));
	}
	for (auto& kv : smallRep) {
		Pack p{ kv.first, kv.second, buildDescForStrand(this, kv.second, tailRatio, S) };
		sml.push_back(std::move(p));
	}
	if (big.empty() || sml.empty()) return;

	auto deg = [](float rad) { return rad * 180.0f / 3.14159265358979323846f; };

	// 5) 작은 그룹마다 최적 큰 그룹 찾기
	unordered_map<int, int> remap; // smallKey -> bigKey
	for (auto& Sg : sml) {
		int bestK = -1; float bestD = 1e9f; float bestAng = 1e9f;
		float secondD = 1e9f;

		for (auto& Bg : big) {
			// 5-1) 방향 게이팅(빠른 거르기)
			auto pr = distPosDirShape(Sg.d, Bg.d, w_pos, w_dir, w_shape);
			float D = pr.first, ang = pr.second; // ang in rad

			if (deg(ang) > dirGateDeg) continue; // 1차 컷

			// 5-2) 최솟값 추적(2등 거리도 기록해서 margin 체크)
			if (D < bestD) { secondD = bestD; bestD = D; bestAng = ang; bestK = Bg.key; }
			else if (D < secondD) { secondD = D; }
		}

		if (bestK < 0) continue;

		// 5-3) 안전장치: 절대/상대 임계 + FB 교차 보수
		bool ok = (bestD <= Dmax) && (secondD - bestD >= margin);
		if (ok && strictFB) {
			int curD = keyDest(Sg.key), toD = keyDest(bestK);
			bool crossFB = ((curD == 0 && toD == 1) || (curD == 1 && toD == 0));
			if (crossFB) {
				ok = (deg(bestAng) <= std::min(30.0f, dirGateDeg)) && (bestD <= std::min(0.8f, Dmax));
			}
		}

		if (ok) remap[Sg.key] = bestK;
	}

	// 6) 통째 적용
	int changed = 0;
	for (auto& kv : remap) {
		int fromKey = kv.first, toKey = kv.second;
		auto& idxs = buckets[fromKey];
		for (int si : idxs) { _finalGroupId[si] = toKey; ++changed; }
	}
	if (changed > 0) {
		fprintf(stderr, "[REP-merge] merged strands=%d  (groups: %zu -> %zu)\n",
			changed, buckets.size(), buckets.size() - remap.size());
	}
}


void HairDataGenerator::draw(void)
{
	//drawPoint();
	drawStrand();
}

void HairDataGenerator::drawPoint(void)
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glBegin(GL_POINTS);
	int numStrands = (int)_strands.size();
	for (auto s : _strands) {
		for (auto p : s) {
			glVertex3f(p.x(), p.y(), p.z());
		}
	}

	glEnd();
	glEnable(GL_LIGHTING);
	glPopMatrix();
}

void HairDataGenerator::drawStrand()
{
	glPushMatrix();
	glDisable(GL_LIGHTING);

	const int numStrands = (int)_strands.size();

	const bool repsOnly = _repOnly && !_repStrands.empty();
	if (repsOnly) glLineWidth(3.0f);   // 대표선 모드일 때 조금 굵게

	for (int si = 0; si < numStrands; ++si) {
		if (repsOnly) {
			// 대표선만 그리기
			if (_repStrands.find(si) == _repStrands.end()) continue;
		}

		const auto& s = _strands[si];
		if (s.size() < 2) continue;

		vec3 c = (_colors.size() == (size_t)numStrands) ? _colors[si] : vec3(1, 1, 1);
		glColor3f((GLfloat)c.x(), (GLfloat)c.y(), (GLfloat)c.z());

		glBegin(GL_LINES);
		for (int j = 0; j < (int)s.size() - 1; ++j) {
			vec3 p0 = s[j];
			vec3 p1 = s[j + 1];
			glVertex3f(p0.x(), p0.y(), p0.z());
			glVertex3f(p1.x(), p1.y(), p1.z());
		}
		glEnd();
	}

	if (repsOnly) glLineWidth(1.0f);

	glEnable(GL_LIGHTING);
	glPopMatrix();
}


