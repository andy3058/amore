# AI 헤어 인플루언서 큐레이션 에이전트

아모레퍼시픽 헤어 브랜드와 인플루언서 최적 매칭을 위한 **학술적 알고리즘 기반** RAG AI 추천 시스템

## 핵심 특징

- **학술적 알고리즘 기반**: 논문 검증된 알고리즘으로 신뢰성 높은 분석
- **RAG 기반 추천**: ChromaDB 벡터 검색 + LLM 분석으로 정확한 인플루언서 매칭
- **LLM 개인화 페르소나**: GPT-4o-mini로 인플루언서별 고유 페르소나 자동 생성
- **Expert/Trendsetter 분류**: TF-IDF + Cosine Similarity 기반 자동 분류
- **FIS (Fake Integrity Score)**: Benford's Law + Chi-squared Test 기반 허수 계정 탐지
- **Hybrid Scoring**: RRF + Temperature Scaling으로 정밀한 순위 결정

## 학술적 기반

### 1. FIS (Fake Integrity Score) - 허수 계정 탐지

| 알고리즘 | 학술 기반 | 적용 |
|---------|----------|------|
| **Benford's Law** | Golbeck (2015), PLOS ONE | 숫자 분포 기반 봇 탐지 |
| **Chi-squared Test** | Pearson's Chi-squared | Benford 적합도 검정 |
| **Modified Z-score** | Iglewicz & Hoaglin (1993) | 참여율 이상치 탐지 (MAD 기반) |
| **Jaccard Similarity** | Jaccard (1901) | 중복 콘텐츠 탐지 |

```
FIS = Σ(wi × Si) × Geographic_Factor

S_benford:    χ² 기반 Benford 법칙 적합도 (w=0.20)
S_engagement: Modified Z-score 참여율 분석 (w=0.25)
S_comment:    댓글 패턴 엔트로피 (w=0.15)
S_activity:   CV(Coefficient of Variation) 활동 패턴 (w=0.15)
S_duplicate:  Jaccard Similarity 중복 탐지 (w=0.15)
S_geo:        지리적 정합성 (w=0.10)
```

### 2. Expert/Trendsetter 분류

| 알고리즘 | 학술 기반 | 적용 |
|---------|----------|------|
| **TF-IDF** | Salton & McGill (1983) | 키워드 가중치 계산 |
| **Cosine Similarity** | Manning et al. (2008) | 프로필 유사도 측정 |
| **Soft Voting Ensemble** | Dietterich (2000) | 다중 신호 결합 |

```
Expert: 미용사, 살롱 원장, 헤어 시술 전문가
  - 분석 전략: 텍스트 Primary (bio/caption 전문성 분석)
  - 키워드: 원장, 미용사, 살롱, 시술, 펌, 염색, 클리닉

Trendsetter: 스타일 크리에이터, 뷰티 인플루언서
  - 분석 전략: 이미지 Primary (시각적 스타일 분석)
  - 키워드: 크리에이터, 인플루언서, OOTD, 데일리룩
```

### 3. RAG 검색 + Hybrid Scoring

| 알고리즘 | 학술 기반 | 적용 |
|---------|----------|------|
| **RRF** | Cormack et al. (2009) | 순위 기반 점수 융합 |
| **Temperature Scaling** | Hinton et al. | 점수 분포 캘리브레이션 |
| **NDCG** | Järvelin & Kekäläinen (2002) | 추천 품질 평가 |

```
Hybrid Score = α×Vector + β×FIS + γ×RRF

α = 0.50 (벡터 유사도)
β = 0.25 (FIS 신뢰도)
γ = 0.25 (RRF 순위 점수)
k = 60 (RRF 상수, 논문 권장값)
```

### 4. LLM 개인화 페르소나 생성

- **GPT-4o-mini** 기반 인플루언서별 고유 페르소나 자동 생성
- RAG 인덱싱 시 사전 생성 + 캐싱 (실시간 API 비용 절감)
- 다양성 확보: temperature=0.8로 창의적 페르소나 생성

```
Expert 예시: "청담 컬러 마스터", "손상모 복구의 정석", "볼륨펌의 달인"
Trendsetter 예시: "오피스룩의 정석", "캠퍼스 스타일 아이콘", "데일리 뷰티 크리에이터"
```

## 시스템 아키텍처

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Crawlers   │───▶│ Processors  │───▶│RAG Analyzer │───▶│  API/UI     │
│   (수집)    │    │ (분석/분류) │    │ (벡터 검색) │    │  (추천)     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
  Instagram API     FIS Calculator      ChromaDB          FastAPI
  Brand JSON        TF-IDF Classifier   OpenAI Embed      Swagger UI
                    Image Analyzer      LLM Persona
```

### 파이프라인 상세

1. **인덱싱 (서버 시작 시)**
   - LLM Vision으로 인플루언서 이미지 분석
   - GPT-4o-mini로 개인화 페르소나 생성 (캐싱)
   - ChromaDB에 임베딩 벡터 저장

2. **검색 (추천 요청 시)**
   - 브랜드+제품+캠페인 설명으로 쿼리 생성
   - Multi-Signal Hybrid Scoring (Vector + FIS + RRF)
   - Temperature Scaling으로 점수 분포 조정

3. **필터링**
   - Expert: 연령 필터 없음 (모든 연령 시술)
   - Trendsetter: 성별/연령대 엄격 필터 (광고 모델 역할)

4. **추천**
   - LLM 페르소나 + 상세 추천 사유 함께 반환

## 빠른 시작

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 설정

# 실행
python server.py

# 또는 스크립트 사용
./run.sh   # 시작
./stop.sh  # 종료
```

## 접속

- **서버**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs

## 프로젝트 구조

```
amore/
├── server.py              # 메인 서버 (FastAPI)
├── run.sh / stop.sh       # 실행/종료 스크립트
├── requirements.txt       # 의존성
├── .env                   # 환경변수 (OPENAI_API_KEY)
│
├── api/                   # API 라우터
│   └── routes.py          # 엔드포인트 정의 + 추천 로직
│
├── pipeline/              # 핵심 파이프라인 모듈
│   ├── __init__.py        # 모듈 초기화
│   │
│   ├── crawlers.py        # 데이터 수집
│   │   ├── BrandCrawler       # 브랜드 JSON 관리
│   │   └── InfluencerCrawler  # Instagram Graph API 수집
│   │
│   ├── processors.py      # 학술적 알고리즘 기반 처리
│   │   ├── FISCalculator        # Benford + Chi-squared 허수 탐지
│   │   ├── InfluencerClassifier # TF-IDF + Cosine 분류
│   │   └── RecommendationEvaluator # NDCG + Diversity 평가
│   │
│   └── rag_analyzer.py    # RAG 시스템 (핵심)
│       ├── InfluencerImageAnalyzer  # LLM Vision 분석 + 페르소나 생성
│       ├── InfluencerRAG            # ChromaDB + Hybrid Scoring
│       └── InfluencerAnalysisManager # 통합 관리자
│
├── config/                # 설정
│   ├── products.py        # 제품 카테고리/키워드
│   └── instagram.py       # Instagram API 설정
│
├── data/                  # 데이터
│   ├── influencers_data.json  # 인플루언서 데이터 (300명)
│   ├── amore_brands.json      # 아모레퍼시픽 헤어 브랜드 (6개)
│   └── rag_index/             # ChromaDB 인덱스 (자동 생성)
│
├── scripts/               # 유틸리티 스크립트
│   └── generate_sample_data.py  # 샘플 데이터 생성
│
└── static/                # 정적 파일
    └── index.html
```

## API 엔드포인트

### 추천 API

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| POST | `/api/recommend` | 인플루언서 추천 (RAG + Hybrid Scoring) |

**요청 예시:**
```json
{
  "brand_name": "라보에이치",
  "product_line": "두피케어",
  "description": "30,40대 여성 대상 탈모 예방 캠페인",
  "target_gender": "female",
  "expert_count": 2,
  "trendsetter_count": 3
}
```

**응답 예시:**
```json
{
  "brand_info": { "name": "라보에이치", ... },
  "recommendations": [
    {
      "username": "hair_master_kim",
      "match_score": 96.5,
      "rag_profile": {
        "llm_persona": "두피 솔루션 전문가",
        "persona": "두피 솔루션 전문가 | 신뢰감 있는",
        "influencer_type": "expert",
        "fis_score": 92.3
      },
      "match_reason": "두피케어 튜토리얼 콘텐츠로 높은 인기를 얻고 있는 전문가입니다..."
    }
  ]
}
```

### 브랜드 API

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| GET | `/api/brands` | 브랜드 목록 |
| GET | `/api/brands/{name}` | 브랜드 상세 |

### 제품 API

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| GET | `/api/product-categories` | 제품 카테고리 |
| GET | `/api/product-categories/{name}` | 카테고리별 제품 |
| GET | `/api/product-types` | 전체 제품 유형 |

### 인플루언서 API

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| GET | `/api/influencers` | 인플루언서 목록 |
| GET | `/api/influencers/{username}` | 인플루언서 상세 분석 |

### RAG 관리 API

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| POST | `/api/rag/analyze` | 인플루언서 분석 및 인덱싱 |
| GET | `/api/rag/status` | RAG 시스템 상태 |
| GET | `/api/rag/influencer/{username}` | RAG 프로필 조회 |

### 기타

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| GET | `/health` | 헬스 체크 |

## 지원 브랜드

| 브랜드 | 스타일 | 주요 제품 |
|--------|--------|----------|
| 려 (Ryo) | Natural | 탈모케어 샴푸, 두피세럼, 트리트먼트 |
| 미쟝센 | Trendy | 에센스, 셀프염색(헬로버블), 스타일링 |
| 라보에이치 | Natural | 두피케어 샴푸, 스캘프 세럼 |
| 아윤채 | Luxury | PRO 샴푸/트리트먼트, 염색약, 펌제 |
| 아모스 프로페셔널 | Classic | 살롱 염색약, 펌제, 클리닉 |
| 롱테이크 | Trendy | 헤어 퍼퓸, 디퓨저, 샴푸 |

## 제품 카테고리

- **소비자용**: 샴푸, 트리트먼트, 에센스, 스타일링, 셀프염색, 헤어 프래그런스
- **전문가용**: 살롱 케어, 살롱 염색, 살롱 펌
- **공통**: 두피케어, 기타

## 기술 스택

| 분류 | 기술 |
|-----|------|
| **Backend** | FastAPI, Python 3.10+ |
| **Vector DB** | ChromaDB |
| **AI/LLM** | OpenAI API (GPT-4o-mini, text-embedding-ada-002) |
| **알고리즘** | Benford's Law, TF-IDF, RRF, Temperature Scaling |
| **Data** | JSON 기반 데이터 저장 |

## 환경 변수

```bash
# .env 파일
OPENAI_API_KEY=sk-...              # OpenAI API 키 (필수)
INSTAGRAM_ACCESS_TOKEN=...         # Instagram Graph API (선택)
INSTAGRAM_BUSINESS_ACCOUNT_ID=...  # Instagram 비즈니스 계정 ID (선택)
```

## 데이터 스키마

### 인플루언서

```json
{
  "username": "hair_master_kim",
  "influencer_type": "expert",
  "followers": 85000,
  "bio": "청담동 헤어살롱 원장 | 15년차 미용사",
  "fis": {
    "score": 85.2,
    "verdict": "신뢰 계정 (A등급)",
    "breakdown": {
      "benford_conformity": 92.5,
      "engagement_authenticity": 88.3,
      "comment_pattern": 75.2,
      "activity_regularity": 85.0,
      "geographic_consistency": 80.0,
      "content_originality": 90.1
    }
  },
  "rag_profile": {
    "llm_persona": "청담 컬러 마스터",
    "main_mood": "세련된",
    "content_type": "전후비교"
  }
}
```

### 브랜드

```json
{
  "brand_name": "라보에이치",
  "aesthetic_style": "Natural",
  "slogan": "두피 스킨케어의 새로운 기준",
  "core_values": ["전문성", "혁신", "자연주의", "효과성", "신뢰성", "지속 가능성"],
  "price_tier": "Mid-range"
}
```

## 참고 문헌

1. Golbeck, J. (2015). "Benford's Law Applies to Online Social Networks" PLOS ONE
2. Mazza et al. (2020). "Bot Detection using Benford's Law" ACM SIN
3. Salton & McGill (1983). "Introduction to Modern Information Retrieval"
4. Manning et al. (2008). "Introduction to Information Retrieval"
5. Cormack et al. (2009). "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
6. Järvelin & Kekäläinen (2002). "Cumulated Gain-Based Evaluation of IR Techniques"
7. Iglewicz & Hoaglin (1993). "How to Detect and Handle Outliers"
8. Dietterich (2000). "Ensemble Methods in Machine Learning"

## 라이선스

MIT License
