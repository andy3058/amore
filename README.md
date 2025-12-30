# AI 헤어 인플루언서 큐레이션 에이전트

아모레퍼시픽 헤어 브랜드와 인플루언서 최적 매칭을 위한 RAG 기반 AI 추천 시스템

## 주요 기능

- **RAG 기반 추천**: ChromaDB 벡터 검색 + LLM 분석으로 정확한 인플루언서 매칭
- **Expert/Trendsetter 분류**: 인플루언서 유형 자동 분류
  - Expert: 미용사, 살롱 원장, 헤어 시술 전문가
  - Trendsetter: 뷰티 인플루언서, 스타일 크리에이터
- **FIS (Fake Integrity Score)**: 6가지 지표로 허수 계정 필터링
- **다각화된 추천 사유**: 인플루언서 특성에 맞는 상세한 추천 이유 제공
- **타겟 필터링**: 성별, 연령대 기반 정밀 필터링

## 시스템 아키텍처

```
Crawler → Processor → RAG Analyzer → API
   ↓          ↓            ↓           ↓
 수집     분류/FIS    벡터 인덱싱    추천
```

### RAG 파이프라인

1. **인덱싱**: LLM Vision으로 인플루언서 이미지 분석 → ChromaDB에 벡터 저장
2. **검색**: 브랜드+제품+캠페인 설명으로 쿼리 생성 → 유사도 검색
3. **필터링**: 성별/연령대/FIS 기준 필터링
4. **추천**: 상세한 추천 사유와 함께 결과 반환

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
├── .env.example           # 환경변수 예시
│
├── api/                   # API 라우터
│   └── routes.py          # 엔드포인트 정의
│
├── pipeline/              # 핵심 파이프라인 모듈
│   ├── __init__.py        # 모듈 초기화
│   │
│   ├── crawlers.py        # 데이터 수집
│   │   ├── BrandCrawler       # 브랜드 JSON 관리
│   │   └── InfluencerCrawler  # Instagram Graph API 수집
│   │
│   ├── processors.py      # 데이터 처리
│   │   ├── InfluencerProcessor  # 메인 처리 파이프라인
│   │   ├── FISCalculator        # 허수 계정 탐지 (6지표)
│   │   ├── InfluencerClassifier # Expert/Trendsetter 분류
│   │   └── ImageAnalyzer        # LLM 비전 이미지 분석
│   │
│   └── rag_analyzer.py    # RAG 시스템 (핵심)
│       ├── InfluencerImageAnalyzer  # LLM Vision 이미지 분석
│       ├── InfluencerRAG            # ChromaDB 벡터 검색
│       └── InfluencerAnalysisManager # 통합 관리자
│
├── config/                # 설정
│   ├── products.py        # 제품 카테고리/키워드
│   └── instagram.py       # Instagram API 설정
│
├── data/                  # 데이터
│   ├── influencers_data.json  # 인플루언서 데이터 (300명)
│   ├── amore_brands.json      # 아모레퍼시픽 헤어 브랜드
│   └── influencer_rag/        # ChromaDB 인덱스 (자동 생성)
│
├── scripts/               # 유틸리티 스크립트
│   └── generate_sample_data.py  # 샘플 데이터 생성
│
└── static/                # 정적 파일
    └── index.html
```

## 핵심 알고리즘

### 1. Expert/Trendsetter 분류

```
Expert: 미용사, 살롱 원장, 시술 전문가
  - 키워드: 원장, 미용사, 살롱, 시술, 펌, 염색 등
  - 분석 전략: 텍스트 Primary (bio/caption 분석)

Trendsetter: 스타일 크리에이터, 뷰티 인플루언서
  - 키워드: 크리에이터, 인플루언서, OOTD, 데일리룩 등
  - 분석 전략: 이미지 Primary (시각적 스타일 분석)
```

### 2. FIS (Fake Integrity Score)

```
FIS = (w1×V + w2×A + w3×E + w4×ACS + w5×DUP) × D/100

V:   조회수 변동성 (CV) - 뷰봇 탐지
A:   참여 비대칭성 (좋아요/조회수) - 좋아요 구매 탐지
E:   댓글 엔트로피 (댓글/조회수) - 봇 댓글 탐지
ACS: 활동 안정성 (업로드 간격)
D:   지리적 정합성 (한국 타겟)
DUP: 중복 콘텐츠 비율

가중치: V=0.20, A=0.25, E=0.15, ACS=0.10, D=0.15, DUP=0.15
점수 기준: 80+ 신뢰 계정, 60-79 주의 필요, 60 미만 허수 의심
```

### 3. RAG 벡터 검색

- **임베딩**: OpenAI text-embedding-ada-002
- **벡터 DB**: ChromaDB (로컬 저장)
- **메타데이터 필터링**: influencer_type, target_gender, fis_score
- **유사도**: 코사인 유사도 기반 검색

## API 엔드포인트

### 추천 API

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| POST | `/api/recommend` | 인플루언서 추천 (RAG 기반) |

**요청 예시:**
```json
{
  "brand_name": "려",
  "product_type": "탈모케어 샴푸",
  "description": "30,40대 여성 대상 탈모 예방 캠페인",
  "target_gender": "female",
  "expert_count": 2,
  "trendsetter_count": 3
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

- **Backend**: FastAPI, Python 3.10+
- **Vector DB**: ChromaDB
- **AI/LLM**: OpenAI API (GPT-4o-mini, text-embedding-ada-002)
- **Data**: JSON 기반 데이터 저장

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
    "verdict": "신뢰 계정"
  },
  "recent_posts": [...]
}
```

### 브랜드

```json
{
  "brand_name": "미쟝센",
  "aesthetic_style": "Trendy",
  "slogan": "나만의 스타일을 완성하다",
  "core_values": ["트렌디", "스타일링", "셀프케어"],
  "price_tier": "Mid-range"
}
```

## 라이선스

MIT License
