# AI 헤어 인플루언서 큐레이션 에이전트

아모레퍼시픽 헤어 브랜드와 인플루언서 최적 매칭을 위한 AI 기반 추천 시스템

## 주요 기능

- **RAG 기반 캠페인 매칭**: 자연어로 캠페인을 설명하면 최적의 인플루언서 추천
- **FIS (Fake Integrity Score)**: 허수 계정 필터링
- **Expert/Trendsetter 분류**: 인플루언서 유형 자동 분류
- **다차원 스코어링**: 캠페인, 제품, 콘텐츠, 미적감각, 전문성, 참여율 종합 평가
- **XAI 추천 사유**: 추천 이유 설명 제공

## 빠른 시작

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 설정

# 실행
./run.sh
# 또는
python server.py

# 종료
./stop.sh
```

## 접속

- **서버**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs

## 프로젝트 구조

```
amore/
├── server.py              # 메인 서버 (진입점)
├── run.sh / stop.sh       # 실행/종료 스크립트
├── requirements.txt       # 의존성
├── .env.example           # 환경변수 예시
│
├── api/                   # API 라우터
│   └── routes.py          # 엔드포인트 정의
│
├── services/              # 서비스 로직
│   ├── chatbot.py         # 챗봇 서비스
│   ├── instagram_scraper.py  # Instagram 데이터 수집
│   └── brand_crawler.py   # 브랜드 정보 크롤러
│
├── config/                # 설정
│   ├── products.py        # 제품 카테고리/키워드
│   └── instagram.py       # Instagram API 설정
│
├── modules/               # 핵심 모듈
│   ├── rag_matcher.py     # RAG 기반 캠페인 매칭 (NEW)
│   ├── matcher.py         # 기본 매칭 알고리즘
│   ├── fis_engine.py      # FIS 점수 계산
│   ├── taxonomy.py        # Expert/Trendsetter 분류
│   ├── brand_analyzer.py  # 브랜드 벡터 분석
│   └── image_analyzer.py  # 이미지 분석
│
├── data/                  # 데이터
│   ├── influencers_data.json  # 인플루언서 100명 샘플
│   └── amore_brands.json      # 아모레퍼시픽 헤어 브랜드 6개
│
└── static/                # 정적 파일
    └── index.html
```

## API 엔드포인트

### 추천 API

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| POST | `/api/recommend` | 기본 인플루언서 추천 |
| POST | `/api/recommend-campaign` | RAG 기반 캠페인 맞춤 추천 |

### 브랜드 API

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| GET | `/api/brands` | 브랜드 목록 |
| GET | `/api/brands/{name}` | 브랜드 상세 |

### 제품 API

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| GET | `/api/product-categories` | 제품 카테고리 (1단계) |
| GET | `/api/product-categories/{name}` | 세부 제품 (2단계) |

### 인플루언서 API

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| GET | `/api/influencers` | 인플루언서 목록 |
| GET | `/api/influencers/{username}` | 인플루언서 분석 |

### 기타

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| POST | `/api/chat` | 챗봇 대화 |
| GET | `/health` | 헬스 체크 |

## RAG 캠페인 매칭 예시

```bash
curl -X POST "http://localhost:8000/api/recommend-campaign" \
  -H "Content-Type: application/json" \
  -d '{
    "brand_name": "려",
    "campaign_query": "탈모 고민 있는 30대 남성 대상 루트젠 샴푸 캠페인",
    "top_k": 5
  }'
```

## 지원 브랜드

| 브랜드 | 카테고리 | 특징 |
|--------|---------|------|
| 려 (Ryo) | Hair Care | 한방, 탈모케어 |
| 미쟝센 | Hair Care | 트렌디, 스타일링 |
| 라보에이치 | Hair Care | 더마, 두피과학 |
| 아모스프로페셔널 | Hair Care | 살롱 전문가용 |
| 아윤채 | Hair Care | 프리미엄, 럭셔리 |
| 롱테이크 | Fragrance | 지속가능, 향수 |

## 기술 스택

- **Backend**: FastAPI, Python 3.10+
- **AI/ML**: OpenAI API (이미지 분석)
- **Data**: JSON 기반 데이터 저장