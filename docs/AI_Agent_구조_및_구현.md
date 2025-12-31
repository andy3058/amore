# AI 헤어 인플루언서 큐레이션 에이전트 - 기능 및 구조

## 3. AI Agent 기능 및 구조

### 3.1 시스템 개요

#### 3.1.1 프로젝트 목표
- **핵심 목표**: 아모레퍼시픽 헤어 브랜드(려, 미쟝센, 라보에이치, 아윤채, 아모스 프로페셔널, 롱테이크)와 최적의 인플루언서 매칭
- **학술적 접근**: Benford's Law, TF-IDF, RRF, Temperature Scaling 등 검증된 알고리즘 적용
- **RAG 기반 추천**: ChromaDB 벡터 검색 + LLM 페르소나 생성으로 맞춤형 추천

#### 3.1.2 기술 스택
```
Backend Framework: FastAPI (Python 3.9+)
Vector Database:   ChromaDB (text-embedding-3-small)
LLM:              OpenAI GPT-4o-mini (페르소나 생성)
Frontend:         HTML/CSS/JavaScript (정적 파일)
API:              Instagram Graph API v21.0
```

#### 3.1.3 디렉토리 구조
```
amore/
├── server.py                 # FastAPI 메인 서버
├── api/
│   └── routes.py            # API 엔드포인트 라우터
├── pipeline/
│   ├── __init__.py          # 모듈 익스포트
│   ├── crawlers.py          # 데이터 수집 (Instagram Graph API)
│   ├── processors.py        # FIS 계산, 분류기, 평가기
│   └── rag_analyzer.py      # RAG 시스템 (ChromaDB + LLM)
├── config/
│   ├── __init__.py          # 설정 익스포트
│   ├── products.py          # 제품 카테고리 정의
│   └── instagram.py         # Instagram API 설정
├── data/
│   ├── influencers_data.json  # 인플루언서 데이터 (300명)
│   └── amore_brands.json      # 브랜드 정보
└── static/
    └── index.html           # 웹 인터페이스
```

---

### 3.2 파이프라인 구조

#### 3.2.1 전체 파이프라인 플로우
```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION                               │
│  Instagram Graph API → BrandCrawler / InfluencerCrawler             │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         PROCESSING                                   │
│  ┌─────────────────┐    ┌──────────────────────┐                    │
│  │  FISCalculator  │    │ InfluencerClassifier │                    │
│  │  ─────────────  │    │ ──────────────────── │                    │
│  │  Benford's Law  │    │ TF-IDF + Cosine Sim  │                    │
│  │  Chi-squared    │    │ Soft Voting Ensemble │                    │
│  │  Modified Z     │    │ Expert/Trendsetter   │                    │
│  └─────────────────┘    └──────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      RAG INDEXING                                    │
│  ┌─────────────────────────┐    ┌─────────────────────────┐         │
│  │ InfluencerImageAnalyzer │    │     InfluencerRAG       │         │
│  │ ─────────────────────── │    │ ───────────────────────  │         │
│  │ LLM Vision Analysis     │ →  │ ChromaDB Vector Store   │         │
│  │ GPT-4o-mini Persona     │    │ text-embedding-3-small  │         │
│  └─────────────────────────┘    └─────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      RECOMMENDATION                                  │
│  Query → Hybrid Scoring (Vector 0.50 + FIS 0.25 + RRF 0.25)         │
│       → Temperature Scaling (T=0.5)                                  │
│       → Marketing Filter (Expert: 느슨 / Trendsetter: 엄격)          │
│       → 추천 결과 + 상세 추천 사유 생성                              │
└─────────────────────────────────────────────────────────────────────┘
```

#### 3.2.2 데이터 흐름
```
1. 수집 단계
   Instagram API → 프로필/미디어 데이터 → influencers_data.json

2. 분석 단계
   Raw Data → FIS 계산 (허수 탐지) → 분류 (Expert/Trendsetter)

3. 인덱싱 단계
   분석 결과 → LLM Vision 분석 → 페르소나 생성 → ChromaDB 저장

4. 검색 단계
   검색 쿼리 → 벡터 유사도 → Hybrid Scoring → 필터링 → 결과 반환
```

---

### 3.3 모듈별 기능

#### 3.3.1 Crawlers (`pipeline/crawlers.py`)

| 클래스 | 기능 | 주요 메서드 |
|--------|------|-------------|
| `BrandCrawler` | 아모레퍼시픽 브랜드 정보 수집 | `crawl_all()`, `_parse_brand_page()` |
| `InfluencerCrawler` | Instagram 인플루언서 데이터 수집 | `search_by_hashtag()`, `get_profile()`, `get_recent_media()` |

**Instagram Graph API 설정** (`config/instagram.py`):
```python
INSTAGRAM_API_VERSION = "v21.0"
RATE_LIMIT_PER_HOUR = 200
HASHTAG_SEARCH_LIMIT_PER_WEEK = 30

PROFILE_FIELDS = ["username", "name", "biography", "followers_count", ...]
MEDIA_FIELDS = ["id", "caption", "media_type", "like_count", "comments_count", ...]
```

#### 3.3.2 Processors (`pipeline/processors.py`)

| 클래스 | 기능 | 알고리즘 |
|--------|------|----------|
| `FISCalculator` | Fake Integrity Score 계산 | Benford's Law, Chi-squared, Modified Z-score, Jaccard |
| `InfluencerClassifier` | Expert/Trendsetter 분류 | TF-IDF, Cosine Similarity, Soft Voting |
| `RecommendationEvaluator` | 추천 품질 평가 | NDCG, Diversity, Coverage |

#### 3.3.3 RAG Analyzer (`pipeline/rag_analyzer.py`)

| 클래스 | 기능 | 기술 |
|--------|------|------|
| `InfluencerImageAnalyzer` | 이미지 분석 + 페르소나 생성 | LLM Vision, GPT-4o-mini |
| `InfluencerRAG` | 벡터 저장 및 검색 | ChromaDB, text-embedding-3-small |
| `InfluencerAnalysisManager` | 전체 분석 파이프라인 관리 | 배치 처리, 에러 핸들링 |

**Hybrid Scoring 가중치**:
```python
WEIGHT_VECTOR = 0.50  # 벡터 유사도 (의미적 관련성)
WEIGHT_FIS = 0.25     # FIS 신뢰도 (허수 필터링)
WEIGHT_RRF = 0.25     # RRF 순위 점수 (순위 융합)
RRF_K = 60            # RRF 상수 (Cormack et al., 2009)
TEMPERATURE = 0.5     # 점수 분포 조정
```

#### 3.3.4 API Routes (`api/routes.py`)

| 엔드포인트 | 메서드 | 기능 |
|------------|--------|------|
| `/api/brands` | GET | 브랜드 목록 조회 |
| `/api/brands/{name}` | GET | 브랜드 상세 정보 |
| `/api/recommend` | POST | 인플루언서 추천 (RAG 기반) |
| `/api/product-categories` | GET | 제품 카테고리 목록 |
| `/api/influencers` | GET | 인플루언서 전체 목록 |
| `/api/influencers/{username}` | GET | 인플루언서 상세 (FIS, 분류 포함) |
| `/api/rag/analyze` | POST | 인플루언서 분석 및 인덱싱 |
| `/api/rag/status` | GET | RAG 시스템 상태 |

---

## 4. AI Agent 개발 구현 과정

### 4.1 데이터 스키마

#### 4.1.1 인플루언서 데이터 (`influencers_data.json`)
```json
{
  "influencers": [
    {
      "username": "hair_master_kim",
      "influencer_type": "expert",
      "followers": 173585,
      "bio": "청담 컬러 전문숍 | 하이톤 염색 | 블리치 전문 | 예약 DM",
      "classification_confidence": 0.9,

      "analysis_strategy": {
        "primary": "text",
        "secondary": "image",
        "reason": "Expert는 bio와 caption에 전문 정보가 풍부함"
      },

      "text_analysis": {
        "analysis_type": "text_primary",
        "specialties_from_bio": ["염색"],
        "certifications_detected": [],
        "techniques_from_caption": ["볼륨펌"],
        "caption_detail_level": "high",
        "text_confidence": 0.92
      },

      "image_analysis": {
        "analysis_type": "image_secondary",
        "verified_specialties": ["염색"],
        "additional_specialties": ["클리닉"],
        "signature_techniques": ["클리닉트리트먼트", "C컬펌", "두피스케일링"],
        "client_hair_types": ["스트레이트", "롱헤어", "히피펌"],
        "color_specialties": ["하이톤", "브라운"],
        "work_environment": "home_salon",
        "content_quality_score": 0.92,
        "expertise_confidence": 0.74,
        "target_gender": "female",
        "target_age": "20대",
        "main_mood": "트렌디한"
      },

      "fis": {
        "score": 91.6,
        "verdict": "신뢰 가능"
      }
    }
  ]
}
```

#### 4.1.2 브랜드 데이터 (`amore_brands.json`)
```json
{
  "brands": {
    "려": {
      "name": "려",
      "slogan": "두피부터 모발 끝까지, 한방 헤어케어",
      "aesthetic_style": "Natural",
      "core_values": ["두피건강", "탈모케어", "한방성분"],
      "target_audience": "30-50대 탈모/두피 고민 고객"
    },
    "미쟝센": {
      "name": "미쟝센",
      "slogan": "당신의 헤어를 완성하다",
      "aesthetic_style": "Trendy",
      "core_values": ["손상케어", "스타일링", "셀프염색"],
      "target_audience": "20-30대 트렌디한 소비자"
    }
  },
  "hair_brands": ["려", "미쟝센", "라보에이치", "아윤채", "아모스 프로페셔널", "롱테이크"]
}
```

#### 4.1.3 추천 요청 스키마 (`RecommendRequest`)
```python
class RecommendRequest(BaseModel):
    brand_name: str                    # 필수: 브랜드명
    product_type: Optional[str]        # 제품 유형 (샴푸, 트리트먼트 등)
    product_line: Optional[str]        # 제품 라인 (자양윤모, 퍼펙트세럼 등)
    description: Optional[str]         # 캠페인 설명
    target_gender: Optional[str]       # female/male/unisex
    expert_count: Optional[int] = 2    # 전문가 추천 수
    trendsetter_count: Optional[int] = 3  # 트렌드세터 추천 수
```

---

### 4.2 Processors 상세

#### 4.2.1 FISCalculator (Fake Integrity Score)

**학술적 기반**:
- Benford's Law: Golbeck, J. (2015). "Benford's Law Applies to Online Social Networks"
- Chi-squared Test: χ² < 15.507 (df=8, α=0.05)
- Modified Z-score: Iglewicz & Hoaglin (1993)

**FIS 계산 공식**:
```
FIS = w_benford × S_benford + w_engagement × S_engagement + w_comment × S_comment
    + w_activity × S_activity + w_geo × S_geo + w_duplicate × S_duplicate

가중치:
- w_benford = 0.20 (Benford 적합도)
- w_engagement = 0.25 (참여율 이상치)
- w_comment = 0.15 (댓글 품질)
- w_activity = 0.15 (활동 패턴)
- w_geo = 0.10 (지역 분포)
- w_duplicate = 0.15 (중복 콘텐츠)
```

**Benford's Law 적용** (`processors.py:84-150`):
```python
BENFORD_EXPECTED = {
    1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 5: 0.079,
    6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
}
CHI_SQUARED_CRITICAL = 15.507  # df=8, α=0.05

def _calculate_benford_score(self, numbers: List[int]) -> float:
    """첫째 자릿수 분포의 Benford's Law 적합도 계산"""
    first_digits = [int(str(abs(n))[0]) for n in numbers if n > 0]
    observed = Counter(first_digits)
    total = sum(observed.values())

    # Chi-squared 검정
    chi_squared = sum(
        ((observed.get(d, 0)/total - expected)**2) / expected
        for d, expected in BENFORD_EXPECTED.items()
    )

    # χ² < 15.507이면 Benford 법칙 따름 (자연스러운 분포)
    if chi_squared < CHI_SQUARED_CRITICAL:
        return 1.0 - (chi_squared / CHI_SQUARED_CRITICAL) * 0.3
    else:
        return max(0.3, 1.0 - (chi_squared / 100))
```

**Modified Z-score** (`processors.py:200-250`):
```python
def _calculate_engagement_anomaly(self, engagement_rates: List[float]) -> float:
    """참여율 이상치 탐지 (Modified Z-score)"""
    median = np.median(engagement_rates)
    mad = np.median([abs(x - median) for x in engagement_rates])

    # MAD가 0이면 (모든 값이 같으면) 완벽한 점수
    if mad == 0:
        return 1.0

    # Modified Z-score 계산 (Iglewicz & Hoaglin, 1993)
    k = 0.6745  # 정규분포 가정 시 상수
    z_scores = [(k * (x - median)) / mad for x in engagement_rates]

    # |Z| > 3.5인 이상치 비율
    outlier_ratio = sum(1 for z in z_scores if abs(z) > 3.5) / len(z_scores)
    return 1.0 - outlier_ratio
```

**Jaccard 유사도 (중복 콘텐츠)** (`processors.py:300-350`):
```python
def _calculate_duplicate_score(self, captions: List[str]) -> float:
    """캡션 중복도 측정 (Jaccard Similarity)"""
    if len(captions) < 2:
        return 1.0

    similarities = []
    for i, cap1 in enumerate(captions):
        for cap2 in captions[i+1:]:
            set1 = set(cap1.split())
            set2 = set(cap2.split())

            # Jaccard Similarity = |A ∩ B| / |A ∪ B|
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            similarity = intersection / union if union > 0 else 0
            similarities.append(similarity)

    avg_similarity = np.mean(similarities)
    # 유사도가 높으면 (중복이 많으면) 점수 하락
    return 1.0 - avg_similarity
```

#### 4.2.2 InfluencerClassifier (Expert/Trendsetter 분류)

**학술적 기반**:
- TF-IDF: Salton, G., & McGill, M. J. (1983). "Introduction to Modern Information Retrieval"
- Cosine Similarity: 벡터 공간 모델 기반 유사도
- Soft Voting Ensemble: 다중 분류기 앙상블

**분류 가중치**:
```python
WEIGHT_KEYWORD = 0.40  # 키워드 기반 점수
WEIGHT_TFIDF = 0.40    # TF-IDF 유사도 점수
WEIGHT_IMAGE = 0.20    # 이미지 분석 점수
```

**TF-IDF 분류** (`processors.py:500-600`):
```python
class InfluencerClassifier:
    """TF-IDF + Soft Voting 앙상블 분류기"""

    EXPERT_KEYWORDS = [
        '미용사', '헤어디자이너', '원장', '실장', '수석', '스타일리스트',
        '살롱', '헤어샵', '미용실', '전문', '시술', '예약', '자격증',
        '펌전문', '염색전문', '커트전문', '두피전문', '클리닉'
    ]

    TRENDSETTER_KEYWORDS = [
        '일상', '데일리', '뷰티', '리뷰', '하울', '언박싱', '추천',
        '꿀팁', '루틴', 'vlog', '브이로그', '룩북', '겟레디', 'GRWM'
    ]

    def classify(self, influencer: Dict) -> Dict:
        """Expert vs Trendsetter 분류"""
        bio = influencer.get('bio', '')
        captions = ' '.join([m.get('caption', '') for m in influencer.get('media', [])])
        text = f"{bio} {captions}"

        # 1. 키워드 매칭 점수
        expert_keyword_score = self._keyword_score(text, self.EXPERT_KEYWORDS)
        trendsetter_keyword_score = self._keyword_score(text, self.TRENDSETTER_KEYWORDS)

        # 2. TF-IDF 유사도 점수
        expert_tfidf_score = self._tfidf_similarity(text, 'expert')
        trendsetter_tfidf_score = self._tfidf_similarity(text, 'trendsetter')

        # 3. 이미지 분석 점수 (work_environment 기반)
        image_analysis = influencer.get('image_analysis', {})
        work_env = image_analysis.get('work_environment', '')
        expert_image_score = 1.0 if work_env in ['salon', 'home_salon'] else 0.3

        # Soft Voting Ensemble
        expert_score = (
            WEIGHT_KEYWORD * expert_keyword_score +
            WEIGHT_TFIDF * expert_tfidf_score +
            WEIGHT_IMAGE * expert_image_score
        )

        trendsetter_score = (
            WEIGHT_KEYWORD * trendsetter_keyword_score +
            WEIGHT_TFIDF * trendsetter_tfidf_score +
            WEIGHT_IMAGE * (1 - expert_image_score)
        )

        # 최종 분류
        if expert_score > trendsetter_score:
            return {
                'type': 'expert',
                'confidence': expert_score / (expert_score + trendsetter_score),
                'scores': {'expert': expert_score, 'trendsetter': trendsetter_score}
            }
        else:
            return {
                'type': 'trendsetter',
                'confidence': trendsetter_score / (expert_score + trendsetter_score),
                'scores': {'expert': expert_score, 'trendsetter': trendsetter_score}
            }
```

#### 4.2.3 RecommendationEvaluator (추천 품질 평가)

**학술적 기반**:
- NDCG: Järvelin, K., & Kekäläinen, J. (2002). "Cumulated Gain-based Evaluation"

```python
class RecommendationEvaluator:
    """추천 시스템 품질 평가기"""

    def evaluate_ndcg(self, recommendations: List, relevance_scores: List, k: int = 10) -> float:
        """NDCG@k 계산"""
        dcg = sum(
            rel / np.log2(i + 2)
            for i, rel in enumerate(relevance_scores[:k])
        )

        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = sum(
            rel / np.log2(i + 2)
            for i, rel in enumerate(ideal_scores[:k])
        )

        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_diversity(self, recommendations: List) -> float:
        """추천 다양성 평가 (카테고리, 스타일 분포)"""
        categories = [r.get('best_categories', '') for r in recommendations]
        unique_ratio = len(set(categories)) / len(categories)
        return unique_ratio
```

---

### 4.3 RAG Analyzer 상세

#### 4.3.1 InfluencerImageAnalyzer

**LLM 페르소나 생성** (`rag_analyzer.py:150-250`):
```python
class InfluencerImageAnalyzer:
    """LLM Vision 기반 인플루언서 이미지 분석 + 페르소나 생성"""

    PERSONA_GENERATION_PROMPT = """
    인플루언서 프로필을 분석하여 마케팅에 활용할 수 있는
    3-5단어의 한국어 페르소나를 생성해주세요.

    예시:
    - 청담 컬러 마스터
    - 데일리 뷰티 크리에이터
    - 손상모 케어 전문가
    - 힙한 스타일 큐레이터
    """

    def generate_persona_with_llm(self, profile_data: Dict) -> str:
        """GPT-4o-mini로 맞춤형 페르소나 생성"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.PERSONA_GENERATION_PROMPT},
                    {"role": "user", "content": json.dumps(profile_data, ensure_ascii=False)}
                ],
                temperature=0.8,  # 창의적인 페르소나 생성
                max_tokens=50
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback: 해시 기반 결정론적 페르소나
            return self._generate_fallback_persona(profile_data)

    def _generate_fallback_persona(self, profile_data: Dict) -> str:
        """LLM 실패 시 해시 기반 폴백 페르소나 생성"""
        username = profile_data.get('username', '')
        inf_type = profile_data.get('influencer_type', 'trendsetter')

        # 해시로 결정론적 선택
        hash_val = int(hashlib.md5(username.encode()).hexdigest(), 16)

        if inf_type == 'expert':
            personas = ['헤어 케어 전문가', '살롱 스타일리스트', '컬러 마스터', '두피 케어 전문가']
        else:
            personas = ['뷰티 크리에이터', '스타일 큐레이터', '데일리 뷰티러', '트렌드 세터']

        return personas[hash_val % len(personas)]
```

#### 4.3.2 InfluencerRAG

**ChromaDB 벡터 저장소** (`rag_analyzer.py:300-450`):
```python
class InfluencerRAG:
    """ChromaDB 기반 인플루언서 벡터 저장 및 검색"""

    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.collection = self.client.get_or_create_collection(
            name="influencers",
            embedding_function=self.embedding_fn
        )

    def index_influencer(self, influencer: Dict, analysis_result: Dict):
        """인플루언서 프로필을 벡터 DB에 저장"""
        username = influencer['username']

        # 임베딩용 텍스트 생성
        document = self._create_document(influencer, analysis_result)

        # 메타데이터 (필터링용)
        metadata = {
            'username': username,
            'influencer_type': influencer.get('influencer_type', 'trendsetter'),
            'fis_score': influencer.get('fis', {}).get('score', 0),
            'followers': influencer.get('followers', 0),
            'target_gender': analysis_result.get('target_gender', 'unisex'),
            'target_age': analysis_result.get('target_age', ''),
            'main_mood': analysis_result.get('main_mood', ''),
            'content_type': analysis_result.get('content_type', ''),
            'llm_persona': analysis_result.get('llm_persona', ''),
            'best_categories': analysis_result.get('best_categories', '')
        }

        self.collection.upsert(
            ids=[username],
            documents=[document],
            metadatas=[metadata]
        )
```

**Multi-Signal Hybrid Scoring** (`rag_analyzer.py:500-650`):
```python
# 하이브리드 스코어링 가중치
WEIGHT_VECTOR = 0.50  # 벡터 유사도 (의미적 관련성)
WEIGHT_FIS = 0.25     # FIS 신뢰도 (허수 필터링)
WEIGHT_RRF = 0.25     # RRF 순위 점수 (다중 신호 융합)
RRF_K = 60            # RRF 상수 (Cormack et al., 2009)
TEMPERATURE = 0.5     # Temperature Scaling 파라미터

def search_influencers(
    self,
    brand_name: str,
    product_type: str = None,
    campaign_description: str = None,
    influencer_type: str = None,
    target_gender: str = None,
    top_k: int = 10
) -> List[Dict]:
    """Multi-Signal Hybrid Scoring 기반 인플루언서 검색"""

    # 1. 쿼리 생성
    query = self._build_query(brand_name, product_type, campaign_description)

    # 2. ChromaDB 벡터 검색
    results = self.collection.query(
        query_texts=[query],
        n_results=top_k * 3,  # 필터링 고려하여 넉넉하게
        where=self._build_filter(influencer_type, target_gender)
    )

    # 3. Hybrid Scoring
    scored_results = []
    for i, (doc_id, distance, metadata) in enumerate(zip(
        results['ids'][0], results['distances'][0], results['metadatas'][0]
    )):
        # 벡터 유사도 (거리 → 유사도 변환)
        vector_score = 1 / (1 + distance)

        # FIS 신뢰도 (0-100 → 0-1 정규화)
        fis_score = metadata.get('fis_score', 50) / 100

        # RRF 순위 점수 (Reciprocal Rank Fusion)
        # RRF(d) = 1 / (k + rank(d)), k=60
        rrf_score = 1 / (RRF_K + i + 1)

        # Hybrid Score
        hybrid_score = (
            WEIGHT_VECTOR * vector_score +
            WEIGHT_FIS * fis_score +
            WEIGHT_RRF * rrf_score
        )

        # Temperature Scaling
        # P(i) = exp(s_i / T) / Σ exp(s_j / T)
        # T=0.5로 점수 분포를 더 뾰족하게 만듦
        scaled_score = np.exp(hybrid_score / TEMPERATURE)

        scored_results.append({
            'username': doc_id,
            'score': scaled_score,
            'hybrid_score': hybrid_score,
            'metadata': metadata,
            'component_scores': {
                'vector': vector_score,
                'fis': fis_score,
                'rrf': rrf_score
            }
        })

    # Temperature Scaling 정규화
    total = sum(r['score'] for r in scored_results)
    for r in scored_results:
        r['score'] = r['score'] / total

    # 점수 순 정렬 및 상위 k개 반환
    scored_results.sort(key=lambda x: x['score'], reverse=True)
    return scored_results[:top_k]
```

**RRF (Reciprocal Rank Fusion) 수식**:
```
RRF(d) = Σ 1 / (k + rank_i(d))

- k: 스무딩 상수 (k=60, Cormack et al., 2009)
- rank_i(d): i번째 순위 리스트에서 문서 d의 순위
```

**Temperature Scaling 수식**:
```
P(i) = exp(s_i / T) / Σ exp(s_j / T)

- T: Temperature 파라미터 (T=0.5)
- s_i: i번째 결과의 hybrid score
- T가 낮을수록 상위 결과에 더 높은 확률 부여
```

#### 4.3.3 InfluencerAnalysisManager

**배치 분석 파이프라인** (`rag_analyzer.py:700-850`):
```python
class InfluencerAnalysisManager:
    """인플루언서 분석 및 인덱싱 관리"""

    def __init__(self):
        self.analyzer = InfluencerImageAnalyzer()
        self.rag = InfluencerRAG()

    def analyze_and_index_all(
        self,
        influencers: List[Dict],
        force_reanalyze: bool = False
    ) -> Dict:
        """전체 인플루언서 분석 및 인덱싱"""
        stats = {'total': len(influencers), 'indexed': 0, 'skipped': 0, 'errors': 0}

        for influencer in influencers:
            username = influencer['username']

            # 이미 인덱싱되어 있으면 스킵
            if not force_reanalyze and self.rag.get_influencer(username):
                stats['skipped'] += 1
                continue

            try:
                # 1. 이미지 분석
                analysis = self.analyzer.analyze_profile(influencer)

                # 2. LLM 페르소나 생성
                analysis['llm_persona'] = self.analyzer.generate_persona_with_llm({
                    'username': username,
                    'influencer_type': influencer.get('influencer_type'),
                    'bio': influencer.get('bio'),
                    'best_categories': analysis.get('best_categories'),
                    'main_mood': analysis.get('main_mood')
                })

                # 3. RAG 인덱싱
                self.rag.index_influencer(influencer, analysis)
                stats['indexed'] += 1

            except Exception as e:
                stats['errors'] += 1
                logger.error(f"분석 실패 ({username}): {e}")

        return stats

    def search_influencers(self, **kwargs) -> List[Dict]:
        """인플루언서 검색 (RAG 래퍼)"""
        return self.rag.search_influencers(**kwargs)
```

---

### 4.4 API Routes 상세

#### 4.4.1 추천 API (`POST /api/recommend`)

**요청 처리 흐름**:
```
1. 요청 파라미터 검증
   └─ brand_name, product_type, description, target_gender, expert_count, trendsetter_count

2. RAG 검색 (타입별 분리)
   ├─ Expert 검색: _rag_manager.search_influencers(influencer_type='expert')
   └─ Trendsetter 검색: _rag_manager.search_influencers(influencer_type='trendsetter')

3. 마케팅 필터링
   ├─ Expert: 느슨한 필터 (전문성 기반, 연령 무관)
   │   └─ 성별: 반대 성별 '전용'만 제외
   └─ Trendsetter: 엄격한 필터 (타겟 페르소나 일치)
       ├─ 성별: 정확히 일치하거나 unisex
       └─ 연령: 정확일치 > 인접(±10년) > 미상

4. 점수 스케일링
   └─ match_score = 55 + hybrid_score * 43 (범위: 55% ~ 98%)

5. 추천 사유 생성
   └─ _generate_recommendation_reason(): 인플루언서 특성별 맞춤 사유
```

**마케팅 관점 필터링 로직** (`routes.py:162-306`):
```python
# ============================================================
# 마케팅 관점 필터링 로직
# ============================================================
#
# 전문가(Expert): 미용사, 원장님
# - 모든 연령대 고객을 시술하므로 연령 필터링 불필요
# - 성별도 대부분 여성/남성 모두 시술 가능
# - 중요한 건: 전문성, FIS 점수, 시술 결과 퀄리티
#
# 트렌드세터(Trendsetter): 일반 인플루언서
# - 본인이 직접 광고 모델이 됨 → 타겟 고객과 동일한 페르소나 필수
# - 30대 타겟 제품 → 30대 인플루언서 (공감대 형성)
# - 성별도 타겟과 일치해야 함 (여성 제품 → 여성 인플루언서)
# ============================================================

def filter_by_gender(results, target_gender, strict=False):
    """
    성별 필터링

    Args:
        strict: True면 정확히 일치해야 함 (트렌드세터용)
               False면 반대 성별만 제외 (전문가용)
    """
    ...

def filter_trendsetter_by_age(results, description):
    """
    트렌드세터 연령대 필터링 (마케팅 관점)

    우선순위:
    1. 정확일치 (30대 타겟 → 30대 인플루언서)
    2. 인접연령 (30대 타겟 → 20대, 40대)
    3. 미상 (연령대 정보 없음)
    """
    ...
```

**응답 형식**:
```json
{
  "brand_info": {
    "name": "려",
    "slogan": "두피부터 모발 끝까지, 한방 헤어케어",
    "aesthetic_style": "Natural",
    "core_values": ["두피건강", "탈모케어", "한방성분"]
  },
  "brand_analysis": "'려'는 자연친화적이고 건강한 이미지의 브랜드입니다...",
  "query": {
    "brand_name": "려",
    "product_type": "샴푸",
    "expert_count": 2,
    "trendsetter_count": 3
  },
  "total_results": 5,
  "recommendations": [
    {
      "rank": 1,
      "username": "hair_master_kim",
      "match_score": 92.3,
      "rag_profile": {
        "persona": "청담 컬러 마스터 | 트렌디한",
        "influencer_type": "expert",
        "target_age": "20대",
        "main_mood": "트렌디한",
        "best_categories": "염색-하이톤, 클리닉-트리트먼트"
      },
      "match_reason": "시술 전후 비교 콘텐츠를 전문적으로 제작하는 헤어 전문가입니다...",
      "influencer_data": { ... }
    }
  ]
}
```

#### 4.4.2 RAG 관리 API

**분석 및 인덱싱** (`POST /api/rag/analyze`):
```python
@router.post("/rag/analyze")
async def rag_analyze_influencers(request: RAGAnalyzeRequest):
    """
    인플루언서 이미지 분석 및 RAG 인덱싱

    Request:
        usernames: List[str] (선택, None이면 전체)
        force_reanalyze: bool (기본 False)

    Response:
        status: 'completed'
        stats: {total, indexed, skipped, errors}
    """
    manager = InfluencerAnalysisManager()
    stats = manager.analyze_and_index_all(targets, force_reanalyze=request.force_reanalyze)
    return {'status': 'completed', 'stats': stats}
```

**상태 확인** (`GET /api/rag/status`):
```python
@router.get("/rag/status")
async def rag_status():
    """
    RAG 시스템 상태 확인

    Response:
        available: bool
        indexed_count: int
        total_influencers: int
        indexed_usernames: List[str] (샘플 10개)
    """
```

---

### 4.5 제품 카테고리

#### 4.5.1 아모레퍼시픽 헤어 브랜드 제품 카테고리 (`config/products.py`)

| 카테고리 | 설명 | 타겟 | 브랜드 |
|----------|------|------|--------|
| 샴푸 | 모발 세정 제품 | Consumer | 려, 미쟝센, 라보에이치, 롱테이크 |
| 트리트먼트 | 모발 영양 및 집중 케어 | Consumer | 려, 미쟝센 |
| 에센스 | 모발 보호 및 윤기 케어 | Consumer | 미쟝센, 려 |
| 두피케어 | 두피 건강 및 탈모 예방 | Both | 려, 라보에이치 |
| 스타일링 | 헤어 스타일링 제품 | Consumer | 미쟝센, 아모스 프로페셔널 |
| 셀프염색 | 가정용 셀프 염색 제품 | Consumer | 미쟝센 |
| 살롱 케어 | 살롱 전용 세정 및 케어 | Professional | 아윤채, 아모스 프로페셔널 |
| 살롱 염색 | 살롱 전용 염색 제품 | Professional | 아윤채, 아모스 프로페셔널 |
| 살롱 펌 | 살롱 전용 펌 제품 | Professional | 아윤채, 아모스 프로페셔널 |
| 헤어 프래그런스 | 헤어 향수 및 라이프스타일 | Consumer | 롱테이크 |

#### 4.5.2 마케팅 접근법별 분류

```python
MARKETING_APPROACH_CATEGORIES = {
    'professional': ['살롱 케어', '살롱 염색', '살롱 펌'],
    'expert_oriented': ['두피케어'],
    'consumer': ['샴푸', '트리트먼트', '에센스', '스타일링', '셀프염색', '헤어 프래그런스', '기타']
}
```

#### 4.5.3 브랜드별 대표 제품 라인

```python
BRAND_PRODUCT_LINES = {
    '려': {
        'main_categories': ['샴푸', '트리트먼트', '두피케어'],
        'featured_lines': ['자양윤모', '청아', '흑운', '두피 세럼'],
        'expertise_level': 'medium_high'
    },
    '미쟝센': {
        'main_categories': ['샴푸', '트리트먼트', '에센스/세럼', '스타일링', '셀프염색'],
        'featured_lines': ['퍼펙트 세럼', '헬로버블', '샤이닝 에센스'],
        'expertise_level': 'low'
    },
    '라보에이치': {
        'main_categories': ['샴푸', '두피케어'],
        'featured_lines': ['탈모증상케어', '두피강화케어', '지성두피케어'],
        'expertise_level': 'high'
    },
    '아윤채': {
        'main_categories': ['살롱 샴푸/트리트먼트', '살롱 염색', '살롱 펌'],
        'featured_lines': ['PRO 샴푸', 'PRO 트리트먼트', 'TAKE HOME 라인'],
        'expertise_level': 'high'
    },
    '아모스 프로페셔널': {
        'main_categories': ['살롱 샴푸/트리트먼트', '살롱 염색', '살롱 펌', '스타일링'],
        'featured_lines': ['그린티 샴푸', '커리큘럼', '염색약', '펌제'],
        'expertise_level': 'high'
    },
    '롱테이크': {
        'main_categories': ['샴푸', '헤어 프래그런스'],
        'featured_lines': ['헤어 퍼퓸', '디퓨저', '샴푸'],
        'expertise_level': 'low'
    }
}
```

---

## 참고 문헌

1. Golbeck, J. (2015). "Benford's Law Applies to Online Social Networks." *PLOS ONE*, 10(8).
2. Salton, G., & McGill, M. J. (1983). *Introduction to Modern Information Retrieval*. McGraw-Hill.
3. Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods." *SIGIR*.
4. Järvelin, K., & Kekäläinen, J. (2002). "Cumulated Gain-based Evaluation of IR Techniques." *ACM TOIS*, 20(4).
5. Iglewicz, B., & Hoaglin, D. C. (1993). *How to Detect and Handle Outliers*. ASQC Quality Press.
6. Robertson, S. E., & Walker, S. (1994). "Some Simple Effective Approximations to the 2-Poisson Model for Probabilistic Weighted Retrieval." *SIGIR*.
7. Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." *ICML*.
8. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*.
