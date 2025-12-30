"""
RAG 기반 인플루언서 분석 및 매칭 모듈
=====================================

1. FIS (Fake Integrity Score)로 허수 계정 필터링
2. Expert/Trendsetter 분류
3. LLM Vision으로 인플루언서 이미지 분석 → 고유 특성 추출
4. 분석 결과를 벡터화하여 ChromaDB에 저장
5. 브랜드 + 제품 쿼리 → RAG 검색으로 최적 인플루언서 추천

Architecture:
    [인플루언서] → [FIS 필터링] → [Expert/Trendsetter 분류]
                                          ↓
    [이미지 10장] → [LLM Vision 분석] → [특성 텍스트] → [임베딩] → [ChromaDB]
                                                                    ↓
    [브랜드+제품 쿼리] → [임베딩] → [유사도 검색] → [FIS 가중치 적용] → [Top-K]
"""

import os
import json
import hashlib
from typing import Dict, List, Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# FIS, Classifier import
from .processors import FISCalculator, InfluencerClassifier


# ============================================================
# LLM Vision 분석 프롬프트
# ============================================================

IMAGE_ANALYSIS_PROMPT = """당신은 헤어/뷰티 마케팅 전문가입니다.
이 인플루언서의 콘텐츠 이미지들을 분석하여 마케팅 활용 관점에서 특성을 추출하세요.

**분석 관점:**
1. 이 인플루언서를 통해 어떤 제품을 홍보하면 좋을지
2. 어떤 타겟 오디언스에게 어필할 수 있는지
3. 어떤 분위기/무드의 캠페인에 적합한지
4. 어떤 메시지를 자연스럽게 전달할 수 있는지

**반드시 JSON 형식으로 응답하세요:**
```json
{
  "influencer_persona": "이 인플루언서를 한 문장으로 정의 (예: '세련된 직장인 여성의 데일리 라이프를 보여주는 트렌드세터')",

  "target_audience": {
    "primary_gender": "female/male/unisex",
    "age_range": "20대 초반/20대 중후반/30대/40대/MZ세대 등",
    "lifestyle": "대학생/직장인/주부/프리랜서 등",
    "keywords": ["키워드1", "키워드2", "키워드3"]
  },

  "content_characteristics": {
    "main_mood": "세련된/자연스러운/화려한/미니멀/고급스러운 등",
    "visual_style": "밝은/어두운/따뜻한/차가운/파스텔/모노톤 등",
    "content_type": "일상브이로그/스타일링튜토리얼/GRWM/리뷰/OOTD 등",
    "production_quality": "프로페셔널/세미프로/캐주얼"
  },

  "hair_characteristics": {
    "dominant_hair_styles": ["웨이브", "스트레이트", "볼륨" 등 2-3개],
    "hair_condition_focus": ["윤기", "볼륨", "손상케어", "두피" 등],
    "color_preference": "자연스러운 브라운/하이톤/애쉬계열/블랙 등"
  },

  "product_fit": {
    "best_categories": ["샴푸-볼륨", "트리트먼트-손상케어" 등 2-3개],
    "suitable_brands_style": ["프리미엄", "내추럴", "트렌디" 등],
    "price_range_fit": "프리미엄/중가/가성비"
  },

  "promotion_style": {
    "ad_approach": "자연스러운PPL/리뷰형/튜토리얼형/감성형",
    "storytelling_strength": ["일상연출", "before-after", "루틴소개" 등],
    "authenticity_level": "높음/중간/낮음 (광고 거부감 정도)"
  },

  "marketing_copy_seeds": [
    "이 인플루언서 콘텐츠에 어울리는 카피 문구 1",
    "이 인플루언서 콘텐츠에 어울리는 카피 문구 2",
    "이 인플루언서 콘텐츠에 어울리는 카피 문구 3"
  ],

  "best_campaign_fit": [
    "어울리는 캠페인 유형 1 (예: '여름철 두피케어 캠페인')",
    "어울리는 캠페인 유형 2",
    "어울리는 캠페인 유형 3"
  ]
}
```

JSON만 출력하세요."""


PROFILE_SUMMARY_PROMPT = """다음은 인플루언서의 이미지 분석 결과들입니다.
이를 종합하여 이 인플루언서의 마케팅 프로필을 하나의 텍스트로 요약하세요.

**분석 결과:**
{analyses}

**요약 형식:**
- 300-500자 정도의 자연스러운 문장
- 마케터가 이 인플루언서를 이해하고 활용할 수 있도록
- 어떤 브랜드/제품에 적합한지, 어떤 캠페인에 활용하면 좋은지 포함

요약문만 출력하세요 (JSON 아님):"""


# ============================================================
# 인플루언서 이미지 분석기
# ============================================================

class InfluencerImageAnalyzer:
    """LLM Vision 기반 인플루언서 이미지 분석"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = "gpt-4o-mini"

    def analyze_influencer(self, influencer: Dict, max_images: int = 5) -> Dict:
        """
        인플루언서의 이미지들을 분석하여 특성 추출

        Args:
            influencer: 인플루언서 데이터 (recent_posts 포함)
            max_images: 분석할 최대 이미지 수

        Returns:
            분석 결과 딕셔너리
        """
        username = influencer.get('username', 'unknown')
        posts = influencer.get('recent_posts', [])

        # 이미지 URL 추출
        image_urls = []
        for post in posts[:max_images]:
            url = post.get('media_url') or post.get('thumbnail_url') or post.get('image_url')
            if url:
                image_urls.append(url)

        # API 사용 가능하면 실제 분석
        if OPENAI_AVAILABLE and self.api_key and image_urls:
            try:
                return self._analyze_with_vision(username, image_urls, influencer)
            except Exception as e:
                print(f"Vision API 분석 실패 ({username}): {e}")

        # 폴백: 시뮬레이션 분석
        return self._simulate_analysis(username, influencer)

    def _analyze_with_vision(self, username: str, image_urls: List[str], influencer: Dict) -> Dict:
        """LLM Vision API로 실제 분석"""
        client = openai.OpenAI(api_key=self.api_key)

        # 이미지들을 함께 분석
        content = [{"type": "text", "text": IMAGE_ANALYSIS_PROMPT}]
        for url in image_urls[:5]:
            content.append({
                "type": "image_url",
                "image_url": {"url": url, "detail": "low"}
            })

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "헤어/뷰티 마케팅 전문가입니다. JSON으로만 응답합니다."},
                {"role": "user", "content": content}
            ],
            max_tokens=1500,
            temperature=0.3
        )

        result_text = response.choices[0].message.content.strip()

        # JSON 파싱
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        analysis = json.loads(result_text.strip())
        analysis['username'] = username
        analysis['analyzed_images'] = len(image_urls)
        analysis['analysis_method'] = 'vision_api'

        # 프로필 요약 생성
        analysis['profile_summary'] = self._generate_profile_summary(analysis)

        return analysis

    def _generate_profile_summary(self, analysis: Dict) -> str:
        """분석 결과를 마케팅 프로필 요약문으로 변환"""
        if not OPENAI_AVAILABLE or not self.api_key:
            return self._generate_summary_fallback(analysis)

        try:
            client = openai.OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": PROFILE_SUMMARY_PROMPT.format(
                        analyses=json.dumps(analysis, ensure_ascii=False, indent=2)
                    )}
                ],
                max_tokens=500,
                temperature=0.5
            )

            return response.choices[0].message.content.strip()
        except:
            return self._generate_summary_fallback(analysis)

    def _generate_summary_fallback(self, analysis: Dict) -> str:
        """프로필 요약 폴백"""
        persona = analysis.get('influencer_persona', '트렌디한 인플루언서')
        target = analysis.get('target_audience', {})
        product_fit = analysis.get('product_fit', {})

        summary = f"{persona}. "

        if target.get('primary_gender') and target.get('age_range'):
            summary += f"주요 타겟은 {target['age_range']} {target['primary_gender']}이며, "

        if product_fit.get('best_categories'):
            summary += f"{', '.join(product_fit['best_categories'][:2])} 제품 홍보에 적합합니다. "

        if analysis.get('marketing_copy_seeds'):
            summary += f"'{analysis['marketing_copy_seeds'][0]}' 같은 메시지를 자연스럽게 전달할 수 있습니다."

        return summary

    def _simulate_analysis(self, username: str, influencer: Dict) -> Dict:
        """시뮬레이션 분석 (API 없을 때) - Expert/Trendsetter 분류 반영"""
        bio = influencer.get('bio', '')
        inf_type = influencer.get('influencer_type', 'trendsetter')
        hash_val = int(hashlib.md5(f"{username}{bio}".encode()).hexdigest(), 16)

        # Expert와 Trendsetter에 따른 다른 시뮬레이션
        if inf_type == 'expert':
            return self._simulate_expert_analysis(username, influencer, hash_val)
        else:
            return self._simulate_trendsetter_analysis(username, influencer, hash_val)

    def _simulate_expert_analysis(self, username: str, influencer: Dict, hash_val: int) -> Dict:
        """Expert형 시뮬레이션 분석"""
        bio = influencer.get('bio', '')

        # Expert 관련 특성
        specialties = ["염색", "펌", "커트", "클리닉", "두피케어", "웨딩헤어"]
        found_specialties = [s for s in specialties if s in bio][:2] or ["헤어시술"]

        genders = ["female", "female", "unisex", "male"]
        ages = ["30대", "40대", "20대 중후반", "30~40대"]
        lifestyles = ["헤어 고민 있는 사람", "전문 시술 원하는 고객", "트렌드 민감층"]

        moods = ["전문적인", "고급스러운", "신뢰감 있는", "세련된"]
        content_types = ["시술결과", "튜토리얼", "전후비교", "헤어팁"]

        categories = [
            "샴푸-두피케어", "트리트먼트-손상복구", "클리닉-전문가용",
            "에센스-윤기", "스타일링-살롱급"
        ]
        brand_styles = ["프로페셔널", "프리미엄", "클리닉"]

        ad_approaches = ["튜토리얼형", "전문가추천", "비포애프터"]
        campaigns = [
            "전문가 추천 캠페인", "살롱 홈케어 캠페인", "손상모 복구 프로젝트",
            "두피케어 솔루션", "프로페셔널 라인 런칭"
        ]

        copies = [
            "살롱에서 쓰는 그대로",
            "전문가가 인정한 홈케어",
            "시술 후 관리의 정석",
            "헤어 디자이너의 선택",
            "전문 시술 효과 그대로 집에서"
        ]

        analysis = {
            "username": username,
            "influencer_type": "expert",
            "influencer_persona": f"헤어 전문가로서 {', '.join(found_specialties)} 분야 시술 결과를 공유하는 Expert",

            "target_audience": {
                "primary_gender": genders[hash_val % len(genders)],
                "age_range": ages[hash_val % len(ages)],
                "lifestyle": lifestyles[hash_val % len(lifestyles)],
                "keywords": ["전문시술", "살롱케어"] + found_specialties[:1]
            },

            "content_characteristics": {
                "main_mood": moods[hash_val % len(moods)],
                "visual_style": "프로페셔널",
                "content_type": content_types[hash_val % len(content_types)],
                "production_quality": "프로페셔널"
            },

            "hair_characteristics": {
                "dominant_hair_styles": found_specialties[:2] if found_specialties else ["펌", "염색"],
                "hair_condition_focus": ["손상케어", "두피건강", "윤기"],
                "color_preference": "다양한 컬러 시술"
            },

            "product_fit": {
                "best_categories": [
                    categories[hash_val % len(categories)],
                    categories[(hash_val + 2) % len(categories)]
                ],
                "suitable_brands_style": [
                    brand_styles[hash_val % len(brand_styles)],
                    brand_styles[(hash_val + 1) % len(brand_styles)]
                ],
                "price_range_fit": "프리미엄"
            },

            "promotion_style": {
                "ad_approach": ad_approaches[hash_val % len(ad_approaches)],
                "storytelling_strength": ["전문가 후기", "시술 과정", "비포애프터"],
                "authenticity_level": "높음"
            },

            "marketing_copy_seeds": [
                copies[hash_val % len(copies)],
                copies[(hash_val + 1) % len(copies)],
                copies[(hash_val + 2) % len(copies)]
            ],

            "best_campaign_fit": [
                campaigns[hash_val % len(campaigns)],
                campaigns[(hash_val + 2) % len(campaigns)],
                campaigns[(hash_val + 3) % len(campaigns)]
            ],

            "analyzed_images": 0,
            "analysis_method": "simulation_expert"
        }

        analysis['profile_summary'] = self._generate_summary_fallback(analysis)
        return analysis

    def _simulate_trendsetter_analysis(self, username: str, influencer: Dict, hash_val: int) -> Dict:
        """Trendsetter형 시뮬레이션 분석"""
        genders = ["female", "female", "female", "unisex", "male"]
        ages = ["20대 초반", "20대 중후반", "30대", "MZ세대", "20대"]
        lifestyles = ["직장인", "대학생", "프리랜서", "크리에이터", "주부"]

        moods = ["세련된", "자연스러운", "화려한", "미니멀", "고급스러운", "캐주얼"]
        styles = ["밝은", "따뜻한", "파스텔", "모노톤", "내추럴"]
        content_types = ["일상브이로그", "GRWM", "OOTD", "스타일링튜토리얼", "리뷰"]

        hair_styles = ["웨이브", "스트레이트", "볼륨펌", "레이어드컷", "내추럴"]
        hair_focuses = ["윤기", "볼륨", "손상케어", "두피건강", "스타일링유지"]

        categories = [
            "샴푸-데일리", "샴푸-볼륨", "샴푸-손상케어", "샴푸-두피케어",
            "트리트먼트-보습", "트리트먼트-손상복구", "에센스-윤기", "스타일링-볼륨"
        ]
        brand_styles = ["프리미엄", "내추럴", "트렌디", "프로페셔널"]

        ad_approaches = ["자연스러운PPL", "리뷰형", "튜토리얼형", "감성형"]
        storytelling = ["일상연출", "before-after", "루틴소개", "꿀팁공유"]

        copies = [
            "매일 쓰기 부담 없는 데일리 헤어 루틴",
            "사진만 봐도 윤기가 느껴지는",
            "바쁜 아침에도 5분이면 완성",
            "손대면 느껴지는 부드러움",
            "하루종일 유지되는 볼륨감",
            "나를 위한 작은 사치",
            "셀프케어가 특별해지는 순간"
        ]

        campaigns = [
            "여름철 두피케어 캠페인", "데일리 홈케어 캠페인", "MZ타겟 신제품 런칭",
            "손상모 복구 캠페인", "윤기 강조 캠페인", "볼륨업 캠페인",
            "프리미엄 라인 런칭", "가성비 라인 프로모션"
        ]

        analysis = {
            "username": username,
            "influencer_type": "trendsetter",
            "influencer_persona": f"{moods[hash_val % len(moods)]} 무드의 {lifestyles[hash_val % len(lifestyles)]} 타겟 인플루언서",

            "target_audience": {
                "primary_gender": genders[hash_val % len(genders)],
                "age_range": ages[hash_val % len(ages)],
                "lifestyle": lifestyles[hash_val % len(lifestyles)],
                "keywords": [
                    lifestyles[hash_val % len(lifestyles)],
                    moods[hash_val % len(moods)],
                    "데일리"
                ]
            },

            "content_characteristics": {
                "main_mood": moods[hash_val % len(moods)],
                "visual_style": styles[hash_val % len(styles)],
                "content_type": content_types[hash_val % len(content_types)],
                "production_quality": ["프로페셔널", "세미프로", "캐주얼"][hash_val % 3]
            },

            "hair_characteristics": {
                "dominant_hair_styles": [
                    hair_styles[hash_val % len(hair_styles)],
                    hair_styles[(hash_val + 1) % len(hair_styles)]
                ],
                "hair_condition_focus": [
                    hair_focuses[hash_val % len(hair_focuses)],
                    hair_focuses[(hash_val + 2) % len(hair_focuses)]
                ],
                "color_preference": ["자연스러운 브라운", "애쉬계열", "하이톤", "블랙"][hash_val % 4]
            },

            "product_fit": {
                "best_categories": [
                    categories[hash_val % len(categories)],
                    categories[(hash_val + 3) % len(categories)]
                ],
                "suitable_brands_style": [
                    brand_styles[hash_val % len(brand_styles)],
                    brand_styles[(hash_val + 1) % len(brand_styles)]
                ],
                "price_range_fit": ["프리미엄", "중가", "가성비"][hash_val % 3]
            },

            "promotion_style": {
                "ad_approach": ad_approaches[hash_val % len(ad_approaches)],
                "storytelling_strength": [
                    storytelling[hash_val % len(storytelling)],
                    storytelling[(hash_val + 1) % len(storytelling)]
                ],
                "authenticity_level": ["높음", "중간"][hash_val % 2]
            },

            "marketing_copy_seeds": [
                copies[hash_val % len(copies)],
                copies[(hash_val + 1) % len(copies)],
                copies[(hash_val + 2) % len(copies)]
            ],

            "best_campaign_fit": [
                campaigns[hash_val % len(campaigns)],
                campaigns[(hash_val + 2) % len(campaigns)],
                campaigns[(hash_val + 4) % len(campaigns)]
            ],

            "analyzed_images": 0,
            "analysis_method": "simulation_trendsetter"
        }

        analysis['profile_summary'] = self._generate_summary_fallback(analysis)
        return analysis


# ============================================================
# RAG 벡터 저장소
# ============================================================

class InfluencerRAG:
    """인플루언서 특성 기반 RAG 시스템 (FIS 점수 포함)"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(Path(__file__).parent.parent / "data" / "influencer_rag")
        self.collection_name = "influencer_profiles_v2"
        self.api_key = os.getenv("OPENAI_API_KEY")

        self._init_db()

    def _init_db(self):
        """ChromaDB 초기화"""
        if not CHROMADB_AVAILABLE:
            print("Warning: ChromaDB not available. Using fallback search.")
            self.client = None
            self.collection = None
            return

        # Persistent client
        self.client = chromadb.PersistentClient(path=self.db_path)

        # OpenAI 임베딩 함수
        if self.api_key:
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.api_key,
                model_name="text-embedding-3-small"
            )
        else:
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # 컬렉션 생성 또는 가져오기
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": "인플루언서 마케팅 프로필 (FIS 포함)"}
        )

    def add_influencer(self, username: str, analysis: Dict, fis_score: float = 80.0, inf_type: str = "trendsetter"):
        """인플루언서 분석 결과를 RAG에 추가 (FIS, 분류 포함)"""
        if not self.collection:
            return

        # 추가 메타데이터 구성 (FIS, 분류 포함)
        metadata = {
            "username": username,
            "influencer_type": inf_type,
            "fis_score": fis_score,
            "persona": analysis.get('influencer_persona', ''),
            "target_gender": analysis.get('target_audience', {}).get('primary_gender', ''),
            "target_age": analysis.get('target_audience', {}).get('age_range', ''),
            "main_mood": analysis.get('content_characteristics', {}).get('main_mood', ''),
            "content_type": analysis.get('content_characteristics', {}).get('content_type', ''),
            "best_categories": ','.join(analysis.get('product_fit', {}).get('best_categories', [])),
            "brand_styles": ','.join(analysis.get('product_fit', {}).get('suitable_brands_style', [])),
            "ad_approach": analysis.get('promotion_style', {}).get('ad_approach', ''),
            "campaigns": ','.join(analysis.get('best_campaign_fit', []))
        }

        # 검색용 확장 텍스트 생성
        searchable_text = self._create_searchable_text(analysis, inf_type)

        # Upsert (업데이트 또는 삽입)
        self.collection.upsert(
            ids=[username],
            documents=[searchable_text],
            metadatas=[metadata]
        )

    def _create_searchable_text(self, analysis: Dict, inf_type: str) -> str:
        """검색 가능한 텍스트 생성 (분류 정보 포함)"""
        parts = []

        # 인플루언서 유형
        if inf_type == "expert":
            parts.append("헤어전문가 미용사 살롱 시술전문")
        else:
            parts.append("트렌드세터 인플루언서 스타일 일상")

        # 페르소나
        if analysis.get('influencer_persona'):
            parts.append(analysis['influencer_persona'])

        # 타겟 오디언스
        target = analysis.get('target_audience', {})
        if target:
            parts.append(f"타겟: {target.get('primary_gender', '')} {target.get('age_range', '')} {target.get('lifestyle', '')}")
            parts.extend(target.get('keywords', []))

        # 콘텐츠 특성
        content = analysis.get('content_characteristics', {})
        if content:
            parts.append(f"무드: {content.get('main_mood', '')} 스타일: {content.get('visual_style', '')}")
            parts.append(content.get('content_type', ''))

        # 헤어 특성
        hair = analysis.get('hair_characteristics', {})
        if hair:
            parts.extend(hair.get('dominant_hair_styles', []))
            parts.extend(hair.get('hair_condition_focus', []))

        # 제품 적합성
        product_fit = analysis.get('product_fit', {})
        if product_fit:
            parts.extend(product_fit.get('best_categories', []))
            parts.extend(product_fit.get('suitable_brands_style', []))

        # 캠페인 적합성
        parts.extend(analysis.get('best_campaign_fit', []))

        # 마케팅 카피
        parts.extend(analysis.get('marketing_copy_seeds', []))

        # 프로필 요약
        if analysis.get('profile_summary'):
            parts.append(analysis['profile_summary'])

        return ' '.join(filter(None, parts))

    def search(self, query: str, top_k: int = 10, filters: Dict = None, min_fis: float = 60.0) -> List[Dict]:
        """
        쿼리로 적합한 인플루언서 검색 (FIS 필터링 적용)

        Args:
            query: 검색 쿼리 (브랜드 + 제품 + 캠페인 설명)
            top_k: 반환할 결과 수
            filters: 메타데이터 필터 (예: {"target_gender": "female"})
            min_fis: 최소 FIS 점수 (이 이상만 반환)

        Returns:
            검색 결과 리스트 (FIS 가중치 적용된 점수 포함)
        """
        if not self.collection:
            return self._fallback_search(query, top_k)

        # ChromaDB 쿼리 - 더 많이 가져와서 FIS로 필터링
        where_filter = None
        if filters:
            # 유효한 필터만 추출
            valid_filters = {k: v for k, v in filters.items() if v}
            if len(valid_filters) == 1:
                where_filter = valid_filters
            elif len(valid_filters) > 1:
                # 복수 조건은 $and로 묶어야 함
                where_filter = {"$and": [{k: v} for k, v in valid_filters.items()]}

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k * 3,  # FIS 필터링을 위해 더 많이 가져옴
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"]
        )

        # 결과 정리 및 FIS 가중치 적용
        output = []
        if results and results['ids'] and results['ids'][0]:
            for i, username in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                fis_score = metadata.get('fis_score', 80.0)

                # FIS 최소 점수 필터링
                if fis_score < min_fis:
                    continue

                # 기본 유사도 점수
                base_score = 1 - results['distances'][0][i] if results['distances'] else 0

                # FIS 가중치 적용 (FIS가 높을수록 점수 향상)
                fis_weight = fis_score / 100.0
                weighted_score = base_score * (0.7 + 0.3 * fis_weight)

                output.append({
                    'username': username,
                    'score': weighted_score,
                    'base_score': base_score,
                    'fis_score': fis_score,
                    'influencer_type': metadata.get('influencer_type', 'trendsetter'),
                    'metadata': metadata,
                    'matched_text': results['documents'][0][i][:200] if results['documents'] else ''
                })

        # FIS 가중 점수로 재정렬 후 top_k만 반환
        output.sort(key=lambda x: x['score'], reverse=True)
        return output[:top_k]

    def _fallback_search(self, query: str, top_k: int) -> List[Dict]:
        """ChromaDB 없을 때 폴백 검색"""
        return []

    def get_all_usernames(self) -> List[str]:
        """저장된 모든 인플루언서 username 반환"""
        if not self.collection:
            return []

        results = self.collection.get()
        return results['ids'] if results else []

    def get_influencer(self, username: str) -> Optional[Dict]:
        """특정 인플루언서 정보 조회"""
        if not self.collection:
            return None

        results = self.collection.get(ids=[username], include=["documents", "metadatas"])

        if results and results['ids']:
            return {
                'username': username,
                'metadata': results['metadatas'][0] if results['metadatas'] else {},
                'profile_text': results['documents'][0] if results['documents'] else ''
            }
        return None


# ============================================================
# 통합 분석 및 인덱싱 매니저
# ============================================================

class InfluencerAnalysisManager:
    """인플루언서 분석 및 RAG 인덱싱 통합 관리 (FIS + 분류 포함)"""

    # FIS 최소 기준 (이 이상이어야 인덱싱)
    MIN_FIS_FOR_INDEX = 50.0

    def __init__(self):
        self.analyzer = InfluencerImageAnalyzer()
        self.rag = InfluencerRAG()
        self.fis_calculator = FISCalculator()
        self.classifier = InfluencerClassifier()
        self.data_path = Path(__file__).parent.parent / "data"

    def analyze_and_index_all(self, influencers: List[Dict], force_reanalyze: bool = False) -> Dict:
        """
        모든 인플루언서 분석 및 RAG 인덱싱 (FIS 필터링 + 분류 포함)

        Args:
            influencers: 인플루언서 데이터 리스트
            force_reanalyze: True면 기존 분석 결과 무시하고 재분석

        Returns:
            처리 결과 통계
        """
        stats = {
            'total': len(influencers),
            'analyzed': 0,
            'indexed': 0,
            'skipped': 0,
            'filtered_by_fis': 0,
            'experts': 0,
            'trendsetters': 0,
            'errors': []
        }

        for inf in influencers:
            username = inf.get('username', '')

            try:
                # 기존 분석 결과 확인
                existing = self.rag.get_influencer(username)

                if existing and not force_reanalyze:
                    stats['skipped'] += 1
                    continue

                # 1. FIS - 기존 데이터 우선 사용, 없으면 계산
                if 'fis' in inf and 'score' in inf['fis']:
                    fis_score = inf['fis']['score']
                else:
                    fis_result = self.fis_calculator.calculate(inf)
                    fis_score = fis_result['fis_score']
                    inf['fis'] = {'score': fis_score, 'verdict': fis_result['verdict']}

                # FIS 최소 기준 미달 시 스킵 (기본값 0으로 변경하여 모두 인덱싱)
                # if fis_score < self.MIN_FIS_FOR_INDEX:
                #     stats['filtered_by_fis'] += 1
                #     continue

                # 2. Expert/Trendsetter - 기존 데이터 우선 사용
                if 'influencer_type' in inf:
                    inf_type = inf['influencer_type']
                else:
                    classification = self.classifier.classify(inf)
                    inf_type = classification['classification'].lower()
                    inf['influencer_type'] = inf_type

                if inf_type == 'expert':
                    stats['experts'] += 1
                else:
                    stats['trendsetters'] += 1

                # 3. 이미지 분석 수행
                analysis = self.analyzer.analyze_influencer(inf)
                stats['analyzed'] += 1

                # 4. RAG 인덱싱 (FIS, 분류 포함)
                self.rag.add_influencer(username, analysis, fis_score=fis_score, inf_type=inf_type)
                stats['indexed'] += 1

                # influencer 데이터에 분석 결과 추가
                inf['rag_analysis'] = analysis

            except Exception as e:
                stats['errors'].append({'username': username, 'error': str(e)})

        return stats

    def search_influencers(
        self,
        brand_name: str,
        product_type: str = None,
        campaign_description: str = None,
        target_gender: str = None,
        influencer_type: str = None,
        min_fis: float = 60.0,
        top_k: int = 10
    ) -> List[Dict]:
        """
        브랜드 + 제품으로 적합한 인플루언서 검색 (FIS + 분류 필터 지원)

        Args:
            brand_name: 브랜드명
            product_type: 제품 유형
            campaign_description: 캠페인 설명
            target_gender: 타겟 성별 필터
            influencer_type: 인플루언서 유형 필터 (expert/trendsetter)
            min_fis: 최소 FIS 점수
            top_k: 반환할 결과 수

        Returns:
            적합한 인플루언서 리스트
        """
        # 검색 쿼리 구성
        query_parts = [brand_name]

        if product_type:
            query_parts.append(product_type)

        if campaign_description:
            query_parts.append(campaign_description)

        query = ' '.join(query_parts)

        # 필터 구성
        filters = {}
        if target_gender:
            filters['target_gender'] = target_gender
        if influencer_type:
            filters['influencer_type'] = influencer_type

        # RAG 검색 (FIS 필터링 포함)
        results = self.rag.search(query, top_k=top_k, filters=filters, min_fis=min_fis)

        return results

    def get_stats(self) -> Dict:
        """현재 RAG 상태 통계"""
        all_usernames = self.rag.get_all_usernames()

        stats = {
            'total_indexed': len(all_usernames),
            'experts': 0,
            'trendsetters': 0
        }

        for username in all_usernames:
            profile = self.rag.get_influencer(username)
            if profile:
                inf_type = profile.get('metadata', {}).get('influencer_type', 'trendsetter')
                if inf_type == 'expert':
                    stats['experts'] += 1
                else:
                    stats['trendsetters'] += 1

        return stats

    def save_analyses_to_json(self, influencers: List[Dict], output_path: str = None):
        """분석 결과를 JSON 파일로 저장"""
        if not output_path:
            output_path = self.data_path / "influencers_rag_analyzed.json"

        # rag_analysis 필드만 포함하여 저장
        analyses = {}
        for inf in influencers:
            if 'rag_analysis' in inf:
                analyses[inf['username']] = {
                    'rag_analysis': inf['rag_analysis'],
                    'fis': inf.get('fis', {}),
                    'influencer_type': inf.get('influencer_type', 'trendsetter')
                }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analyses, f, ensure_ascii=False, indent=2)

        return str(output_path)


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RAG 기반 인플루언서 분석 시스템 테스트 (FIS + 분류)")
    print("=" * 60)

    # 테스트 데이터 로드
    data_path = Path(__file__).parent.parent / "data" / "influencers_data.json"

    if data_path.exists():
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        influencers = data.get('influencers', [])[:10]  # 테스트용 10명

        print(f"\n테스트 대상: {len(influencers)}명")

        # 분석 매니저 초기화
        manager = InfluencerAnalysisManager()

        # 분석 및 인덱싱
        print("\n1. 분석 및 인덱싱 (FIS 필터링 + 분류)...")
        stats = manager.analyze_and_index_all(influencers, force_reanalyze=True)
        print(f"   - 분석: {stats['analyzed']}명")
        print(f"   - 인덱싱: {stats['indexed']}명")
        print(f"   - FIS 필터링: {stats['filtered_by_fis']}명")
        print(f"   - Expert: {stats['experts']}명")
        print(f"   - Trendsetter: {stats['trendsetters']}명")
        print(f"   - 스킵: {stats['skipped']}명")

        # 검색 테스트
        print("\n2. 검색 테스트...")
        results = manager.search_influencers(
            brand_name="려",
            product_type="샴푸",
            campaign_description="30대 여성 두피 케어",
            min_fis=60.0,
            top_k=5
        )

        print(f"   검색 결과: {len(results)}명")
        for r in results:
            print(f"   - @{r['username']} (score: {r['score']:.3f}, FIS: {r['fis_score']:.1f}, type: {r['influencer_type']})")
            print(f"     {r['metadata'].get('persona', '')[:50]}...")
    else:
        print(f"데이터 파일 없음: {data_path}")
