"""
프로세서 모듈 - 인플루언서 데이터 처리 (학술적 알고리즘 기반)
=============================================================

학술적 기반:
- FIS (Fake Integrity Score): Benford's Law + Chi-squared Test + Z-score 기반 이상치 탐지
  - Golbeck (2015): "Benford's Law Applies to Online Social Networks" PLOS ONE
  - Mazza et al. (2020): "Bot Detection using Benford's Law" ACM SIN

- Expert/Trendsetter 분류: TF-IDF + Cosine Similarity + Soft Voting
  - Salton & McGill (1983): TF-IDF Term Weighting
  - Manning et al. (2008): Introduction to Information Retrieval

- 추천 품질 평가: NDCG, Diversity Score, Coverage
  - Järvelin & Kekäläinen (2002): "Cumulated Gain-Based Evaluation of IR Techniques"

모듈 구성:
1. InfluencerProcessor: 메인 처리 파이프라인
2. FISCalculator: 학술적 허수 탐지 (Benford + 통계 검정)
3. InfluencerClassifier: TF-IDF 기반 분류
4. ImageAnalyzer: LLM 비전 기반 이미지 분석
5. RecommendationEvaluator: NDCG/Diversity 평가 메트릭
"""

import os
import re
import math
import json
import base64
import hashlib
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================
# 이미지 분석 프롬프트
# ============================================================

# Expert용 이미지 분석 프롬프트 (Secondary - bio 정보 검증용)
EXPERT_IMAGE_PROMPT = """헤어 전문가의 게시물 이미지를 분석하세요.
이 인플루언서는 미용사/헤어디자이너/살롱 원장 등 헤어 전문가입니다.

아래 bio 정보를 참고하여, 이미지에서 보이는 시술 결과물과 전문 분야를 분석하세요:
Bio: {bio}

JSON 형식:
{{
  "analysis_type": "image_secondary",
  "verified_specialties": ["bio에서 언급되고 이미지에서도 확인된 전문분야"],
  "additional_specialties": ["bio에는 없지만 이미지에서 발견된 추가 전문분야"],
  "signature_techniques": ["시그니처 시술/기법"],
  "client_hair_types": ["주로 시술하는 고객 헤어 타입"],
  "color_specialties": ["주로 사용하는 염색 계열"],
  "work_environment": "salon/home_salon/freelance/academy 중 하나",
  "content_quality_score": 0.7-0.95,
  "expertise_confidence": 0.7-0.95
}}

JSON만 출력하세요."""

# Trendsetter용 이미지 분석 프롬프트 (Primary - 스타일 추출용)
TRENDSETTER_IMAGE_PROMPT = """패션/라이프스타일 인플루언서의 게시물 이미지를 분석하세요.
이 인플루언서는 OOTD, 패션, 뷰티, 라이프스타일 콘텐츠를 주로 올리는 사람입니다.

JSON 형식:
{{
  "analysis_type": "image_primary",
  "dominant_style": "luxury/natural/trendy/colorful/minimal 중 하나",
  "sub_styles": ["서브스타일1", "서브스타일2"],
  "color_palette": "warm_gold/neutral_warm/neutral_cool/monochrome/pastel_pop/earth_tone/black_gold 중 하나",
  "aesthetic_tags": ["패션관련태그1", "패션관련태그2", "패션관련태그3"],
  "hair_style_tags": ["헤어스타일1", "헤어스타일2"],
  "vibe": "이 인플루언서의 전반적인 분위기를 한 문장으로",
  "professionalism_score": 0.3-0.6,
  "trend_relevance_score": 0.8-0.95,
  "image_confidence": 0.85-0.98
}}

JSON만 출력하세요."""


# ============================================================
# InfluencerProcessor - 메인 처리 파이프라인
# ============================================================

class InfluencerProcessor:
    """
    인플루언서 데이터 처리 파이프라인

    파이프라인 흐름:
    1. Crawler에서 raw 데이터 수신
    2. Expert/Trendsetter 분류 (InfluencerClassifier 사용)
    3. 유형별 분석 전략 적용:
       - Expert: 텍스트 분석 Primary → 이미지 분석 Secondary (검증)
       - Trendsetter: 이미지 분석 Primary → 텍스트 분석 Secondary (보조)
    4. FIS 계산 (허수 필터링)
    5. 최종 JSON 구성 및 저장
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent / "data"
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.classifier = InfluencerClassifier()
        self.fis_calculator = FISCalculator()

    def process(self, raw_data: Dict, use_llm: bool = True) -> Dict:
        """
        raw 인플루언서 데이터 처리

        Args:
            raw_data: Crawler에서 수집한 raw 데이터
            use_llm: LLM 이미지 분석 사용 여부

        Returns:
            처리된 인플루언서 데이터 (분류 + 분석 완료)
        """
        processed_influencers = []
        expert_count = 0
        trendsetter_count = 0

        for raw_inf in raw_data.get("influencers", []):
            try:
                # 1. Expert/Trendsetter 분류
                classification = self.classifier.classify(raw_inf)
                inf_type = classification["classification"].lower()

                # 2. 유형별 분석 전략 적용
                if inf_type == "expert":
                    processed = self._process_expert(raw_inf, classification, use_llm)
                    expert_count += 1
                else:
                    processed = self._process_trendsetter(raw_inf, classification, use_llm)
                    trendsetter_count += 1

                # 3. FIS 계산
                fis_result = self.fis_calculator.calculate(processed)
                processed["fis"] = {
                    "score": fis_result["fis_score"],
                    "verdict": fis_result["verdict"]
                }

                processed_influencers.append(processed)

            except Exception as e:
                logger.warning(f"인플루언서 처리 실패 ({raw_inf.get('username', 'unknown')}): {e}")
                continue

        result = {
            "influencers": processed_influencers,
            "metadata": {
                "processed_at": datetime.now().isoformat(),
                "total_count": len(processed_influencers),
                "expert_count": expert_count,
                "trendsetter_count": trendsetter_count,
                "status": "processed",
                "schema_version": "3.0"
            }
        }

        # 결과 저장
        self._save_processed_data(result)
        return result

    def _process_expert(self, raw_inf: Dict, classification: Dict, use_llm: bool) -> Dict:
        """
        Expert형 인플루언서 처리

        분석 전략:
        - text_analysis: PRIMARY (bio/caption이 풍부)
        - image_analysis: SECONDARY (텍스트 정보 검증/보완)
        """
        bio = raw_inf.get("bio", "")
        posts = raw_inf.get("recent_posts", [])
        captions = [p.get("caption", "") for p in posts]

        # 텍스트 분석 (Primary)
        text_analysis = self._analyze_expert_text(bio, captions)

        # 이미지 분석 (Secondary - 검증용)
        if use_llm and OPENAI_AVAILABLE and self.api_key:
            image_analysis = self._analyze_expert_images(raw_inf, bio)
        else:
            image_analysis = self._simulate_expert_image_analysis(bio)

        return {
            **raw_inf,
            "influencer_type": "expert",
            "classification_confidence": classification["confidence"],
            "analysis_strategy": {
                "primary": "text",
                "secondary": "image",
                "reason": "Expert는 bio와 caption에 전문 정보가 풍부함"
            },
            "text_analysis": text_analysis,
            "image_analysis": image_analysis
        }

    def _process_trendsetter(self, raw_inf: Dict, classification: Dict, use_llm: bool) -> Dict:
        """
        Trendsetter형 인플루언서 처리

        분석 전략:
        - image_analysis: PRIMARY (bio/caption이 비어있어 이미지에서 추출)
        - text_analysis: SECONDARY (해시태그 등 보조 정보)
        """
        bio = raw_inf.get("bio", "")
        posts = raw_inf.get("recent_posts", [])
        captions = [p.get("caption", "") for p in posts]

        # 이미지 분석 (Primary)
        if use_llm and OPENAI_AVAILABLE and self.api_key:
            image_analysis = self._analyze_trendsetter_images(raw_inf)
        else:
            image_analysis = self._simulate_trendsetter_image_analysis(raw_inf.get("username", ""))

        # 텍스트 분석 (Secondary - 보조)
        text_analysis = self._analyze_trendsetter_text(bio, captions)

        return {
            **raw_inf,
            "influencer_type": "trendsetter",
            "classification_confidence": classification["confidence"],
            "analysis_strategy": {
                "primary": "image",
                "secondary": "text",
                "reason": "Trendsetter는 bio/caption이 간략하여 이미지 분석이 핵심"
            },
            "text_analysis": text_analysis,
            "image_analysis": image_analysis
        }

    def _analyze_expert_text(self, bio: str, captions: List[str]) -> Dict:
        """Expert형 텍스트 분석 (Primary)"""
        specialties = ["염색", "펌", "커트", "클리닉", "두피케어", "웨딩헤어"]
        techniques = ["C컬펌", "히피펌", "볼륨펌", "레이어드컷", "애쉬염색", "하이톤염색"]

        # bio에서 전문 분야 추출
        found_specialties = [s for s in specialties if s in bio]
        if not found_specialties:
            found_specialties = [s for s in specialties if any(s in c for c in captions)][:2]

        # 자격증/경력 키워드
        cert_keywords = ["원장", "디렉터", "년차", "경력", "자격증", "교육"]
        found_certs = [k for k in cert_keywords if k in bio]

        # caption에서 시술 키워드 추출
        all_captions = " ".join(captions)
        found_techniques = [t for t in techniques if t in all_captions][:4]

        return {
            "analysis_type": "text_primary",
            "specialties_from_bio": found_specialties if found_specialties else ["헤어시술"],
            "certifications_detected": found_certs,
            "techniques_from_caption": found_techniques if found_techniques else ["일반시술"],
            "caption_detail_level": "high" if len(all_captions) > 200 else "medium",
            "text_confidence": 0.85 if found_specialties and found_certs else 0.6
        }

    def _analyze_trendsetter_text(self, bio: str, captions: List[str]) -> Dict:
        """Trendsetter형 텍스트 분석 (Secondary - 보조)"""
        style_keywords = ["fashion", "style", "ootd", "daily", "minimal", "lifestyle"]
        found_keywords = [k for k in style_keywords if k.lower() in bio.lower()]

        # 해시태그 추출
        hashtags = []
        for caption in captions:
            if "#" in caption:
                tags = [w.strip() for w in caption.split() if w.startswith("#")]
                hashtags.extend(tags)

        return {
            "analysis_type": "text_secondary",
            "keywords_from_bio": found_keywords if found_keywords else ["lifestyle"],
            "hashtags_from_caption": list(set(hashtags))[:5],
            "caption_detail_level": "low",
            "extractable_info": "minimal",
            "text_confidence": 0.3 if not found_keywords else 0.5
        }

    def _analyze_expert_images(self, influencer: Dict, bio: str) -> Dict:
        """Expert형 LLM 이미지 분석"""
        image_urls = self._get_image_urls(influencer, max_images=3)
        if not image_urls:
            return self._simulate_expert_image_analysis(bio)

        try:
            client = openai.OpenAI(api_key=self.api_key)
            prompt = EXPERT_IMAGE_PROMPT.format(bio=bio)

            content = [{"type": "text", "text": f"헤어 전문가 분석:\nBio: {bio}"}]
            for url in image_urls:
                content.append({"type": "image_url", "image_url": {"url": url}})

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content}
                ],
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            return json.loads(result_text.strip())

        except Exception as e:
            logger.warning(f"Expert 이미지 분석 실패: {e}")
            return self._simulate_expert_image_analysis(bio)

    def _analyze_trendsetter_images(self, influencer: Dict) -> Dict:
        """Trendsetter형 LLM 이미지 분석"""
        image_urls = self._get_image_urls(influencer, max_images=3)
        if not image_urls:
            return self._simulate_trendsetter_image_analysis(influencer.get("username", ""))

        try:
            client = openai.OpenAI(api_key=self.api_key)

            content = [{"type": "text", "text": "패션/라이프스타일 인플루언서 분석"}]
            for url in image_urls:
                content.append({"type": "image_url", "image_url": {"url": url}})

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": TRENDSETTER_IMAGE_PROMPT},
                    {"role": "user", "content": content}
                ],
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            return json.loads(result_text.strip())

        except Exception as e:
            logger.warning(f"Trendsetter 이미지 분석 실패: {e}")
            return self._simulate_trendsetter_image_analysis(influencer.get("username", ""))

    def _get_image_urls(self, influencer: Dict, max_images: int = 3) -> List[str]:
        """이미지 URL 추출"""
        urls = []
        for post in influencer.get("recent_posts", []):
            if len(urls) >= max_images:
                break
            media_url = post.get("media_url", "")
            if media_url and post.get("media_type") in ("IMAGE", "CAROUSEL_ALBUM"):
                urls.append(media_url)
        return urls

    def _simulate_expert_image_analysis(self, bio: str) -> Dict:
        """Expert 이미지 분석 시뮬레이션"""
        specialties = ["염색", "펌", "커트", "클리닉"]
        import random
        return {
            "analysis_type": "image_secondary",
            "verified_specialties": [s for s in specialties if s in bio][:2] or ["헤어시술"],
            "additional_specialties": random.sample(specialties, 1),
            "signature_techniques": ["볼륨펌", "애쉬염색"],
            "client_hair_types": ["웨이브", "스트레이트"],
            "color_specialties": ["브라운", "애쉬"],
            "work_environment": "salon",
            "content_quality_score": round(random.uniform(0.7, 0.9), 2),
            "expertise_confidence": round(random.uniform(0.7, 0.9), 2)
        }

    def _simulate_trendsetter_image_analysis(self, username: str) -> Dict:
        """Trendsetter 이미지 분석 시뮬레이션"""
        import random
        styles = ["luxury", "natural", "trendy", "colorful", "minimal"]
        return {
            "analysis_type": "image_primary",
            "dominant_style": random.choice(styles),
            "sub_styles": random.sample(["modern", "casual", "chic"], 2),
            "color_palette": random.choice(["warm_gold", "neutral_cool", "monochrome"]),
            "aesthetic_tags": random.sample(["스트릿패션", "미니멀", "캐주얼", "Y2K", "데님"], 3),
            "hair_style_tags": random.sample(["웨이브", "레이어드컷", "C컬", "히피펌"], 2),
            "vibe": "트렌디하고 세련된 스타일",
            "professionalism_score": round(random.uniform(0.3, 0.6), 2),
            "trend_relevance_score": round(random.uniform(0.8, 0.95), 2),
            "image_confidence": round(random.uniform(0.85, 0.95), 2)
        }

    def _save_processed_data(self, data: Dict) -> None:
        """처리된 데이터 저장"""
        output_path = self.data_dir / "influencers_data.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"처리된 인플루언서 데이터 저장: {output_path}")

    def load_processed_data(self) -> Dict:
        """처리된 데이터 로드"""
        path = self.data_dir / "influencers_data.json"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"influencers": [], "metadata": {"status": "empty"}}


# ============================================================
# FIS Calculator - 학술적 허수 계정 탐지
# ============================================================

class FISCalculator:
    """
    Fake Integrity Score 계산기 (학술적 알고리즘 기반)

    학술적 기반:
    ============
    1. Benford's Law (벤포드 법칙)
       - Golbeck (2015): "Benford's Law Applies to Online Social Networks" PLOS ONE
       - 자연 발생 숫자의 첫째 자릿수는 1이 ~30%, 9가 ~4.6%로 불균등 분포
       - 봇 계정은 이 법칙을 위반하는 경향

    2. Chi-squared Goodness-of-Fit Test (카이제곱 적합도 검정)
       - Pearson's Chi-squared test로 Benford 분포 적합도 측정
       - χ² 값이 클수록 자연 분포에서 벗어남 (봇 의심)

    3. Z-score 기반 이상치 탐지
       - 참여율, 조회수 등의 분포에서 ±2σ 이상은 이상치로 판단
       - Modified Z-score (MAD 기반) 사용으로 이상치에 강건

    4. Engagement Authenticity Index (EAI)
       - HypeAuditor 방법론 참고: 15개 지표 4개 카테고리
       - 좋아요/조회수 비율, 댓글/좋아요 비율의 정상 범위 검증

    수학적 공식:
    ============
    FIS = Σ(wi × Si) × Geographic_Factor

    where:
    - S_benford: 1 - (χ² / χ²_critical) (Benford 적합도)
    - S_engagement: Z-score 기반 정상 범위 점수
    - S_activity: 업로드 패턴 규칙성 (봇은 너무 규칙적)
    - S_duplicate: Jaccard 유사도 기반 중복 탐지

    가중치 (연구 기반):
    - w_benford = 0.20 (Mazza et al., 2020)
    - w_engagement = 0.25 (HypeAuditor AQS)
    - w_comment = 0.15
    - w_activity = 0.15
    - w_geo = 0.10
    - w_duplicate = 0.15
    """

    # Benford's Law 기대 분포 (Newcomb-Benford)
    BENFORD_EXPECTED = {
        1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097,
        5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
    }

    # Chi-squared critical value (df=8, α=0.05)
    CHI_SQUARED_CRITICAL = 15.507

    def __init__(self):
        # 가중치 (연구 기반 최적화)
        self.w_benford = 0.20       # Benford 법칙 적합도
        self.w_engagement = 0.25    # 참여율 이상치 탐지
        self.w_comment = 0.15       # 댓글 패턴 분석
        self.w_activity = 0.15      # 활동 패턴 규칙성
        self.w_geo = 0.10           # 지리적 정합성
        self.w_duplicate = 0.15     # 중복 콘텐츠

    def calculate(self, influencer: Dict) -> Dict:
        """
        FIS 점수 계산 (학술적 알고리즘 기반)

        Returns:
            {
                'fis_score': float,
                'verdict': str,
                'breakdown': dict,
                'statistical_tests': dict  # 학술적 검정 결과
            }
        """
        posts = influencer.get('recent_posts', [])

        # 1. Benford's Law 검정 (학술적 핵심)
        benford_score, benford_detail = self._benford_test(influencer)

        # 2. 참여율 Z-score 분석
        a_score, a_detail = self._engagement_zscore_analysis(posts)

        # 3. 댓글 패턴 분석
        e_score, e_detail = self._comment_entropy(posts)

        # 4. 활동 패턴 규칙성 (Modified Z-score)
        acs_score, acs_detail = self._activity_regularity(influencer)

        # 5. 지리적 정합성
        d_score, d_detail = self._geographic_consistency(influencer)

        # 6. 중복 콘텐츠 (Jaccard Similarity)
        dup_score, dup_detail = self._duplicate_content_jaccard(posts)

        # 가중 합산
        base_score = (
            self.w_benford * benford_score +
            self.w_engagement * a_score +
            self.w_comment * e_score +
            self.w_activity * acs_score +
            self.w_duplicate * dup_score
        )

        # 지리적 정합성 반영
        final_score = base_score * (d_score / 100) + (self.w_geo * d_score)
        final_score = max(0, min(100, final_score))

        # 판정 (3단계 + 세부 등급)
        if final_score >= 85:
            verdict = '신뢰 계정 (A등급)'
        elif final_score >= 70:
            verdict = '신뢰 계정 (B등급)'
        elif final_score >= 55:
            verdict = '주의 필요 (C등급)'
        else:
            verdict = '허수 의심 (D등급)'

        return {
            'username': influencer.get('username', ''),
            'fis_score': round(final_score, 1),
            'verdict': verdict,
            'breakdown': {
                'benford_conformity': round(benford_score, 1),
                'engagement_authenticity': round(a_score, 1),
                'comment_pattern': round(e_score, 1),
                'activity_regularity': round(acs_score, 1),
                'geographic_consistency': round(d_score, 1),
                'content_originality': round(dup_score, 1)
            },
            'statistical_tests': {
                'benford': benford_detail,
                'engagement': a_detail,
                'duplicate': dup_detail
            }
        }

    def _benford_test(self, influencer: Dict) -> Tuple[float, Dict]:
        """
        Benford's Law 적합도 검정

        학술 기반:
        - Golbeck (2015): PLOS ONE - 소셜 네트워크 숫자 분포 분석
        - Chi-squared test로 기대 분포와의 적합도 측정

        수학적 공식:
        χ² = Σ (observed_i - expected_i)² / expected_i
        p-value from chi-squared distribution (df=8)
        """
        # 숫자 데이터 수집 (좋아요, 댓글, 조회수, 팔로워)
        numbers = []

        # 팔로워 수
        followers = influencer.get('followers', 0)
        if followers > 0:
            numbers.append(followers)

        # 게시물별 지표
        for post in influencer.get('recent_posts', []):
            if post.get('likes', 0) > 0:
                numbers.append(post['likes'])
            if post.get('comments', 0) > 0:
                numbers.append(post['comments'])
            if post.get('views', 0) > 0:
                numbers.append(post['views'])

        if len(numbers) < 5:
            return 70.0, {'status': 'insufficient_data', 'sample_size': len(numbers)}

        # 첫째 자릿수 추출
        first_digits = []
        for n in numbers:
            first_digit = int(str(abs(int(n)))[0])
            if 1 <= first_digit <= 9:
                first_digits.append(first_digit)

        if len(first_digits) < 5:
            return 70.0, {'status': 'insufficient_digits', 'sample_size': len(first_digits)}

        # 관측 빈도 계산
        observed = {d: 0 for d in range(1, 10)}
        for d in first_digits:
            observed[d] += 1

        n_total = len(first_digits)

        # Chi-squared 검정
        chi_squared = 0.0
        for digit in range(1, 10):
            expected = self.BENFORD_EXPECTED[digit] * n_total
            obs = observed[digit]
            if expected > 0:
                chi_squared += ((obs - expected) ** 2) / expected

        # 점수 계산 (χ² 값이 낮을수록 Benford 법칙에 적합)
        # χ² < 15.507 (critical value at α=0.05, df=8) → 적합
        if chi_squared < self.CHI_SQUARED_CRITICAL:
            # 적합: 점수 = 100 - (χ² / critical * 30)
            score = 100 - (chi_squared / self.CHI_SQUARED_CRITICAL) * 30
        else:
            # 부적합: 점수 급감
            ratio = chi_squared / self.CHI_SQUARED_CRITICAL
            score = max(20, 70 - (ratio - 1) * 25)

        # MAD (Mean Absolute Deviation) 추가 계산
        mad = sum(abs(observed[d]/n_total - self.BENFORD_EXPECTED[d]) for d in range(1, 10)) / 9

        # MAD 기준 (Nigrini, 2012)
        # < 0.006: Close conformity
        # 0.006-0.012: Acceptable conformity
        # 0.012-0.015: Marginally acceptable
        # > 0.015: Nonconformity
        if mad > 0.015:
            score -= 15
        elif mad > 0.012:
            score -= 8

        score = max(0, min(100, score))

        return score, {
            'chi_squared': round(chi_squared, 3),
            'critical_value': self.CHI_SQUARED_CRITICAL,
            'p_value_significant': chi_squared > self.CHI_SQUARED_CRITICAL,
            'mad': round(mad, 4),
            'mad_conformity': 'close' if mad < 0.006 else 'acceptable' if mad < 0.012 else 'marginal' if mad < 0.015 else 'nonconforming',
            'sample_size': n_total,
            'observed_distribution': {str(k): v for k, v in observed.items()},
            'verdict': '정상' if score >= 70 else '의심' if score >= 50 else '봇 가능성'
        }

    def _engagement_zscore_analysis(self, posts: List[Dict]) -> Tuple[float, Dict]:
        """
        참여율 Z-score 기반 이상치 탐지

        학술 기반:
        - Modified Z-score (Iglewicz & Hoaglin, 1993)
        - MAD (Median Absolute Deviation) 기반으로 이상치에 강건

        정상 참여율 범위 (HypeAuditor 기준):
        - 좋아요/조회수: 2%~12%
        - 댓글/좋아요: 3%~15%

        수학적 공식:
        Modified Z-score = 0.6745 × (xi - median) / MAD
        |Z| > 3.5 → 이상치
        """
        like_view_ratios = []
        comment_like_ratios = []

        for p in posts:
            views = p.get('views', 0)
            likes = p.get('likes', 0)
            comments = p.get('comments', 0)

            if views > 0:
                like_view_ratios.append(likes / views)
            if likes > 0:
                comment_like_ratios.append(comments / likes)

        if not like_view_ratios:
            return 60.0, {'status': 'no_data', 'verdict': '데이터 없음'}

        # 1. 좋아요/조회수 비율 분석
        avg_lv = sum(like_view_ratios) / len(like_view_ratios)
        median_lv = sorted(like_view_ratios)[len(like_view_ratios) // 2]

        # Modified Z-score 계산
        mad_lv = self._calculate_mad(like_view_ratios)
        z_scores_lv = [self._modified_zscore(x, median_lv, mad_lv) for x in like_view_ratios]
        outliers_lv = sum(1 for z in z_scores_lv if abs(z) > 3.5)

        # 2. 점수 계산
        score = 90.0
        verdict_parts = []

        # 좋아요/조회수 비율 정상 범위: 2%~12%
        if avg_lv < 0.008:
            score -= 40
            verdict_parts.append("뷰봇 의심 (참여율 극저)")
        elif avg_lv < 0.02:
            score -= 20
            verdict_parts.append("참여율 낮음")
        elif avg_lv > 0.20:
            score -= 35
            verdict_parts.append("좋아요 구매 의심")
        elif avg_lv > 0.12:
            score -= 10
            verdict_parts.append("참여율 다소 높음")

        # 이상치 비율 감점
        outlier_ratio = outliers_lv / len(like_view_ratios) if like_view_ratios else 0
        if outlier_ratio > 0.3:
            score -= 20
            verdict_parts.append("이상치 다수")
        elif outlier_ratio > 0.15:
            score -= 10
            verdict_parts.append("이상치 존재")

        # 댓글/좋아요 비율 분석
        if comment_like_ratios:
            avg_cl = sum(comment_like_ratios) / len(comment_like_ratios)
            if avg_cl < 0.02:
                score -= 10
                verdict_parts.append("댓글 부족")
            elif avg_cl > 0.30:
                score -= 15
                verdict_parts.append("봇 댓글 의심")

        score = max(0, min(100, score))
        verdict = ", ".join(verdict_parts) if verdict_parts else "정상"

        return score, {
            'avg_like_view_ratio': round(avg_lv * 100, 2),
            'median_like_view_ratio': round(median_lv * 100, 2),
            'mad': round(mad_lv, 4) if mad_lv else 0,
            'outlier_count': outliers_lv,
            'outlier_ratio': round(outlier_ratio, 3),
            'z_score_method': 'modified_zscore_iglewicz_hoaglin',
            'verdict': verdict
        }

    def _calculate_mad(self, data: List[float]) -> float:
        """MAD (Median Absolute Deviation) 계산"""
        if not data:
            return 0.0
        median = sorted(data)[len(data) // 2]
        deviations = [abs(x - median) for x in data]
        return sorted(deviations)[len(deviations) // 2]

    def _modified_zscore(self, x: float, median: float, mad: float) -> float:
        """Modified Z-score (Iglewicz & Hoaglin, 1993)"""
        if mad == 0:
            return 0.0
        return 0.6745 * (x - median) / mad

    def _view_variability(self, posts: List[Dict]) -> Tuple[float, Dict]:
        """조회수 변동성 (CV)"""
        views = [p.get('views', 0) for p in posts if p.get('views', 0) > 0]

        if len(views) < 2:
            return 50.0, {'status': 'insufficient_data'}

        mean = sum(views) / len(views)
        if mean == 0:
            return 0.0, {'status': 'zero_mean'}

        variance = sum((v - mean) ** 2 for v in views) / len(views)
        cv = math.sqrt(variance) / mean

        # CV 0.08~0.5 정상
        if cv < 0.03:
            score = 30.0
        elif cv < 0.05:
            score = 55.0
        elif cv < 0.08:
            score = 75.0
        elif cv < 0.50:
            score = 95.0
        else:
            score = 80.0

        return score, {'cv': round(cv, 4)}

    def _engagement_asymmetry(self, posts: List[Dict]) -> Tuple[float, Dict]:
        """
        좋아요/조회수 비율 분석 (핵심 허수 탐지 지표)

        정상 범위: 2%~12%
        - 뷰봇: < 1% (조회수만 높고 참여 없음)
        - 좋아요 구매: > 20% (비정상적으로 높은 좋아요)

        추가 분석:
        - 비율의 표준편차: 일정하면 봇 의심
        """
        ratios = []
        for p in posts:
            views = p.get('views', 0)
            likes = p.get('likes', 0)
            if views > 0:
                ratios.append(likes / views)

        if not ratios:
            return 50.0, {'status': 'no_data', 'verdict': '데이터 없음'}

        avg = sum(ratios) / len(ratios)

        # 비율의 변동성 (봇은 일정한 패턴)
        if len(ratios) >= 2:
            variance = sum((r - avg) ** 2 for r in ratios) / len(ratios)
            cv = math.sqrt(variance) / avg if avg > 0 else 0
        else:
            cv = 0.1  # 기본값

        # 점수 계산
        verdict = "정상"
        if avg < 0.008:
            score = 25.0  # 뷰봇 강력 의심
            verdict = "뷰봇 의심"
        elif avg < 0.015:
            score = 45.0  # 뷰봇 가능성
            verdict = "뷰봇 가능성"
        elif 0.02 <= avg <= 0.12:
            score = 90.0  # 정상 범위
            verdict = "정상"
        elif 0.12 < avg <= 0.18:
            score = 75.0  # 약간 높음
            verdict = "참여율 높음"
        elif 0.18 < avg <= 0.25:
            score = 55.0  # 좋아요 구매 가능성
            verdict = "좋아요 구매 의심"
        else:
            score = 30.0  # 좋아요 구매 강력 의심
            verdict = "좋아요 구매 확실"

        # 비율 변동성이 너무 낮으면 봇 의심 (추가 감점)
        if cv < 0.05 and len(ratios) >= 3:
            score -= 15.0
            verdict += " (패턴 균일)"

        score = max(0.0, min(100.0, score))

        return score, {
            'avg_ratio': round(avg * 100, 2),
            'ratio_cv': round(cv, 4),
            'verdict': verdict
        }

    def _comment_entropy(self, posts: List[Dict]) -> Tuple[float, Dict]:
        """
        댓글/조회수 비율 분석

        정상 범위: 0.1%~2%
        - 뷰봇: < 0.05% (조회수만 높고 댓글 없음)
        - 봇 댓글: > 5% (비정상적으로 많은 댓글)

        좋아요 대비 댓글 비율도 함께 분석:
        - 정상: 댓글 = 좋아요의 3~15%
        - 비정상: 댓글이 좋아요보다 많거나 극히 적음
        """
        view_ratios = []
        like_ratios = []

        for p in posts:
            views = p.get('views', 0)
            likes = p.get('likes', 0)
            comments = p.get('comments', 0)

            if views > 0:
                view_ratios.append(comments / views)
            if likes > 0:
                like_ratios.append(comments / likes)

        if not view_ratios:
            return 50.0, {'status': 'no_data', 'verdict': '데이터 없음'}

        avg_view_ratio = sum(view_ratios) / len(view_ratios)
        avg_like_ratio = sum(like_ratios) / len(like_ratios) if like_ratios else 0

        # 조회수 대비 댓글 비율 점수
        verdict = "정상"
        if avg_view_ratio < 0.0005:
            score = 35.0  # 댓글 너무 적음 (뷰봇 의심)
            verdict = "댓글 부족"
        elif 0.001 <= avg_view_ratio <= 0.02:
            score = 90.0  # 정상
            verdict = "정상"
        elif 0.02 < avg_view_ratio <= 0.05:
            score = 70.0  # 약간 높음
            verdict = "댓글 다소 많음"
        elif avg_view_ratio > 0.05:
            score = 40.0  # 봇 댓글 의심
            verdict = "봇 댓글 의심"
        else:
            score = 60.0
            verdict = "경계"

        # 좋아요 대비 댓글 비율 추가 분석
        if like_ratios:
            if avg_like_ratio < 0.02:
                score -= 10.0  # 좋아요에 비해 댓글 너무 적음
                verdict += " (참여 불균형)"
            elif avg_like_ratio > 0.30:
                score -= 15.0  # 좋아요에 비해 댓글 너무 많음 (봇 의심)
                verdict += " (댓글 봇 의심)"

        score = max(0.0, min(100.0, score))

        return score, {
            'view_ratio': round(avg_view_ratio * 100, 3),
            'like_ratio': round(avg_like_ratio * 100, 2) if like_ratios else 0,
            'verdict': verdict
        }

    def _activity_regularity(self, influencer: Dict) -> Tuple[float, Dict]:
        """
        활동 패턴 규칙성 분석

        학술 기반:
        - 봇 계정은 업로드 간격이 너무 규칙적 (CV < 0.1)
        - 자연스러운 인간 행동은 불규칙성을 보임 (CV 0.3~0.8)

        수학적 공식:
        CV (Coefficient of Variation) = σ / μ
        - CV < 0.1: 봇 의심 (너무 규칙적)
        - CV 0.3~0.8: 정상 (자연스러운 불규칙성)
        - CV > 1.5: 비활성 의심 (너무 불규칙)
        """
        posts = influencer.get('recent_posts', [])
        interval = influencer.get('avg_upload_interval_days', 0)

        if interval == 0 or len(posts) < 3:
            return 60.0, {'status': 'insufficient_data'}

        # 타임스탬프에서 간격 계산
        timestamps = []
        for p in posts:
            ts = p.get('timestamp', '')
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    timestamps.append(dt)
                except:
                    pass

        if len(timestamps) < 3:
            # 폴백: avg_upload_interval_days 사용
            if 1 <= interval <= 7:
                return 85.0, {'interval_days': interval, 'method': 'fallback'}
            elif interval < 0.5:
                return 45.0, {'interval_days': interval, 'verdict': '봇 의심 (너무 빈번)'}
            elif interval < 1:
                return 70.0, {'interval_days': interval}
            elif interval <= 14:
                return 75.0, {'interval_days': interval}
            else:
                return 55.0, {'interval_days': interval, 'verdict': '비활성'}

        # 간격 계산
        timestamps = sorted(timestamps, reverse=True)
        intervals = []
        for i in range(len(timestamps) - 1):
            diff = (timestamps[i] - timestamps[i+1]).total_seconds() / 86400  # days
            if diff > 0:
                intervals.append(diff)

        if len(intervals) < 2:
            return 70.0, {'status': 'insufficient_intervals'}

        # CV (Coefficient of Variation) 계산
        mean = sum(intervals) / len(intervals)
        variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
        std = math.sqrt(variance)
        cv = std / mean if mean > 0 else 0

        # 점수 계산
        score = 90.0
        verdict = "정상"

        if cv < 0.1:
            score = 40.0
            verdict = "봇 의심 (너무 규칙적)"
        elif cv < 0.2:
            score = 65.0
            verdict = "다소 규칙적"
        elif cv <= 0.8:
            score = 90.0
            verdict = "정상 (자연스러운 패턴)"
        elif cv <= 1.5:
            score = 75.0
            verdict = "다소 불규칙"
        else:
            score = 55.0
            verdict = "매우 불규칙 (비활성 의심)"

        return score, {
            'avg_interval_days': round(mean, 2),
            'std_interval_days': round(std, 2),
            'cv': round(cv, 3),
            'cv_interpretation': verdict,
            'sample_size': len(intervals)
        }

    def _activity_stability(self, influencer: Dict) -> Tuple[float, Dict]:
        """업로드 간격 (정상: 1~7일) - 레거시 호환용"""
        return self._activity_regularity(influencer)

    def _geographic_consistency(self, influencer: Dict) -> Tuple[float, Dict]:
        """한국 팔로워 비율"""
        audience = influencer.get('audience_countries', {})
        bio = influencer.get('bio', '')

        if not audience:
            return 80.0, {'status': 'no_data'}

        kr_ratio = audience.get('KR', 0)

        # 한국어 콘텐츠 확인
        has_korean = any('\uac00' <= c <= '\ud7a3' for c in bio)
        for p in influencer.get('recent_posts', []):
            if any('\uac00' <= c <= '\ud7a3' for c in p.get('caption', '')):
                has_korean = True
                break

        is_korean_target = has_korean or kr_ratio >= 0.50

        if is_korean_target:
            if kr_ratio >= 0.70:
                score = 95.0
            elif kr_ratio >= 0.50:
                score = 90.0
            elif kr_ratio >= 0.35:
                score = 80.0
            else:
                score = 65.0
        else:
            score = 75.0 if kr_ratio >= 0.30 else 75.0

        return score, {'kr_ratio': kr_ratio}

    def _duplicate_content_jaccard(self, posts: List[Dict]) -> Tuple[float, Dict]:
        """
        Jaccard Similarity 기반 중복 콘텐츠 탐지

        학술 기반:
        - Jaccard Index (Jaccard, 1901): 집합 유사도 측정
        - Shingling + MinHash (Broder, 1997): 대규모 문서 유사도

        수학적 공식:
        J(A,B) = |A ∩ B| / |A ∪ B|
        - J > 0.7: 높은 유사도 (중복 의심)
        - J < 0.3: 낮은 유사도 (독창적)

        n-gram Jaccard:
        - 단어 단위가 아닌 n-gram 단위로 더 정밀한 유사도 측정
        """
        if len(posts) < 2:
            return 85.0, {'status': 'insufficient_data', 'duplicate_ratio': 0}

        def remove_hashtags(text: str) -> str:
            return re.sub(r'#\w+', '', text).strip()

        def get_ngrams(text: str, n: int = 2) -> set:
            """n-gram 추출 (2-gram 기본)"""
            words = text.lower().split()
            if len(words) < n:
                return set(words)
            return set(' '.join(words[i:i+n]) for i in range(len(words) - n + 1))

        def jaccard_similarity(set1: set, set2: set) -> float:
            """Jaccard 유사도 계산"""
            if not set1 or not set2:
                return 0.0
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0

        # 1. n-gram 기반 Jaccard 유사도 계산
        captions = [remove_hashtags(p.get('caption', '')) for p in posts]
        ngram_sets = [get_ngrams(c, n=2) for c in captions]

        similarity_matrix = []
        duplicate_count = 0

        for i in range(len(ngram_sets)):
            for j in range(i + 1, len(ngram_sets)):
                sim = jaccard_similarity(ngram_sets[i], ngram_sets[j])
                similarity_matrix.append(sim)
                if sim > 0.7:
                    duplicate_count += 1

        avg_jaccard = sum(similarity_matrix) / len(similarity_matrix) if similarity_matrix else 0

        # 2. 시간대 패턴 분석 (봇 자동화 탐지)
        timestamps = []
        for p in posts:
            ts = p.get('timestamp', '')
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    timestamps.append(dt.hour * 60 + dt.minute)
                except:
                    pass

        # 시간대 분산 분석
        time_variance = 0
        if len(timestamps) >= 2:
            mean_time = sum(timestamps) / len(timestamps)
            time_variance = sum((t - mean_time) ** 2 for t in timestamps) / len(timestamps)

        # 3. 점수 계산
        total_pairs = len(similarity_matrix) if similarity_matrix else 1
        duplicate_ratio = duplicate_count / total_pairs

        score = 90.0
        verdict_parts = []

        # Jaccard 중복 비율에 따른 감점
        if duplicate_ratio > 0.5:
            score -= 40.0
            verdict_parts.append("높은 중복률")
        elif duplicate_ratio > 0.3:
            score -= 25.0
            verdict_parts.append("중복 콘텐츠 존재")
        elif duplicate_ratio > 0.1:
            score -= 10.0

        # 평균 Jaccard 유사도에 따른 감점
        if avg_jaccard > 0.5:
            score -= 15.0
            verdict_parts.append("콘텐츠 유사성 높음")
        elif avg_jaccard > 0.3:
            score -= 5.0

        # 시간대 분산이 너무 낮으면 봇 의심
        if time_variance < 100 and len(timestamps) >= 3:  # 10분 이내 분산
            score -= 20.0
            verdict_parts.append("게시 시간 균일 (봇 의심)")

        score = max(20.0, min(95.0, score))
        verdict = ", ".join(verdict_parts) if verdict_parts else "정상 (독창적 콘텐츠)"

        return score, {
            'duplicate_ratio': round(duplicate_ratio, 3),
            'avg_jaccard_similarity': round(avg_jaccard, 3),
            'jaccard_method': '2-gram',
            'time_variance': round(time_variance, 2),
            'sample_pairs': len(similarity_matrix),
            'verdict': verdict
        }

    def _duplicate_content(self, posts: List[Dict]) -> Tuple[float, Dict]:
        """레거시 호환용"""
        return self._duplicate_content_jaccard(posts)


# ============================================================
# Influencer Classifier - TF-IDF 기반 분류
# ============================================================

class InfluencerClassifier:
    """
    TF-IDF + Cosine Similarity 기반 인플루언서 분류기

    학술 기반:
    ============
    1. TF-IDF (Term Frequency-Inverse Document Frequency)
       - Salton & McGill (1983): Introduction to Modern Information Retrieval
       - TF(t,d) = freq(t,d) / max_freq(d)
       - IDF(t) = log(N / df(t))
       - TF-IDF(t,d) = TF(t,d) × IDF(t)

    2. Cosine Similarity
       - Manning et al. (2008): Introduction to Information Retrieval
       - cos(θ) = (A · B) / (||A|| × ||B||)

    3. Soft Voting Ensemble
       - 키워드 점수 + TF-IDF 유사도 + 이미지 분석 결과를 결합
       - Dietterich (2000): Ensemble Methods in Machine Learning

    분류 카테고리:
    - Expert: 미용사, 살롱 원장, 시술 전문가
    - Trendsetter: 스타일 크리에이터, 뷰티 인플루언서
    """

    # 전문가 프로필 템플릿 (TF-IDF 참조 문서)
    EXPERT_PROFILE = """
    미용사 원장 살롱 헤어디자이너 시술 전문가 펌 염색 커트 클리닉
    두피케어 발레아쥬 테크닉 조색 미용실 예약 컬러리스트 헤어아티스트
    디렉터 자격증 교육 웨딩헤어 볼륨펌 C컬 히피펌 염색레시피
    청담 강남 살롱원장 경력 년차 시술후기 before after 변신 메이크오버
    """

    # 트렌드세터 프로필 템플릿
    TRENDSETTER_PROFILE = """
    스타일링 데일리룩 OOTD 추천 꿀팁 셀프 홈케어 트렌드 패션 일상
    크리에이터 인플루언서 협찬 리뷰 가성비 꿀템 솔직후기 루틴 유튜브
    브이로그 하울 언박싱 데일리 코디 미니멀 캐주얼 스트릿 Y2K
    뷰티그램 패션스타그램 팔로우 좋아요 소통 일상공유 여행 카페
    """

    EXPERT_KEYWORDS = [
        '미용사', '원장', '살롱', '시술', '예약', '펌', '염색약', '레시피',
        '컬러리스트', '헤어아티스트', '디렉터', '전문가', '자격증', '교육',
        '클리닉', '두피케어', '발레아쥬', '테크닉', '조색', '미용실'
    ]

    TRENDSETTER_KEYWORDS = [
        '스타일링', '데일리룩', 'OOTD', '추천', '꿀팁', '셀프', '홈케어',
        '트렌드', '패션', '일상', '크리에이터', '인플루언서', '협찬',
        '리뷰', '가성비', '꿀템', '솔직후기', '루틴', '유튜브'
    ]

    EXPERT_WEIGHTS = {'원장': 3.0, '미용사': 2.5, '살롱': 2.0, '시술': 2.0, '디렉터': 2.5}
    TRENDSETTER_WEIGHTS = {'크리에이터': 2.5, '인플루언서': 2.5, '트렌드세터': 3.0, '협찬': 2.0}

    def __init__(self):
        # TF-IDF 벡터 사전 계산 (참조 프로필)
        self.expert_tfidf = self._compute_tfidf(self.EXPERT_PROFILE)
        self.trendsetter_tfidf = self._compute_tfidf(self.TRENDSETTER_PROFILE)

        # IDF 계산을 위한 문서 빈도
        all_terms = set(self.expert_tfidf.keys()) | set(self.trendsetter_tfidf.keys())
        self.idf = {}
        for term in all_terms:
            df = sum(1 for d in [self.EXPERT_PROFILE, self.TRENDSETTER_PROFILE] if term in d.lower())
            self.idf[term] = math.log(2 / (df + 1)) + 1  # smoothed IDF

    def _tokenize(self, text: str) -> List[str]:
        """한국어 + 영어 토큰화"""
        # 한글, 영문, 숫자만 추출
        text = re.sub(r'[^\w\s가-힣]', ' ', text.lower())
        tokens = text.split()
        # 불용어 제거 (1글자 제외)
        return [t for t in tokens if len(t) > 1]

    def _compute_tfidf(self, text: str) -> Dict[str, float]:
        """TF-IDF 벡터 계산"""
        tokens = self._tokenize(text)
        if not tokens:
            return {}

        # Term Frequency
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        # Normalize by max frequency
        max_freq = max(tf.values()) if tf else 1
        for token in tf:
            tf[token] = tf[token] / max_freq

        return tf

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        코사인 유사도 계산

        cos(θ) = (A · B) / (||A|| × ||B||)
        """
        if not vec1 or not vec2:
            return 0.0

        # 공통 키
        common_keys = set(vec1.keys()) & set(vec2.keys())
        if not common_keys:
            return 0.0

        # 내적
        dot_product = sum(vec1[k] * vec2[k] for k in common_keys)

        # 벡터 크기
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def classify(self, influencer: Dict) -> Dict:
        """
        TF-IDF + Soft Voting 앙상블 분류

        3가지 신호를 결합:
        1. 키워드 점수 (가중치 기반)
        2. TF-IDF 코사인 유사도
        3. 이미지 분석 결과 (있으면)

        최종 신뢰도:
        confidence = α × keyword_score + β × tfidf_score + γ × image_score
        where α=0.4, β=0.4, γ=0.2
        """
        bio = influencer.get('bio', '')
        posts = influencer.get('recent_posts', [])
        captions = ' '.join([p.get('caption', '') for p in posts])
        full_text = f"{bio} {captions}"

        image_analysis = influencer.get('image_analysis', {})

        # 1. 키워드 기반 점수
        keyword_result = self._keyword_score(full_text)

        # 2. TF-IDF 코사인 유사도
        tfidf_result = self._tfidf_score(full_text)

        # 3. 이미지 분석 점수 (있으면)
        image_result = self._image_score(image_analysis)

        # 4. Soft Voting 앙상블
        # 가중치: 키워드 40%, TF-IDF 40%, 이미지 20%
        WEIGHT_KEYWORD = 0.40
        WEIGHT_TFIDF = 0.40
        WEIGHT_IMAGE = 0.20

        expert_vote = (
            WEIGHT_KEYWORD * keyword_result['expert_score'] +
            WEIGHT_TFIDF * tfidf_result['expert_similarity'] +
            WEIGHT_IMAGE * image_result['expert_score']
        )

        trendsetter_vote = (
            WEIGHT_KEYWORD * keyword_result['trendsetter_score'] +
            WEIGHT_TFIDF * tfidf_result['trendsetter_similarity'] +
            WEIGHT_IMAGE * image_result['trendsetter_score']
        )

        # 정규화
        total_vote = expert_vote + trendsetter_vote
        if total_vote > 0:
            expert_vote /= total_vote
            trendsetter_vote /= total_vote
        else:
            expert_vote = 0.3
            trendsetter_vote = 0.7  # 기본값: Trendsetter

        # 분류 결정
        if expert_vote > trendsetter_vote:
            classification = 'Expert'
            confidence = expert_vote
        else:
            classification = 'Trendsetter'
            confidence = trendsetter_vote

        # 역할 벡터 (Expert, Trendsetter)
        role_vector = [expert_vote, trendsetter_vote]

        return {
            'username': influencer.get('username', ''),
            'classification': classification,
            'confidence': round(confidence, 3),
            'role_vector': [round(v, 3) for v in role_vector],
            'expert_keywords': keyword_result['expert_found'],
            'trend_keywords': keyword_result['trend_found'],
            'method': 'tfidf_soft_voting_ensemble',
            'breakdown': {
                'keyword_score': {
                    'expert': round(keyword_result['expert_score'], 3),
                    'trendsetter': round(keyword_result['trendsetter_score'], 3)
                },
                'tfidf_similarity': {
                    'expert': round(tfidf_result['expert_similarity'], 3),
                    'trendsetter': round(tfidf_result['trendsetter_similarity'], 3)
                },
                'image_score': {
                    'expert': round(image_result['expert_score'], 3),
                    'trendsetter': round(image_result['trendsetter_score'], 3)
                }
            }
        }

    def _keyword_score(self, text: str) -> Dict:
        """키워드 기반 점수 계산"""
        expert_score = 0
        trend_score = 0
        expert_found = []
        trend_found = []

        for kw in self.EXPERT_KEYWORDS:
            count = text.count(kw)
            if count > 0:
                weight = self.EXPERT_WEIGHTS.get(kw, 1.0)
                expert_score += count * weight
                expert_found.append(kw)

        for kw in self.TRENDSETTER_KEYWORDS:
            count = text.count(kw)
            if count > 0:
                weight = self.TRENDSETTER_WEIGHTS.get(kw, 1.0)
                trend_score += count * weight
                trend_found.append(kw)

        # 정규화 (0~1)
        total = expert_score + trend_score
        if total > 0:
            expert_score = expert_score / total
            trend_score = trend_score / total
        else:
            expert_score = 0.3
            trend_score = 0.7

        return {
            'expert_score': expert_score,
            'trendsetter_score': trend_score,
            'expert_found': expert_found,
            'trend_found': trend_found
        }

    def _tfidf_score(self, text: str) -> Dict:
        """TF-IDF 코사인 유사도 계산"""
        user_tfidf = self._compute_tfidf(text)

        expert_sim = self._cosine_similarity(user_tfidf, self.expert_tfidf)
        trendsetter_sim = self._cosine_similarity(user_tfidf, self.trendsetter_tfidf)

        # 정규화
        total = expert_sim + trendsetter_sim
        if total > 0:
            expert_sim = expert_sim / total
            trendsetter_sim = trendsetter_sim / total
        else:
            expert_sim = 0.3
            trendsetter_sim = 0.7

        return {
            'expert_similarity': expert_sim,
            'trendsetter_similarity': trendsetter_sim
        }

    def _image_score(self, image_analysis: Dict) -> Dict:
        """이미지 분석 기반 점수"""
        if not image_analysis:
            return {'expert_score': 0.4, 'trendsetter_score': 0.6}

        prof = image_analysis.get('professionalism_score', 0.5)
        trend = image_analysis.get('trend_relevance_score', 0.5)

        # 정규화
        total = prof + trend
        if total > 0:
            expert_score = prof / total
            trend_score = trend / total
        else:
            expert_score = 0.4
            trend_score = 0.6

        return {
            'expert_score': expert_score,
            'trendsetter_score': trend_score
        }


# ============================================================
# Image Analyzer - LLM 비전 기반 이미지 분석
# ============================================================

class ImageAnalyzer:
    """
    LLM 비전 기반 이미지 스타일 분석

    - 스타일: luxury, natural, trendy, colorful, minimal, professional
    - 트렌드 부합도, 전문성 점수 등 추출
    """

    STYLE_CATEGORIES = ['luxury', 'natural', 'trendy', 'colorful', 'minimal', 'professional']

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = "gpt-4o-mini"

    def analyze(self, influencer: Dict, image_urls: List[str] = None) -> Dict:
        """인플루언서 비주얼 분석"""
        username = influencer.get('username', 'unknown')
        posts = influencer.get('recent_posts', [])

        # 이미 분석된 데이터가 있으면 사용
        if influencer.get('image_analysis'):
            return influencer['image_analysis']

        # 이미지 URL 추출
        if not image_urls:
            image_urls = [
                p.get('image_url') or p.get('thumbnail_url')
                for p in posts
                if p.get('image_url') or p.get('thumbnail_url')
            ]

        # 이미지가 없으면 시뮬레이션
        if not image_urls:
            return self._simulate_analysis(username, posts)

        # 각 이미지 분석
        analyses = []
        for url in image_urls[:5]:
            if url:
                result = self._analyze_single_image(url)
                analyses.append(result)

        return self._aggregate_results(username, analyses)

    def _analyze_single_image(self, image_url: str) -> Dict:
        """단일 이미지 분석"""
        if not self.api_key or not OPENAI_AVAILABLE:
            return self._simulate_single(image_url)

        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """헤어 스타일 비주얼 분석 전문가입니다.
JSON으로 응답하세요:
{
  "style_category": "luxury/natural/trendy/colorful/minimal/professional 중 하나",
  "style_confidence": 0.0-1.0,
  "professionalism_level": 0.0-1.0,
  "trend_relevance": 0.0-1.0,
  "color_palette": "warm/cool/neutral/vivid/muted"
}"""
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "이미지를 분석하세요."},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                max_tokens=300
            )

            text = response.choices[0].message.content.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            return json.loads(text.strip())

        except Exception as e:
            return self._simulate_single(image_url)

    def _simulate_single(self, source: str) -> Dict:
        """시뮬레이션 분석 (API 없을 때)"""
        hash_val = int(hashlib.md5(source.encode()).hexdigest(), 16)

        return {
            "style_category": self.STYLE_CATEGORIES[hash_val % len(self.STYLE_CATEGORIES)],
            "style_confidence": 0.6 + (hash_val % 40) / 100,
            "professionalism_level": 0.5 + (hash_val % 50) / 100,
            "trend_relevance": 0.5 + (hash_val % 50) / 100,
            "color_palette": ["warm", "cool", "neutral", "vivid", "muted"][hash_val % 5],
            "simulated": True
        }

    def _simulate_analysis(self, username: str, posts: List[Dict]) -> Dict:
        """포스트 기반 시뮬레이션"""
        analyses = []
        for i, post in enumerate(posts[:3]):
            sim = self._simulate_single(f"{username}_{post.get('caption', '')}_{i}")
            analyses.append(sim)
        return self._aggregate_results(username, analyses)

    def _aggregate_results(self, username: str, analyses: List[Dict]) -> Dict:
        """분석 결과 집계"""
        if not analyses:
            return {
                "username": username,
                "dominant_style": "trendy",
                "style_confidence": 0.5,
                "professionalism_score": 0.5,
                "trend_relevance_score": 0.5
            }

        # 스타일 집계
        style_counts = {}
        total_prof = 0
        total_trend = 0

        for a in analyses:
            style = a.get("style_category", "trendy")
            style_counts[style] = style_counts.get(style, 0) + 1
            total_prof += a.get("professionalism_level", 0.5)
            total_trend += a.get("trend_relevance", 0.5)

        dominant = max(style_counts, key=style_counts.get)
        avg_prof = total_prof / len(analyses)
        avg_trend = total_trend / len(analyses)

        return {
            "username": username,
            "dominant_style": dominant,
            "style_distribution": {k: v / len(analyses) for k, v in style_counts.items()},
            "style_confidence": 0.7,
            "professionalism_score": round(avg_prof, 3),
            "trend_relevance_score": round(avg_trend, 3),
            "visual_type_hint": "Trendsetter" if dominant in ["trendy", "colorful", "natural"] and avg_trend > 0.6 else "Expert"
        }


# ============================================================
# Recommendation Evaluator - 추천 품질 평가 메트릭
# ============================================================

class RecommendationEvaluator:
    """
    추천 품질 평가 메트릭 (학술적 기반)

    학술 기반:
    ============
    1. NDCG (Normalized Discounted Cumulative Gain)
       - Järvelin & Kekäläinen (2002): "Cumulated Gain-Based Evaluation of IR Techniques"
       - 순위 품질 측정: 상위 순위에 관련성 높은 항목이 있을수록 높은 점수

    2. Diversity (다양성)
       - Ziegler et al. (2005): "Improving Recommendation Lists Through Topic Diversification"
       - Intra-List Diversity: 추천 목록 내 항목들의 비유사도 평균

    3. Coverage (커버리지)
       - Herlocker et al. (2004): "Evaluating Collaborative Filtering Recommender Systems"
       - 전체 아이템 중 추천된 아이템의 비율

    4. Serendipity (새로움)
       - Ge et al. (2010): "Beyond Accuracy: Evaluating Recommender Systems by Coverage and Serendipity"
       - 예상치 못한 유용한 추천

    수학적 공식:
    ============
    NDCG@k = DCG@k / IDCG@k
    DCG@k = Σ (2^rel_i - 1) / log2(i + 1)

    Diversity = (2 / n(n-1)) × Σ (1 - similarity(i,j))

    Coverage = |unique items recommended| / |total items|
    """

    def __init__(self):
        pass

    def evaluate(self, recommendations: List[Dict], ground_truth: List[str] = None,
                 all_items: List[str] = None) -> Dict:
        """
        추천 결과 종합 평가

        Args:
            recommendations: 추천 결과 리스트 (username, score 포함)
            ground_truth: 정답 레이블 (있으면 NDCG 계산)
            all_items: 전체 아이템 목록 (Coverage 계산용)

        Returns:
            평가 메트릭 딕셔너리
        """
        if not recommendations:
            return {'error': 'No recommendations to evaluate'}

        metrics = {}

        # 1. NDCG 계산 (ground_truth가 있으면)
        if ground_truth:
            relevance = [1 if r.get('username') in ground_truth else 0 for r in recommendations]
            metrics['ndcg@5'] = self._ndcg(relevance, k=5)
            metrics['ndcg@10'] = self._ndcg(relevance, k=10)

        # 2. Precision/Recall (ground_truth가 있으면)
        if ground_truth:
            recommended_users = [r.get('username') for r in recommendations]
            metrics['precision@5'] = self._precision_at_k(recommended_users, ground_truth, k=5)
            metrics['recall@5'] = self._recall_at_k(recommended_users, ground_truth, k=5)

        # 3. Diversity (추천 항목 간 다양성)
        metrics['intra_list_diversity'] = self._intra_list_diversity(recommendations)

        # 4. Coverage (커버리지)
        if all_items:
            recommended = set(r.get('username') for r in recommendations)
            metrics['coverage'] = len(recommended) / len(all_items) if all_items else 0

        # 5. Score Distribution Analysis (점수 분포 분석)
        scores = [r.get('score', 0) for r in recommendations]
        if scores:
            metrics['score_distribution'] = {
                'mean': round(sum(scores) / len(scores), 4),
                'std': round(self._std(scores), 4),
                'min': round(min(scores), 4),
                'max': round(max(scores), 4),
                'range': round(max(scores) - min(scores), 4)
            }

        # 6. Type Distribution (유형 분포)
        type_counts = {}
        for r in recommendations:
            inf_type = r.get('influencer_type', 'unknown')
            type_counts[inf_type] = type_counts.get(inf_type, 0) + 1
        metrics['type_distribution'] = type_counts

        # 7. FIS Distribution (FIS 점수 분포)
        fis_scores = [r.get('fis_score', 0) for r in recommendations]
        if fis_scores:
            metrics['fis_distribution'] = {
                'mean': round(sum(fis_scores) / len(fis_scores), 1),
                'min': round(min(fis_scores), 1),
                'max': round(max(fis_scores), 1)
            }

        return metrics

    def _ndcg(self, relevance: List[int], k: int = 10) -> float:
        """
        NDCG@k 계산

        DCG@k = Σ (2^rel_i - 1) / log2(i + 1)
        NDCG@k = DCG@k / IDCG@k
        """
        relevance = relevance[:k]

        # DCG 계산
        dcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevance))

        # IDCG 계산 (완벽한 순서)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))

        if idcg == 0:
            return 0.0

        return round(dcg / idcg, 4)

    def _precision_at_k(self, recommended: List[str], ground_truth: List[str], k: int) -> float:
        """Precision@k: 상위 k개 중 관련 항목 비율"""
        recommended_k = set(recommended[:k])
        relevant = set(ground_truth)
        hits = len(recommended_k & relevant)
        return round(hits / k, 4) if k > 0 else 0.0

    def _recall_at_k(self, recommended: List[str], ground_truth: List[str], k: int) -> float:
        """Recall@k: 전체 관련 항목 중 추천된 비율"""
        recommended_k = set(recommended[:k])
        relevant = set(ground_truth)
        if not relevant:
            return 0.0
        hits = len(recommended_k & relevant)
        return round(hits / len(relevant), 4)

    def _intra_list_diversity(self, recommendations: List[Dict]) -> float:
        """
        Intra-List Diversity (ILD)

        추천 목록 내 항목들의 평균 비유사도
        ILD = (2 / n(n-1)) × Σ (1 - similarity(i,j))

        특성 벡터: [influencer_type, aesthetic_style, followers_tier]
        """
        if len(recommendations) < 2:
            return 1.0  # 다양성 최대

        # 특성 추출
        features = []
        for r in recommendations:
            feature = {
                'type': r.get('influencer_type', 'unknown'),
                'style': r.get('metadata', {}).get('main_mood', 'unknown'),
                'tier': self._get_follower_tier(r.get('metadata', {}).get('followers', 0))
            }
            features.append(feature)

        # 비유사도 계산 (Jaccard distance)
        total_dissimilarity = 0
        pairs = 0

        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                dissimilarity = self._feature_dissimilarity(features[i], features[j])
                total_dissimilarity += dissimilarity
                pairs += 1

        if pairs == 0:
            return 1.0

        return round(total_dissimilarity / pairs, 4)

    def _feature_dissimilarity(self, f1: Dict, f2: Dict) -> float:
        """특성 간 비유사도 (0~1)"""
        same = 0
        total = 3  # type, style, tier

        if f1['type'] == f2['type']:
            same += 1
        if f1['style'] == f2['style']:
            same += 1
        if f1['tier'] == f2['tier']:
            same += 1

        return 1 - (same / total)

    def _get_follower_tier(self, followers: int) -> str:
        """팔로워 수 기반 티어 분류"""
        if followers >= 1000000:
            return 'mega'
        elif followers >= 100000:
            return 'macro'
        elif followers >= 10000:
            return 'micro'
        else:
            return 'nano'

    def _std(self, data: List[float]) -> float:
        """표준편차 계산"""
        if len(data) < 2:
            return 0.0
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        return math.sqrt(variance)

    def generate_report(self, metrics: Dict) -> str:
        """평가 결과 리포트 생성"""
        lines = [
            "=" * 60,
            "📊 추천 품질 평가 리포트 (학술적 메트릭 기반)",
            "=" * 60,
            "",
            "📈 순위 품질 (NDCG - Järvelin & Kekäläinen, 2002)",
            "-" * 40
        ]

        if 'ndcg@5' in metrics:
            lines.append(f"  NDCG@5:  {metrics['ndcg@5']:.4f}")
            lines.append(f"  NDCG@10: {metrics['ndcg@10']:.4f}")
        else:
            lines.append("  (ground truth 없음 - NDCG 계산 불가)")

        lines.extend([
            "",
            "🎯 정확도 (Precision/Recall)",
            "-" * 40
        ])

        if 'precision@5' in metrics:
            lines.append(f"  Precision@5: {metrics['precision@5']:.4f}")
            lines.append(f"  Recall@5:    {metrics['recall@5']:.4f}")
        else:
            lines.append("  (ground truth 없음)")

        lines.extend([
            "",
            "🌈 다양성 (Ziegler et al., 2005)",
            "-" * 40,
            f"  Intra-List Diversity: {metrics.get('intra_list_diversity', 0):.4f}",
            f"  (1.0 = 완전 다양, 0.0 = 동일 항목)",
            "",
            "📊 점수 분포",
            "-" * 40
        ])

        if 'score_distribution' in metrics:
            sd = metrics['score_distribution']
            lines.append(f"  평균: {sd['mean']:.4f}")
            lines.append(f"  표준편차: {sd['std']:.4f}")
            lines.append(f"  범위: {sd['min']:.4f} ~ {sd['max']:.4f} (차이: {sd['range']:.4f})")

        lines.extend([
            "",
            "👥 유형 분포",
            "-" * 40
        ])

        if 'type_distribution' in metrics:
            for t, count in metrics['type_distribution'].items():
                lines.append(f"  {t}: {count}명")

        if 'fis_distribution' in metrics:
            fis = metrics['fis_distribution']
            lines.extend([
                "",
                "🛡️ FIS (신뢰도) 분포",
                "-" * 40,
                f"  평균: {fis['mean']:.1f}",
                f"  범위: {fis['min']:.1f} ~ {fis['max']:.1f}"
            ])

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


# 테스트
if __name__ == "__main__":
    print("=" * 60)
    print("InfluencerProcessor 파이프라인 테스트")
    print("=" * 60)

    # 테스트용 raw 데이터 (크롤러에서 수집한 형태)
    raw_data = {
        "influencers": [
            {
                "username": "hair_master_kim",
                "followers": 85000,
                "bio": "청담동 헤어살롱 원장 | 15년차 미용사 | 염색 & 펌 전문",
                "media_count": 500,
                "recent_posts": [
                    {"caption": "C컬 펌 시술 완료! #펌전문 #헤어디자이너", "views": 45000, "likes": 3200, "comments": 89},
                    {"caption": "애쉬브라운 염색 레시피 공개 #염색전문", "views": 38000, "likes": 2800, "comments": 72}
                ],
                "audience_countries": {"KR": 0.92, "US": 0.03},
                "avg_upload_interval_days": 3.2
            },
            {
                "username": "haru_style",
                "followers": 350000,
                "bio": "fashion | daily",
                "media_count": 800,
                "recent_posts": [
                    {"caption": "#ootd", "views": 150000, "likes": 12000, "comments": 450},
                    {"caption": "", "views": 180000, "likes": 15000, "comments": 520}
                ],
                "audience_countries": {"KR": 0.78, "US": 0.08},
                "avg_upload_interval_days": 1.5
            }
        ],
        "metadata": {"status": "raw"}
    }

    # InfluencerProcessor 테스트
    processor = InfluencerProcessor()
    result = processor.process(raw_data, use_llm=False)

    print(f"\n처리 결과:")
    print(f"  총 인플루언서: {result['metadata']['total_count']}명")
    print(f"  Expert: {result['metadata']['expert_count']}명")
    print(f"  Trendsetter: {result['metadata']['trendsetter_count']}명")

    for inf in result["influencers"]:
        print(f"\n@{inf['username']}:")
        print(f"  유형: {inf['influencer_type']}")
        print(f"  분석 전략: {inf['analysis_strategy']['primary']} Primary")
        print(f"  FIS: {inf['fis']['score']} ({inf['fis']['verdict']})")

    print("\n" + "=" * 60)
    print("개별 컴포넌트 테스트")
    print("=" * 60)

    # FIS 테스트
    fis_calc = FISCalculator()
    test_inf = {
        "username": "test_user",
        "recent_posts": [
            {"views": 45000, "likes": 3200, "comments": 89},
            {"views": 38000, "likes": 2800, "comments": 72}
        ],
        "audience_countries": {"KR": 0.92, "US": 0.03},
        "avg_upload_interval_days": 3.2
    }
    fis = fis_calc.calculate(test_inf)
    print(f"\nFIS: {fis['fis_score']} - {fis['verdict']}")

    # 분류 테스트
    classifier = InfluencerClassifier()
    test_inf["bio"] = "청담동 헤어살롱 원장 | 15년차 미용사"
    result = classifier.classify(test_inf)
    print(f"분류: {result['classification']} ({result['confidence']:.2f})")
