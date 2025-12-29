"""
프로세서 모듈 - 인플루언서 데이터 처리
======================================

1. InfluencerProcessor: 메인 처리 파이프라인
   - Expert/Trendsetter 분류
   - 유형별 분석 전략 적용 (텍스트/이미지)
   - 최종 JSON 구성

2. FISCalculator: Fake Integrity Score 계산 (허수 필터링)
3. InfluencerClassifier: Expert/Trendsetter 분류 로직
4. ImageAnalyzer: 이미지 스타일 분석 (LLM 비전)
"""

import os
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
# FIS Calculator - 허수 계정 탐지
# ============================================================

class FISCalculator:
    """
    Fake Integrity Score 계산기

    허수 계정 탐지 지표:
    - V: 조회수 변동성 (CV < 0.1 → 뷰봇 의심, 조작된 조회수는 균일함)
    - A: 참여 비대칭성 (좋아요/조회수 비율)
        - 정상: 2%~12%
        - 뷰봇: < 1% (조회수만 높고 참여 없음)
        - 좋아요 구매: > 20% (비정상적으로 높은 좋아요)
    - E: 댓글 엔트로피 (댓글/조회수 비율)
        - 정상: 0.1%~2%
        - 봇 댓글: > 5% (비정상적으로 많은 댓글)
    - ACS: 활동 안정성 (업로드 간격)
    - D: 지리적 정합성 (한국 타겟 확인)
    - DUP: 중복 콘텐츠 비율 (콘텐츠 재활용/봇 패턴 탐지)

    FIS = (w1×V + w2×E + w3×A + w4×ACS + w5×DUP) × D/100

    2025년 12월 업데이트:
    - Instagram Graph API에서 Reels views 데이터 수집 가능 (Business Discovery API)
    - 조회수 대비 좋아요/댓글 비율로 허수 계정 탐지 정확도 향상
    """

    def __init__(self):
        # 가중치 (조회수 기반 참여율 분석 강화)
        self.w_view = 0.20          # 조회수 변동성
        self.w_engagement = 0.25    # 좋아요/조회수 비율 (핵심 지표)
        self.w_comment = 0.15       # 댓글/조회수 비율
        self.w_activity = 0.10      # 업로드 간격
        self.w_geo = 0.15           # 지리적 정합성
        self.w_duplicate = 0.15     # 중복 콘텐츠

    def calculate(self, influencer: Dict) -> Dict:
        """FIS 점수 계산"""
        posts = influencer.get('recent_posts', [])

        v_score, v_detail = self._view_variability(posts)
        a_score, a_detail = self._engagement_asymmetry(posts)
        e_score, e_detail = self._comment_entropy(posts)
        acs_score, acs_detail = self._activity_stability(influencer)
        d_score, d_detail = self._geographic_consistency(influencer)
        dup_score, dup_detail = self._duplicate_content(posts)

        # 기본 점수 (중복 콘텐츠 포함)
        base_score = (
            self.w_view * v_score +
            self.w_comment * e_score +
            self.w_engagement * a_score +
            self.w_activity * acs_score +
            self.w_duplicate * dup_score
        )

        # 지리적 정합성 반영
        final_score = base_score * (d_score / 100) + (self.w_geo * d_score)
        final_score = max(0, min(100, final_score))

        # 판정
        if final_score >= 80:
            verdict = '신뢰 계정'
        elif final_score >= 60:
            verdict = '주의 필요'
        else:
            verdict = '허수 의심'

        return {
            'username': influencer.get('username', ''),
            'fis_score': round(final_score, 1),
            'verdict': verdict,
            'breakdown': {
                'view_variability': v_score,
                'engagement_asymmetry': a_score,
                'comment_entropy': e_score,
                'activity_stability': acs_score,
                'geographic_consistency': d_score,
                'duplicate_content': dup_score
            },
            'duplicate_detail': dup_detail
        }

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

    def _activity_stability(self, influencer: Dict) -> Tuple[float, Dict]:
        """업로드 간격 (정상: 1~7일)"""
        interval = influencer.get('avg_upload_interval_days', 0)

        if interval == 0:
            return 50.0, {'status': 'no_data'}

        if 1 <= interval <= 7:
            score = 90.0
        elif 0.5 <= interval < 1:
            score = 75.0
        elif 7 < interval <= 14:
            score = 80.0
        elif interval < 0.5:
            score = 40.0  # 봇 의심
        else:
            score = 60.0

        return score, {'interval_days': interval}

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

    def _duplicate_content(self, posts: List[Dict]) -> Tuple[float, Dict]:
        """
        중복 콘텐츠 탐지

        탐지 방법:
        1. caption 유사도 분석 (해시태그 제외 후 비교)
        2. 동일 시간대 게시 패턴 (봇 자동화 의심)
        3. 연속 게시물 간 텍스트 유사도

        Returns:
            (점수, 상세정보) - 점수가 높을수록 좋음 (중복 적음)
        """
        if len(posts) < 2:
            return 85.0, {'status': 'insufficient_data', 'duplicate_ratio': 0}

        # 해시태그 제거 함수
        def remove_hashtags(text: str) -> str:
            import re
            return re.sub(r'#\w+', '', text).strip()

        # 텍스트 유사도 계산 (Jaccard similarity)
        def text_similarity(text1: str, text2: str) -> float:
            if not text1 or not text2:
                return 0.0
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = words1 & words2
            union = words1 | words2
            return len(intersection) / len(union) if union else 0.0

        # 1. caption 유사도 분석
        captions = [remove_hashtags(p.get('caption', '')) for p in posts]
        similarity_pairs = []
        duplicate_count = 0

        for i in range(len(captions)):
            for j in range(i + 1, len(captions)):
                sim = text_similarity(captions[i], captions[j])
                similarity_pairs.append(sim)
                if sim > 0.7:  # 70% 이상 유사하면 중복으로 간주
                    duplicate_count += 1

        avg_similarity = sum(similarity_pairs) / len(similarity_pairs) if similarity_pairs else 0

        # 2. 시간대 패턴 분석 (동일 시간 게시 - 봇 의심)
        timestamps = []
        for p in posts:
            ts = p.get('timestamp', '')
            if ts:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    timestamps.append(dt.hour * 60 + dt.minute)  # 분 단위로 변환
                except:
                    pass

        same_time_count = 0
        if len(timestamps) >= 2:
            for i in range(len(timestamps)):
                for j in range(i + 1, len(timestamps)):
                    # 5분 이내 동일 시간대면 자동화 의심
                    if abs(timestamps[i] - timestamps[j]) <= 5:
                        same_time_count += 1

        # 3. 점수 계산
        total_pairs = len(similarity_pairs) if similarity_pairs else 1
        duplicate_ratio = duplicate_count / total_pairs
        same_time_ratio = same_time_count / total_pairs if total_pairs > 0 else 0

        # 기본 점수 (중복 없으면 90점)
        score = 90.0

        # 중복 콘텐츠 감점 (최대 -40점)
        if duplicate_ratio > 0.5:
            score -= 40.0
        elif duplicate_ratio > 0.3:
            score -= 25.0
        elif duplicate_ratio > 0.1:
            score -= 10.0

        # 자동화 의심 감점 (최대 -20점)
        if same_time_ratio > 0.3:
            score -= 20.0
        elif same_time_ratio > 0.1:
            score -= 10.0

        # 평균 유사도가 높으면 추가 감점
        if avg_similarity > 0.5:
            score -= 15.0
        elif avg_similarity > 0.3:
            score -= 5.0

        score = max(20.0, min(95.0, score))

        return score, {
            'duplicate_ratio': round(duplicate_ratio, 3),
            'avg_similarity': round(avg_similarity, 3),
            'same_time_pattern': same_time_count,
            'verdict': '정상' if score >= 70 else '중복 의심' if score >= 50 else '봇 의심'
        }


# ============================================================
# Influencer Classifier - Expert/Trendsetter 분류
# ============================================================

class InfluencerClassifier:
    """
    인플루언서 분류기

    Expert: 미용사, 살롱 원장, 시술 전문가
    Trendsetter: 스타일 크리에이터, 뷰티 인플루언서
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

    def classify(self, influencer: Dict) -> Dict:
        """인플루언서 분류"""
        bio = influencer.get('bio', '')
        posts = influencer.get('recent_posts', [])
        captions = ' '.join([p.get('caption', '') for p in posts])
        full_text = f"{bio} {captions}"

        image_analysis = influencer.get('image_analysis', {})

        # 키워드 점수 계산
        expert_score = 0
        trend_score = 0
        expert_found = []
        trend_found = []

        for kw in self.EXPERT_KEYWORDS:
            count = full_text.count(kw)
            if count > 0:
                weight = self.EXPERT_WEIGHTS.get(kw, 1.0)
                expert_score += count * weight
                expert_found.append(kw)

        for kw in self.TRENDSETTER_KEYWORDS:
            count = full_text.count(kw)
            if count > 0:
                weight = self.TRENDSETTER_WEIGHTS.get(kw, 1.0)
                trend_score += count * weight
                trend_found.append(kw)

        total = expert_score + trend_score

        # 분류 결정
        if total == 0:
            # 이미지 분석 결과 활용
            if image_analysis:
                trend_rel = image_analysis.get('trend_relevance_score', 0.5)
                prof = image_analysis.get('professionalism_score', 0.5)

                if trend_rel > prof and trend_rel > 0.5:
                    classification = 'Trendsetter'
                    confidence = min(0.8, trend_rel)
                elif prof > trend_rel and prof > 0.5:
                    classification = 'Expert'
                    confidence = min(0.8, prof)
                else:
                    classification = 'Trendsetter'
                    confidence = 0.5
            else:
                classification = 'Trendsetter'
                confidence = 0.4
        else:
            expert_ratio = expert_score / total
            trend_ratio = trend_score / total

            if expert_ratio > trend_ratio:
                classification = 'Expert'
                confidence = expert_ratio
            else:
                classification = 'Trendsetter'
                confidence = trend_ratio

        # 역할 벡터
        if classification == 'Expert':
            role_vector = [confidence, 1 - confidence]
        else:
            role_vector = [1 - confidence, confidence]

        return {
            'username': influencer.get('username', ''),
            'classification': classification,
            'confidence': round(confidence, 3),
            'role_vector': role_vector,
            'expert_keywords': expert_found,
            'trend_keywords': trend_found
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
