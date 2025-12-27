"""
Image Analyzer Module - 인플루언서 이미지/릴스 분석 모듈
LLM 기반 비주얼 스타일 분석 (특히 트렌드세터 분석에 중점)
"""

from typing import Dict, List, Optional, Tuple
import base64
import os
import json
from pathlib import Path

# OpenAI Vision API 사용을 위한 설정
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# 스타일 카테고리 정의
VISUAL_STYLE_CATEGORIES = {
    "luxury": {
        "keywords": ["고급스러운", "프리미엄", "세련된", "럭셔리", "하이엔드"],
        "visual_cues": ["깔끔한 배경", "고급 조명", "미니멀한 구도", "고화질", "프로페셔널 촬영"],
        "score_weight": 1.0
    },
    "natural": {
        "keywords": ["자연스러운", "내추럴", "건강한", "순수한", "유기농"],
        "visual_cues": ["자연광", "그린톤", "야외촬영", "식물배경", "가공없는느낌"],
        "score_weight": 0.8
    },
    "trendy": {
        "keywords": ["트렌디", "힙한", "MZ", "감각적", "스타일리시"],
        "visual_cues": ["독특한앵글", "네온컬러", "필터사용", "스트릿감성", "레트로"],
        "score_weight": 0.9
    },
    "colorful": {
        "keywords": ["화려한", "컬러풀", "비비드", "팝한", "에너지틱"],
        "visual_cues": ["강렬한색감", "다양한컬러", "화사한조명", "포화도높음", "밝은분위기"],
        "score_weight": 0.7
    },
    "minimal": {
        "keywords": ["미니멀", "심플", "깔끔한", "모던", "절제된"],
        "visual_cues": ["단색배경", "여백활용", "정돈된구도", "무채색", "클린"],
        "score_weight": 0.8
    },
    "professional": {
        "keywords": ["전문적", "살롱", "시술", "기술적", "프로페셔널"],
        "visual_cues": ["시술과정", "비포애프터", "작업대", "도구노출", "살롱인테리어"],
        "score_weight": 1.0
    }
}

# 트렌드세터 스타일 특화 분석 요소
TRENDSETTER_VISUAL_ELEMENTS = {
    "fashion_integration": "패션과 헤어스타일의 조화",
    "lifestyle_vibes": "라이프스타일 무드 연출",
    "aesthetic_consistency": "피드 전체의 미적 일관성",
    "trend_awareness": "최신 트렌드 반영도",
    "personal_branding": "개인 브랜딩 요소",
    "engagement_potential": "시각적 참여 유도력"
}


class ImageAnalyzer:
    """LLM 기반 이미지 분석기"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: OpenAI API 키 (환경변수 OPENAI_API_KEY도 사용 가능)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = "gpt-4o-mini"  # Vision 지원 모델

    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """이미지를 base64로 인코딩"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.standard_b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            print(f"이미지 인코딩 실패: {e}")
            return None

    def analyze_single_image(self, image_source: str, is_url: bool = True) -> Dict:
        """
        단일 이미지 분석 (URL 또는 로컬 파일)

        Args:
            image_source: 이미지 URL 또는 로컬 파일 경로
            is_url: True면 URL, False면 로컬 파일

        Returns:
            분석 결과 딕셔너리
        """
        if not self.api_key:
            return self._simulate_image_analysis(image_source)

        try:
            if is_url:
                image_content = {"type": "image_url", "image_url": {"url": image_source}}
            else:
                base64_image = self.encode_image_to_base64(image_source)
                if not base64_image:
                    return {"error": "이미지 인코딩 실패"}
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }

            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """당신은 헤어 스타일 및 인플루언서 비주얼 분석 전문가입니다.
이미지를 분석하여 다음 정보를 JSON 형태로 추출해주세요:
1. style_category: luxury/natural/trendy/colorful/minimal/professional 중 하나
2. style_confidence: 0.0-1.0 사이의 확신도
3. hair_style_features: 헤어스타일 특징 리스트
4. visual_mood: 전반적인 비주얼 무드 설명
5. color_palette: 주요 색감 (warm/cool/neutral/vivid/muted)
6. professionalism_level: 전문성 수준 (0.0-1.0)
7. trend_relevance: 트렌드 부합도 (0.0-1.0)
8. target_appeal: 어필할 것 같은 타겟층 설명"""
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "이 헤어 관련 이미지를 분석해주세요."},
                            image_content
                        ]
                    }
                ],
                max_tokens=500
            )

            result_text = response.choices[0].message.content
            # JSON 파싱 시도
            try:
                # JSON 블록 추출
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0]
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0]
                return json.loads(result_text.strip())
            except:
                return {
                    "raw_analysis": result_text,
                    "style_category": "trendy",
                    "style_confidence": 0.7
                }

        except Exception as e:
            print(f"이미지 분석 API 오류: {e}")
            return self._simulate_image_analysis(image_source)

    def _simulate_image_analysis(self, image_source: str) -> Dict:
        """API 없이 시뮬레이션 분석 (MVP용)"""
        import hashlib

        # 이미지 소스를 해시하여 일관된 결과 생성
        hash_val = int(hashlib.md5(image_source.encode()).hexdigest(), 16)

        styles = list(VISUAL_STYLE_CATEGORIES.keys())
        selected_style = styles[hash_val % len(styles)]

        return {
            "style_category": selected_style,
            "style_confidence": 0.6 + (hash_val % 40) / 100,
            "hair_style_features": ["웨이브", "볼륨감", "자연스러운 컬러"],
            "visual_mood": VISUAL_STYLE_CATEGORIES[selected_style]["keywords"][0],
            "color_palette": ["warm", "cool", "neutral", "vivid", "muted"][hash_val % 5],
            "professionalism_level": 0.5 + (hash_val % 50) / 100,
            "trend_relevance": 0.5 + (hash_val % 50) / 100,
            "target_appeal": "20-30대 트렌드에 민감한 여성",
            "simulated": True
        }

    def analyze_influencer_visuals(
        self,
        influencer_data: Dict,
        image_urls: Optional[List[str]] = None
    ) -> Dict:
        """
        인플루언서의 비주얼 스타일 종합 분석

        Args:
            influencer_data: 인플루언서 기본 데이터
            image_urls: 분석할 이미지 URL 리스트 (릴스 썸네일 등)

        Returns:
            비주얼 분석 결과
        """
        username = influencer_data.get("username", "unknown")
        posts = influencer_data.get("recent_posts", [])

        # 이미지 URL이 없으면 posts에서 추출 시도
        if not image_urls:
            image_urls = [
                post.get("image_url") or post.get("thumbnail_url")
                for post in posts
                if post.get("image_url") or post.get("thumbnail_url")
            ]

        # 각 이미지 분석
        image_analyses = []
        for url in image_urls[:5]:  # 최대 5개 분석
            if url:
                analysis = self.analyze_single_image(url, is_url=True)
                image_analyses.append(analysis)

        # 이미지가 없으면 시뮬레이션 데이터 생성
        if not image_analyses:
            for i, post in enumerate(posts[:3]):
                simulated = self._simulate_image_analysis(
                    f"{username}_{post.get('caption', '')}_{i}"
                )
                image_analyses.append(simulated)

        # 결과 집계
        return self._aggregate_visual_analysis(username, image_analyses)

    def _aggregate_visual_analysis(
        self,
        username: str,
        analyses: List[Dict]
    ) -> Dict:
        """여러 이미지 분석 결과를 집계"""
        if not analyses:
            return {
                "username": username,
                "visual_style": "unknown",
                "confidence": 0.0,
                "error": "분석할 이미지 없음"
            }

        # 스타일 카테고리 집계
        style_counts = {}
        style_confidences = {}
        total_professionalism = 0
        total_trend = 0
        color_palettes = []

        for analysis in analyses:
            style = analysis.get("style_category", "trendy")
            conf = analysis.get("style_confidence", 0.5)

            style_counts[style] = style_counts.get(style, 0) + 1
            style_confidences[style] = style_confidences.get(style, [])
            style_confidences[style].append(conf)

            total_professionalism += analysis.get("professionalism_level", 0.5)
            total_trend += analysis.get("trend_relevance", 0.5)

            if analysis.get("color_palette"):
                color_palettes.append(analysis["color_palette"])

        # 가장 많이 나온 스타일 선택
        dominant_style = max(style_counts, key=style_counts.get)
        avg_confidence = sum(style_confidences[dominant_style]) / len(style_confidences[dominant_style])

        # 전문성 기반 Expert/Trendsetter 보조 판단
        avg_professionalism = total_professionalism / len(analyses)
        avg_trend = total_trend / len(analyses)

        # 트렌드세터 특화 분석
        is_trendsetter_visual = (
            dominant_style in ["trendy", "colorful", "natural"] and
            avg_trend > 0.6
        )

        return {
            "username": username,
            "dominant_visual_style": dominant_style,
            "style_confidence": round(avg_confidence, 3),
            "style_distribution": {k: v / len(analyses) for k, v in style_counts.items()},
            "professionalism_score": round(avg_professionalism, 3),
            "trend_relevance_score": round(avg_trend, 3),
            "color_palette_tendency": max(set(color_palettes), key=color_palettes.count) if color_palettes else "neutral",
            "visual_type_hint": "Trendsetter" if is_trendsetter_visual else "Expert",
            "analysis_count": len(analyses),
            "detailed_analyses": analyses
        }


def analyze_influencer_style(
    influencer_data: Dict,
    image_urls: Optional[List[str]] = None,
    api_key: Optional[str] = None
) -> Dict:
    """
    인플루언서 비주얼 스타일 분석 헬퍼 함수

    Args:
        influencer_data: 인플루언서 데이터
        image_urls: 이미지 URL 리스트 (선택)
        api_key: OpenAI API 키 (선택)

    Returns:
        비주얼 분석 결과
    """
    analyzer = ImageAnalyzer(api_key=api_key)
    return analyzer.analyze_influencer_visuals(influencer_data, image_urls)


def get_visual_style_vector(analysis_result: Dict) -> List[float]:
    """
    비주얼 분석 결과를 벡터로 변환

    Args:
        analysis_result: analyze_influencer_style 결과

    Returns:
        비주얼 스타일 벡터 [luxury, natural, trendy, colorful, minimal, professional]
    """
    style_dist = analysis_result.get("style_distribution", {})

    vector = [
        style_dist.get("luxury", 0.0),
        style_dist.get("natural", 0.0),
        style_dist.get("trendy", 0.0),
        style_dist.get("colorful", 0.0),
        style_dist.get("minimal", 0.0),
        style_dist.get("professional", 0.0)
    ]

    # 정규화
    total = sum(vector)
    if total > 0:
        vector = [v / total for v in vector]

    return vector


# 테스트용 코드
if __name__ == "__main__":
    # 테스트 인플루언서 데이터
    test_influencer = {
        "username": "style_queen_yuna",
        "bio": "Daily Hair & Fashion | 스타일 크리에이터",
        "recent_posts": [
            {"caption": "오늘의 데일리룩 + 헤어스타일링!", "image_url": None},
            {"caption": "집에서 셀프로 하는 물결펌!", "image_url": None},
            {"caption": "이번 주 추천 헤어케어 제품!", "image_url": None}
        ]
    }

    result = analyze_influencer_style(test_influencer)

    print(f"Username: {result['username']}")
    print(f"Dominant Style: {result['dominant_visual_style']}")
    print(f"Style Confidence: {result['style_confidence']}")
    print(f"Visual Type Hint: {result['visual_type_hint']}")
    print(f"Trend Relevance: {result['trend_relevance_score']}")
    print(f"Style Distribution: {result['style_distribution']}")