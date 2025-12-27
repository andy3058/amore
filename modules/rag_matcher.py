"""
RAG 기반 매칭 시스템
자연어 쿼리를 해석하고 제품/캠페인 특성을 고려한 인플루언서 매칭
"""

from typing import Dict, List, Tuple, Optional
import re
import math

from .taxonomy import analyze_influencer
from .fis_engine import calculate_fis_score


# ============== 캠페인/제품 특성 정의 ==============

CAMPAIGN_TYPES = {
    "신제품_런칭": {
        "keywords": ["신제품", "런칭", "출시", "새로운", "뉴", "new", "launch"],
        "preferred_influencer": "Trendsetter",
        "content_style": ["언박싱", "첫인상", "리뷰", "소개"],
        "weight_boost": {"trend_relevance": 1.3, "engagement": 1.2}
    },
    "탈모케어": {
        "keywords": ["탈모", "탈모케어", "두피", "볼륨", "가는모발", "힘없는", "빠지는"],
        "preferred_influencer": "Expert",
        "content_style": ["전문지식", "시술", "비포애프터", "과학적"],
        "weight_boost": {"expertise": 1.4, "trust": 1.3},
        "target_bio_keywords": ["두피", "탈모", "클리닉", "전문", "의사", "피부과"]
    },
    "염색_컬러": {
        "keywords": ["염색", "컬러", "색상", "발레아쥬", "하이라이트", "브릿지", "탈색"],
        "preferred_influencer": "Expert",
        "content_style": ["시술과정", "조색", "컬러리스트", "전후비교"],
        "weight_boost": {"expertise": 1.3, "visual": 1.2},
        "target_bio_keywords": ["염색", "컬러", "컬러리스트", "발레아쥬"]
    },
    "스타일링": {
        "keywords": ["스타일링", "헤어스타일", "왁스", "에센스", "세럼", "오일", "고데기"],
        "preferred_influencer": "Trendsetter",
        "content_style": ["튜토리얼", "데일리", "OOTD", "꿀팁"],
        "weight_boost": {"trend_relevance": 1.2, "engagement": 1.2},
        "target_aesthetic": ["트렌디", "스트릿", "캐주얼", "모던"]
    },
    "손상모케어": {
        "keywords": ["손상", "손상모", "데미지", "복구", "트리트먼트", "클리닉", "케라틴"],
        "preferred_influencer": "Expert",
        "content_style": ["비포애프터", "시술", "전문", "과학적"],
        "weight_boost": {"expertise": 1.3, "trust": 1.2},
        "target_bio_keywords": ["손상모", "클리닉", "트리트먼트", "케라틴", "복구"]
    },
    "프리미엄_럭셔리": {
        "keywords": ["프리미엄", "럭셔리", "고급", "VIP", "하이엔드", "명품"],
        "preferred_influencer": "Both",
        "content_style": ["럭셔리", "고급", "세련", "우아"],
        "weight_boost": {"luxury_score": 1.4, "aesthetic": 1.2},
        "target_aesthetic": ["럭셔리", "글램", "하이엔드", "우아", "세련"]
    },
    "가성비_대중": {
        "keywords": ["가성비", "저렴", "대중", "올리브영", "다이소", "드럭스토어"],
        "preferred_influencer": "Trendsetter",
        "content_style": ["솔직리뷰", "가성비", "추천", "비교"],
        "weight_boost": {"engagement": 1.3, "relatability": 1.2},
        "target_bio_keywords": ["가성비", "리뷰", "추천", "솔직", "저렴이"]
    },
    "향수_프래그런스": {
        "keywords": ["향수", "퍼퓸", "향기", "프래그런스", "디퓨저", "센트"],
        "preferred_influencer": "Trendsetter",
        "content_style": ["감성", "라이프스타일", "무드", "분위기"],
        "weight_boost": {"aesthetic": 1.3, "lifestyle": 1.2},
        "target_aesthetic": ["미니멀", "내추럴", "힐링", "감성", "모던"]
    },
    "홈케어_셀프": {
        "keywords": ["홈케어", "셀프", "집에서", "DIY", "혼자", "쉽게"],
        "preferred_influencer": "Trendsetter",
        "content_style": ["튜토리얼", "꿀팁", "간편", "일상"],
        "weight_boost": {"engagement": 1.2, "relatability": 1.3},
        "target_bio_keywords": ["홈케어", "셀프", "팁", "꿀팁", "튜토리얼"]
    },
    "살롱_전문가용": {
        "keywords": ["살롱", "전문가용", "프로", "미용사", "미용실", "시술"],
        "preferred_influencer": "Expert",
        "content_style": ["전문시술", "교육", "테크닉", "프로"],
        "weight_boost": {"expertise": 1.5, "professionalism": 1.3},
        "target_bio_keywords": ["살롱", "원장", "디렉터", "미용사", "프로"]
    }
}

PRODUCT_CATEGORIES = {
    "샴푸": {
        "related_campaigns": ["탈모케어", "손상모케어", "홈케어_셀프", "가성비_대중"],
        "content_keywords": ["세정력", "두피", "거품", "향", "성분"],
        "expert_weight": 0.5,
        "trendsetter_weight": 0.5
    },
    "트리트먼트": {
        "related_campaigns": ["손상모케어", "홈케어_셀프", "프리미엄_럭셔리"],
        "content_keywords": ["영양", "윤기", "촉촉", "복구", "코팅"],
        "expert_weight": 0.6,
        "trendsetter_weight": 0.4
    },
    "염색약": {
        "related_campaigns": ["염색_컬러", "살롱_전문가용", "홈케어_셀프"],
        "content_keywords": ["발색", "지속력", "색상", "손상", "커버력"],
        "expert_weight": 0.7,
        "trendsetter_weight": 0.3
    },
    "두피케어": {
        "related_campaigns": ["탈모케어", "손상모케어", "살롱_전문가용"],
        "content_keywords": ["두피", "탈모", "스케일링", "영양", "진정"],
        "expert_weight": 0.8,
        "trendsetter_weight": 0.2
    },
    "스타일링": {
        "related_campaigns": ["스타일링", "홈케어_셀프", "가성비_대중"],
        "content_keywords": ["고정력", "볼륨", "웨이브", "자연스러움", "지속력"],
        "expert_weight": 0.3,
        "trendsetter_weight": 0.7
    },
    "에센스": {
        "related_campaigns": ["손상모케어", "스타일링", "프리미엄_럭셔리"],
        "content_keywords": ["윤기", "부드러움", "영양", "코팅", "가벼움"],
        "expert_weight": 0.4,
        "trendsetter_weight": 0.6
    },
    "향수": {
        "related_campaigns": ["향수_프래그런스", "프리미엄_럭셔리"],
        "content_keywords": ["향", "지속력", "잔향", "분위기", "무드"],
        "expert_weight": 0.2,
        "trendsetter_weight": 0.8
    }
}


# ============== 자연어 쿼리 파서 ==============

class QueryParser:
    """자연어 캠페인 쿼리를 분석하고 구조화된 요구사항으로 변환"""

    def __init__(self):
        self.campaign_types = CAMPAIGN_TYPES
        self.product_categories = PRODUCT_CATEGORIES

    def parse(self, query: str, brand_data: Dict = None) -> Dict:
        """
        자연어 쿼리를 파싱하여 캠페인 요구사항 추출

        Args:
            query: 자연어 캠페인 설명 (예: "신제품 탈모샴푸 런칭 캠페인")
            brand_data: 브랜드 정보 (선택)

        Returns:
            파싱된 캠페인 요구사항
        """
        query_lower = query.lower()

        result = {
            "original_query": query,
            "detected_campaigns": [],
            "detected_products": [],
            "preferred_influencer_type": None,
            "weight_adjustments": {},
            "target_keywords": [],
            "target_aesthetics": [],
            "confidence": 0.0
        }

        # 1. 캠페인 유형 감지
        campaign_scores = {}
        for campaign_name, campaign_info in self.campaign_types.items():
            score = 0
            for keyword in campaign_info["keywords"]:
                if keyword in query_lower:
                    score += 1
            if score > 0:
                campaign_scores[campaign_name] = score

        # 점수순 정렬
        sorted_campaigns = sorted(campaign_scores.items(), key=lambda x: x[1], reverse=True)
        result["detected_campaigns"] = [c[0] for c in sorted_campaigns[:3]]

        # 2. 제품 카테고리 감지
        for product_name in self.product_categories.keys():
            if product_name in query or product_name.lower() in query_lower:
                result["detected_products"].append(product_name)

        # 브랜드 데이터에서 제품 카테고리 추출
        if brand_data:
            product_cats = brand_data.get("product_categories", [])
            for cat in product_cats:
                if cat not in result["detected_products"]:
                    result["detected_products"].append(cat)

        # 3. 선호 인플루언서 타입 결정
        expert_score = 0
        trendsetter_score = 0

        for campaign in result["detected_campaigns"]:
            camp_info = self.campaign_types.get(campaign, {})
            pref = camp_info.get("preferred_influencer", "Both")
            if pref == "Expert":
                expert_score += 2
            elif pref == "Trendsetter":
                trendsetter_score += 2
            else:
                expert_score += 1
                trendsetter_score += 1

        for product in result["detected_products"]:
            prod_info = self.product_categories.get(product, {})
            expert_score += prod_info.get("expert_weight", 0.5) * 2
            trendsetter_score += prod_info.get("trendsetter_weight", 0.5) * 2

        if expert_score > trendsetter_score * 1.3:
            result["preferred_influencer_type"] = "Expert"
        elif trendsetter_score > expert_score * 1.3:
            result["preferred_influencer_type"] = "Trendsetter"
        else:
            result["preferred_influencer_type"] = "Both"

        # 4. 가중치 조정 수집
        for campaign in result["detected_campaigns"]:
            camp_info = self.campaign_types.get(campaign, {})
            for weight_key, weight_val in camp_info.get("weight_boost", {}).items():
                if weight_key in result["weight_adjustments"]:
                    result["weight_adjustments"][weight_key] = max(
                        result["weight_adjustments"][weight_key], weight_val
                    )
                else:
                    result["weight_adjustments"][weight_key] = weight_val

        # 5. 타겟 키워드 수집
        for campaign in result["detected_campaigns"]:
            camp_info = self.campaign_types.get(campaign, {})
            result["target_keywords"].extend(camp_info.get("target_bio_keywords", []))
            result["target_aesthetics"].extend(camp_info.get("target_aesthetic", []))

        # 중복 제거
        result["target_keywords"] = list(set(result["target_keywords"]))
        result["target_aesthetics"] = list(set(result["target_aesthetics"]))

        # 6. 신뢰도 계산
        total_matches = len(result["detected_campaigns"]) + len(result["detected_products"])
        result["confidence"] = min(1.0, total_matches * 0.25 + 0.3)

        return result


# ============== RAG 기반 매칭 엔진 ==============

class RAGMatcher:
    """
    RAG(Retrieval-Augmented Generation) 기반 매칭 엔진
    캠페인 요구사항과 인플루언서 특성을 다차원으로 매칭
    """

    def __init__(self):
        self.query_parser = QueryParser()

    def match(
        self,
        query: str,
        brand_data: Dict,
        influencers: List[Dict],
        top_k: int = 10,
        min_fis: float = 60.0
    ) -> Dict:
        """
        캠페인 쿼리 기반 인플루언서 매칭

        Args:
            query: 캠페인 설명 (자연어)
            brand_data: 브랜드 데이터
            influencers: 인플루언서 리스트
            top_k: 반환할 상위 K명
            min_fis: 최소 FIS 점수

        Returns:
            매칭 결과
        """
        # 1. 쿼리 파싱
        parsed_query = self.query_parser.parse(query, brand_data)

        # 2. 인플루언서별 점수 계산
        scored_influencers = []

        for influencer in influencers:
            # FIS 점수 계산
            fis_result = calculate_fis_score(influencer)
            fis_score = fis_result['fis_score']

            if fis_score < min_fis:
                continue

            # 분류 분석
            taxonomy_result = analyze_influencer(influencer)
            classification = taxonomy_result['classification']

            # Unknown 분류 처리
            if classification == 'Unknown':
                image_analysis = influencer.get('image_analysis', {})
                if image_analysis:
                    trend_score = image_analysis.get('trend_relevance_score', 0.5)
                    prof_score = image_analysis.get('professionalism_score', 0.5)
                    classification = 'Trendsetter' if trend_score > prof_score else 'Expert'

            # 다차원 점수 계산
            scores = self._calculate_multidimensional_score(
                influencer, parsed_query, classification, taxonomy_result
            )

            # FIS 반영
            scores['final_score'] = scores['weighted_total'] * (fis_score / 100)

            scored_influencers.append({
                'influencer': influencer,
                'classification': classification,
                'taxonomy': taxonomy_result,
                'fis_score': fis_score,
                'fis_verdict': fis_result['verdict'],
                'scores': scores
            })

        # 3. 정렬 및 다양성 보장
        sorted_results = self._select_with_diversity(
            scored_influencers, top_k, parsed_query['preferred_influencer_type']
        )

        # 4. 결과 포맷팅
        return self._format_results(sorted_results, brand_data, parsed_query, len(influencers))

    def _calculate_multidimensional_score(
        self,
        influencer: Dict,
        parsed_query: Dict,
        classification: str,
        taxonomy_result: Dict
    ) -> Dict:
        """다차원 매칭 점수 계산"""

        scores = {
            'campaign_fit': 0.0,      # 캠페인 적합도
            'product_fit': 0.0,       # 제품 적합도
            'content_fit': 0.0,       # 콘텐츠 스타일 적합도
            'aesthetic_fit': 0.0,     # 미학적 적합도
            'expertise_fit': 0.0,     # 전문성 적합도
            'engagement_fit': 0.0,    # 참여도 적합도
            'weighted_total': 0.0     # 가중 합계
        }

        bio = influencer.get('bio', '').lower()
        image_analysis = influencer.get('image_analysis', {})
        aesthetic_tags = image_analysis.get('aesthetic_tags', [])
        hair_tags = image_analysis.get('hair_style_tags', [])
        posts = influencer.get('recent_posts', [])
        captions = ' '.join([p.get('caption', '') for p in posts]).lower()

        # 1. 캠페인 적합도
        campaign_score = 0
        for campaign in parsed_query['detected_campaigns']:
            camp_info = CAMPAIGN_TYPES.get(campaign, {})

            # 선호 인플루언서 타입 매칭
            pref_type = camp_info.get('preferred_influencer', 'Both')
            if pref_type == classification or pref_type == 'Both':
                campaign_score += 0.3

            # Bio 키워드 매칭
            target_keywords = camp_info.get('target_bio_keywords', [])
            for kw in target_keywords:
                if kw in bio:
                    campaign_score += 0.15

        scores['campaign_fit'] = min(1.0, campaign_score)

        # 2. 제품 적합도
        product_score = 0
        for product in parsed_query['detected_products']:
            prod_info = PRODUCT_CATEGORIES.get(product, {})

            # 인플루언서 타입별 가중치
            if classification == 'Expert':
                product_score += prod_info.get('expert_weight', 0.5) * 0.5
            else:
                product_score += prod_info.get('trendsetter_weight', 0.5) * 0.5

            # 콘텐츠 키워드 매칭
            content_keywords = prod_info.get('content_keywords', [])
            for kw in content_keywords:
                if kw in captions or kw in bio:
                    product_score += 0.1

        scores['product_fit'] = min(1.0, product_score)

        # 3. 콘텐츠 스타일 적합도
        content_score = 0
        for campaign in parsed_query['detected_campaigns']:
            camp_info = CAMPAIGN_TYPES.get(campaign, {})
            content_styles = camp_info.get('content_style', [])

            for style in content_styles:
                style_lower = style.lower()
                if style_lower in captions or style_lower in bio:
                    content_score += 0.2

        # 헤어 스타일 태그 보너스
        if hair_tags:
            content_score += 0.2

        scores['content_fit'] = min(1.0, content_score)

        # 4. 미학적 적합도
        aesthetic_score = 0
        target_aesthetics = parsed_query.get('target_aesthetics', [])

        for target in target_aesthetics:
            if target in aesthetic_tags:
                aesthetic_score += 0.25

        # 이미지 분석 스타일 매칭
        dominant_style = image_analysis.get('dominant_style', '')
        if dominant_style:
            style_map = {
                'luxury': ['럭셔리', '글램', '하이엔드', '프리미엄'],
                'trendy': ['트렌디', 'Y2K', '스트릿', '모던'],
                'natural': ['내추럴', '힐링', '친환경', '자연'],
                'minimal': ['미니멀', '클린', '심플', '모던'],
                'colorful': ['컬러풀', '비비드', '팝', '개성']
            }
            matching_aesthetics = style_map.get(dominant_style, [])
            for aesthetic in target_aesthetics:
                if aesthetic in matching_aesthetics:
                    aesthetic_score += 0.15

        scores['aesthetic_fit'] = min(1.0, aesthetic_score)

        # 5. 전문성 적합도
        expertise_score = 0

        if classification == 'Expert':
            expertise_score += 0.3

            # 전문가 키워드
            expert_keywords = ['원장', '디렉터', '전문', '자격증', '경력', '살롱', '클리닉']
            for kw in expert_keywords:
                if kw in bio:
                    expertise_score += 0.1

        # 이미지 분석 전문성 점수
        prof_score = image_analysis.get('professionalism_score', 0.5)
        expertise_score += prof_score * 0.3

        scores['expertise_fit'] = min(1.0, expertise_score)

        # 6. 참여도 적합도 (engagement)
        engagement_score = 0

        if posts:
            avg_likes = sum(p.get('likes', 0) for p in posts) / len(posts)
            followers = influencer.get('followers', 1)
            engagement_rate = avg_likes / followers if followers > 0 else 0

            # 참여율 기준 점수화 (3% 이상 우수)
            if engagement_rate >= 0.05:
                engagement_score = 1.0
            elif engagement_rate >= 0.03:
                engagement_score = 0.8
            elif engagement_rate >= 0.02:
                engagement_score = 0.6
            else:
                engagement_score = 0.4

        # 트렌드 관련성 점수 반영
        trend_relevance = image_analysis.get('trend_relevance_score', 0.5)
        engagement_score = (engagement_score + trend_relevance) / 2

        scores['engagement_fit'] = engagement_score

        # 7. 가중 합계 계산
        weight_adjustments = parsed_query.get('weight_adjustments', {})

        # 기본 가중치
        weights = {
            'campaign_fit': 0.25,
            'product_fit': 0.20,
            'content_fit': 0.15,
            'aesthetic_fit': 0.15,
            'expertise_fit': 0.15,
            'engagement_fit': 0.10
        }

        # 가중치 조정 적용
        if 'expertise' in weight_adjustments:
            weights['expertise_fit'] *= weight_adjustments['expertise']
        if 'engagement' in weight_adjustments:
            weights['engagement_fit'] *= weight_adjustments['engagement']
        if 'aesthetic' in weight_adjustments:
            weights['aesthetic_fit'] *= weight_adjustments['aesthetic']
        if 'trend_relevance' in weight_adjustments:
            weights['engagement_fit'] *= weight_adjustments['trend_relevance']
        if 'luxury_score' in weight_adjustments:
            weights['aesthetic_fit'] *= weight_adjustments['luxury_score']

        # 정규화
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # 가중 합계
        weighted_total = sum(scores[k] * weights[k] for k in weights.keys())
        scores['weighted_total'] = weighted_total

        return scores

    def _select_with_diversity(
        self,
        scored_results: List[Dict],
        top_k: int,
        preferred_type: str
    ) -> List[Dict]:
        """다양성을 보장하면서 상위 K명 선택"""

        # 타입별 분류
        experts = [r for r in scored_results if r['classification'] == 'Expert']
        trendsetters = [r for r in scored_results if r['classification'] == 'Trendsetter']

        # 점수순 정렬
        experts.sort(key=lambda x: x['scores']['final_score'], reverse=True)
        trendsetters.sort(key=lambda x: x['scores']['final_score'], reverse=True)

        # 비율 결정
        if preferred_type == 'Expert':
            expert_ratio = 0.65
        elif preferred_type == 'Trendsetter':
            expert_ratio = 0.35
        else:
            expert_ratio = 0.5

        expert_count = min(len(experts), max(1, int(top_k * expert_ratio)))
        trend_count = min(len(trendsetters), top_k - expert_count)

        # 부족분 채우기
        if expert_count + trend_count < top_k:
            remaining = top_k - expert_count - trend_count
            if len(experts) > expert_count:
                expert_count += min(len(experts) - expert_count, remaining)
                remaining = top_k - expert_count - trend_count
            if remaining > 0 and len(trendsetters) > trend_count:
                trend_count += min(len(trendsetters) - trend_count, remaining)

        # 선택 및 정렬
        selected = experts[:expert_count] + trendsetters[:trend_count]
        selected.sort(key=lambda x: x['scores']['final_score'], reverse=True)

        return selected[:top_k]

    def _format_results(
        self,
        results: List[Dict],
        brand_data: Dict,
        parsed_query: Dict,
        total_analyzed: int
    ) -> Dict:
        """결과 포맷팅"""

        recommendations = []

        for i, result in enumerate(results, 1):
            inf = result['influencer']
            scores = result['scores']

            # 추천 사유 생성
            reason = self._generate_reason(inf, result, brand_data, parsed_query)

            recommendations.append({
                'rank': i,
                'username': inf.get('username', ''),
                'followers': inf.get('followers', 0),
                'type': result['classification'],
                'match_score': round(scores['final_score'] * 100, 1),
                'fis_score': result['fis_score'],
                'reason': reason,
                'score_breakdown': {
                    'campaign_fit': round(scores['campaign_fit'] * 100, 1),
                    'product_fit': round(scores['product_fit'] * 100, 1),
                    'content_fit': round(scores['content_fit'] * 100, 1),
                    'aesthetic_fit': round(scores['aesthetic_fit'] * 100, 1),
                    'expertise_fit': round(scores['expertise_fit'] * 100, 1),
                    'engagement_fit': round(scores['engagement_fit'] * 100, 1)
                },
                'details': {
                    'bio': inf.get('bio', ''),
                    'fis_verdict': result['fis_verdict'],
                    'image_analysis': inf.get('image_analysis', {})
                }
            })

        return {
            'query_analysis': {
                'original_query': parsed_query['original_query'],
                'detected_campaigns': parsed_query['detected_campaigns'],
                'detected_products': parsed_query['detected_products'],
                'preferred_type': parsed_query['preferred_influencer_type'],
                'confidence': parsed_query['confidence']
            },
            'brand_info': {
                'name': brand_data.get('brand_name', ''),
                'aesthetic_style': brand_data.get('aesthetic_style', ''),
                'product_categories': brand_data.get('product_categories', [])
            },
            'total_analyzed': total_analyzed,
            'total_matched': len(results),
            'recommendations': recommendations
        }

    def _generate_reason(
        self,
        influencer: Dict,
        result: Dict,
        brand_data: Dict,
        parsed_query: Dict
    ) -> str:
        """개인화된 추천 사유 생성"""

        username = influencer.get('username', '')
        classification = result['classification']
        scores = result['scores']
        bio = influencer.get('bio', '')
        image_analysis = influencer.get('image_analysis', {})

        brand_name = brand_data.get('brand_name', '브랜드')

        # 가장 높은 적합도 찾기
        score_items = [
            ('캠페인 적합도', scores['campaign_fit']),
            ('제품 적합도', scores['product_fit']),
            ('콘텐츠 스타일', scores['content_fit']),
            ('미적 감각', scores['aesthetic_fit']),
            ('전문성', scores['expertise_fit']),
            ('참여도', scores['engagement_fit'])
        ]
        top_scores = sorted(score_items, key=lambda x: x[1], reverse=True)[:2]

        # 인플루언서 특성 설명
        vibe = image_analysis.get('vibe', '')
        if vibe:
            style_desc = vibe
        elif classification == 'Expert':
            style_desc = "전문적인 시술 노하우를 공유하는 헤어 전문가"
        else:
            style_desc = "트렌디한 스타일링 콘텐츠를 제작하는 인플루언서"

        # 캠페인 맞춤 설명
        campaigns = parsed_query.get('detected_campaigns', [])
        campaign_desc = ""
        if campaigns:
            if '탈모케어' in campaigns or '두피케어' in campaigns:
                campaign_desc = "두피/탈모 케어 캠페인에 높은 전문성을 보유"
            elif '염색_컬러' in campaigns:
                campaign_desc = "염색 시술 콘텐츠에 강점"
            elif '스타일링' in campaigns:
                campaign_desc = "스타일링 튜토리얼 콘텐츠에 강점"
            elif '프리미엄_럭셔리' in campaigns:
                campaign_desc = "럭셔리 브랜드 이미지와 높은 조화"
            elif '향수_프래그런스' in campaigns:
                campaign_desc = "감성적 라이프스타일 콘텐츠에 강점"
            else:
                campaign_desc = f"{campaigns[0].replace('_', ' ')} 캠페인에 적합"

        # 강점 설명
        strength_desc = f"{top_scores[0][0]}({int(top_scores[0][1]*100)}%), {top_scores[1][0]}({int(top_scores[1][1]*100)}%)에서 높은 점수"

        # 팔로워 규모
        followers = influencer.get('followers', 0)
        if followers >= 100000:
            follower_desc = f"팔로워 {followers/10000:.1f}만명"
        else:
            follower_desc = f"팔로워 {followers/10000:.1f}만명"

        # 최종 조합
        reason = f"@{username}은 {style_desc}입니다. {campaign_desc}하며, {strength_desc}를 기록했습니다. '{brand_name}' 브랜드 캠페인에 적합합니다. ({follower_desc})"

        return reason


# ============== 통합 매칭 함수 ==============

def match_with_campaign(
    campaign_query: str,
    brand_data: Dict,
    influencers: List[Dict],
    top_k: int = 10,
    min_fis: float = 60.0
) -> Dict:
    """
    캠페인 쿼리 기반 인플루언서 매칭 (메인 함수)

    Args:
        campaign_query: 캠페인 설명 (예: "신제품 탈모샴푸 런칭, 전문가 중심")
        brand_data: 브랜드 데이터
        influencers: 인플루언서 리스트
        top_k: 반환할 상위 K명
        min_fis: 최소 FIS 점수

    Returns:
        매칭 결과
    """
    matcher = RAGMatcher()
    return matcher.match(campaign_query, brand_data, influencers, top_k, min_fis)


# 테스트 코드
if __name__ == "__main__":
    import json
    import os

    # 데이터 로드
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'influencers_data.json')
    brand_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'amore_brands.json')

    with open(data_path, 'r', encoding='utf-8') as f:
        influencer_data = json.load(f)

    with open(brand_path, 'r', encoding='utf-8') as f:
        brand_data = json.load(f)

    influencers = influencer_data['influencers']

    # 테스트 쿼리
    test_queries = [
        ("탈모샴푸 신제품 런칭 캠페인", "려"),
        ("트렌디한 염색약 홍보, MZ세대 타겟", "미쟝센"),
        ("두피 스킨케어 전문 캠페인", "라보에이치"),
        ("지속가능한 향수 라이프스타일 캠페인", "롱테이크")
    ]

    for query, brand_name in test_queries:
        brand = brand_data['brands'].get(brand_name, {})

        print(f"\n{'='*60}")
        print(f"쿼리: {query}")
        print(f"브랜드: {brand_name}")
        print('='*60)

        results = match_with_campaign(query, brand, influencers, top_k=5)

        print(f"\n[쿼리 분석]")
        print(f"  감지된 캠페인: {results['query_analysis']['detected_campaigns']}")
        print(f"  감지된 제품: {results['query_analysis']['detected_products']}")
        print(f"  선호 타입: {results['query_analysis']['preferred_type']}")

        print(f"\n[추천 결과]")
        for rec in results['recommendations'][:3]:
            print(f"\n  {rec['rank']}. @{rec['username']} ({rec['type']})")
            print(f"     매칭: {rec['match_score']}%")
            print(f"     사유: {rec['reason'][:80]}...")