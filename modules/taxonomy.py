"""
Taxonomy Module - 인플루언서 분류 모듈
Expert(전문가) vs Trendsetter(트렌드세터) 이원화 분류
텍스트 + 이미지(릴스) 분석 통합
"""

from typing import Dict, List, Tuple, Optional
import re

# 이미지 분석 모듈 임포트 (순환 참조 방지를 위해 함수 내에서 임포트)
def _get_image_analyzer():
    try:
        from .image_analyzer import analyze_influencer_style, get_visual_style_vector
        return analyze_influencer_style, get_visual_style_vector
    except ImportError:
        return None, None


# 전문가 키워드 (미용사, 살롱 전문가 등)
EXPERT_KEYWORDS = [
    '미용사', '원장', '살롱', '시술', '예약', '펌', '염색약', '레시피',
    '컬러리스트', '헤어아티스트', '디렉터', '전문가', '자격증', '교육',
    '클리닉', '두피케어', '트리트먼트', '발레아쥬', '테크닉', '조색',
    '시술과정', '미용실', '바버샵', '헤어샵', '프리미엄살롱', '전담',
    '경력', '근무', '촬영', '화보', '웨딩헤어', '전문의', '피부과'
]

# 트렌드세터 키워드 (스타일 크리에이터, 인플루언서 등)
TRENDSETTER_KEYWORDS = [
    '스타일링', '데일리룩', 'OOTD', '추천', '꿀팁', '셀프', '홈케어',
    '트렌드', '패션', '일상', '크리에이터', '인플루언서', '협찬',
    '리뷰', '가성비', '꿀템', '추천템', '비교', '솔직후기', '루틴',
    '셀프케어', '셀프염색', '셀프스타일링', '뷰티러', '뷰티', 'DIY',
    '트렌드세터', '파티룩', '글램', '라이프스타일', '유튜브', 'YouTube'
]

# 키워드 가중치 (중요 키워드는 더 높은 점수)
EXPERT_WEIGHTS = {
    '원장': 3.0, '미용사': 2.5, '살롱': 2.0, '시술': 2.0, '예약': 1.5,
    '컬러리스트': 2.5, '디렉터': 2.5, '자격증': 2.0, '전문가': 1.5,
    '헤어아티스트': 2.5, '전문의': 3.0, '경력': 1.5
}

TRENDSETTER_WEIGHTS = {
    '크리에이터': 2.5, '인플루언서': 2.5, '트렌드세터': 3.0, '협찬': 2.0,
    '꿀팁': 1.5, '추천': 1.5, 'OOTD': 2.0, '데일리룩': 1.5, '리뷰': 1.5,
    '유튜브': 2.0, 'YouTube': 2.0
}


def extract_text_features(bio: str, captions: List[str]) -> Dict[str, any]:
    """
    Bio와 캡션에서 텍스트 특징을 추출합니다.

    Args:
        bio: 인플루언서 프로필 소개글
        captions: 최근 게시물 캡션 리스트

    Returns:
        텍스트 특징 딕셔너리
    """
    # 전체 텍스트 결합
    full_text = bio + ' ' + ' '.join(captions)
    full_text_lower = full_text.lower()

    # 해시태그 추출
    hashtags = re.findall(r'#(\w+)', full_text)

    # 키워드 빈도 계산
    expert_matches = []
    trend_matches = []

    for keyword in EXPERT_KEYWORDS:
        count = full_text.count(keyword)
        if count > 0:
            weight = EXPERT_WEIGHTS.get(keyword, 1.0)
            expert_matches.append({
                'keyword': keyword,
                'count': count,
                'weight': weight,
                'score': count * weight
            })

    for keyword in TRENDSETTER_KEYWORDS:
        count = full_text.count(keyword)
        if count > 0:
            weight = TRENDSETTER_WEIGHTS.get(keyword, 1.0)
            trend_matches.append({
                'keyword': keyword,
                'count': count,
                'weight': weight,
                'score': count * weight
            })

    return {
        'full_text': full_text,
        'hashtags': hashtags,
        'expert_matches': expert_matches,
        'trend_matches': trend_matches,
        'expert_score': sum(m['score'] for m in expert_matches),
        'trend_score': sum(m['score'] for m in trend_matches),
        'text_length': len(full_text)
    }


def classify_influencer(bio: str, captions: List[str], image_analysis: Optional[Dict] = None) -> Tuple[str, float, Dict]:
    """
    인플루언서를 Expert 또는 Trendsetter로 분류합니다.
    텍스트가 없거나 부족한 경우 이미지 분석 결과를 활용합니다.

    Args:
        bio: 인플루언서 프로필 소개글
        captions: 최근 게시물 캡션 리스트
        image_analysis: 이미지 분석 결과 (선택적)

    Returns:
        (분류 결과, 신뢰도 점수, 상세 분석 결과)
    """
    features = extract_text_features(bio, captions)

    expert_score = features['expert_score']
    trend_score = features['trend_score']

    total_score = expert_score + trend_score

    if total_score == 0:
        # 텍스트 키워드가 없는 경우 이미지 분석 결과 활용
        if image_analysis:
            trend_relevance = image_analysis.get('trend_relevance_score', 0.5)
            prof_score = image_analysis.get('professionalism_score', 0.5)
            dominant_style = image_analysis.get('dominant_style', '')

            # 이미지 분석 기반 분류
            if trend_relevance > prof_score and trend_relevance > 0.5:
                classification = 'Trendsetter'
                confidence = min(0.8, trend_relevance)
            elif prof_score > trend_relevance and prof_score > 0.5:
                classification = 'Expert'
                confidence = min(0.8, prof_score)
            elif dominant_style in ['trendy', 'colorful', 'minimal']:
                classification = 'Trendsetter'
                confidence = 0.6
            elif dominant_style in ['luxury', 'classic']:
                # 럭셔리/클래식 스타일은 타입을 명확히 구분하기 어려움
                classification = 'Trendsetter'  # 기본값으로 Trendsetter
                confidence = 0.5
            else:
                classification = 'Trendsetter'  # 기본값
                confidence = 0.5
        else:
            # 이미지 분석도 없는 경우 기본값
            classification = 'Trendsetter'  # Unknown 대신 Trendsetter로 기본 분류
            confidence = 0.4
    else:
        # 점수 비율로 분류
        expert_ratio = expert_score / total_score
        trend_ratio = trend_score / total_score

        if expert_ratio > trend_ratio:
            classification = 'Expert'
            confidence = expert_ratio
        elif trend_ratio > expert_ratio:
            classification = 'Trendsetter'
            confidence = trend_ratio
        else:
            # 동점인 경우 Bio 기준으로 판단
            bio_expert = sum(1 for k in EXPERT_KEYWORDS if k in bio)
            bio_trend = sum(1 for k in TRENDSETTER_KEYWORDS if k in bio)

            if bio_expert >= bio_trend:
                classification = 'Expert'
                confidence = 0.5
            else:
                classification = 'Trendsetter'
                confidence = 0.5

    analysis = {
        'classification': classification,
        'confidence': confidence,
        'expert_score': expert_score,
        'trend_score': trend_score,
        'expert_keywords_found': [m['keyword'] for m in features['expert_matches']],
        'trend_keywords_found': [m['keyword'] for m in features['trend_matches']],
        'hashtags': features['hashtags'],
        'used_image_analysis': total_score == 0 and image_analysis is not None
    }

    return classification, confidence, analysis


def get_role_vector(classification: str, confidence: float) -> List[float]:
    """
    분류 결과를 벡터로 변환합니다.

    Args:
        classification: 분류 결과 (Expert/Trendsetter)
        confidence: 신뢰도 점수

    Returns:
        역할 벡터 [expert_score, trendsetter_score]
    """
    if classification == 'Expert':
        return [confidence, 1 - confidence]
    elif classification == 'Trendsetter':
        return [1 - confidence, confidence]
    else:
        return [0.5, 0.5]


def analyze_influencer(influencer_data: Dict, use_image_analysis: bool = True) -> Dict:
    """
    인플루언서 데이터를 분석하고 분류 결과를 반환합니다.
    텍스트 분석 + 이미지(릴스) 분석을 통합합니다.

    Args:
        influencer_data: 인플루언서 JSON 데이터
        use_image_analysis: 이미지 분석 사용 여부 (트렌드세터 분석에 중요)

    Returns:
        분류 및 분석 결과
    """
    username = influencer_data.get('username', '')
    bio = influencer_data.get('bio', '')

    # 캡션 추출
    posts = influencer_data.get('recent_posts', [])
    captions = [post.get('caption', '') for post in posts]

    # 이미지 분석 데이터가 이미 있는지 확인 (MVP용 시뮬레이션 데이터)
    stored_image_analysis = influencer_data.get('image_analysis', {})

    # 텍스트 기반 분류 수행 (이미지 분석 데이터도 함께 전달)
    text_classification, text_confidence, text_analysis = classify_influencer(
        bio, captions, stored_image_analysis if stored_image_analysis else None
    )
    visual_analysis = None
    visual_type_hint = None

    if stored_image_analysis:
        # 저장된 이미지 분석 데이터 사용
        visual_analysis = stored_image_analysis
        trend_score = stored_image_analysis.get('trend_relevance_score', 0.5)
        prof_score = stored_image_analysis.get('professionalism_score', 0.5)

        # 트렌드 점수가 높으면 트렌드세터, 전문성 점수가 높으면 Expert
        if trend_score > prof_score and trend_score > 0.6:
            visual_type_hint = 'Trendsetter'
        elif prof_score > trend_score and prof_score > 0.6:
            visual_type_hint = 'Expert'
        else:
            visual_type_hint = None
    elif use_image_analysis:
        # 실제 이미지 분석 호출 (API 사용 시)
        analyze_style_func, get_vector_func = _get_image_analyzer()
        if analyze_style_func:
            visual_analysis = analyze_style_func(influencer_data)
            visual_type_hint = visual_analysis.get('visual_type_hint')

    # 최종 분류 결정 (텍스트 + 이미지 통합)
    final_classification, final_confidence = _combine_text_and_visual_classification(
        text_classification, text_confidence,
        visual_type_hint, visual_analysis
    )

    # 역할 벡터 생성
    role_vector = get_role_vector(final_classification, final_confidence)

    result = {
        'username': username,
        'classification': final_classification,
        'confidence': round(final_confidence, 3),
        'role_vector': role_vector,
        'analysis': text_analysis,
        'text_classification': text_classification,
        'text_confidence': round(text_confidence, 3)
    }

    # 이미지 분석 결과 추가
    if visual_analysis:
        result['visual_analysis'] = {
            'dominant_style': visual_analysis.get('dominant_style') or visual_analysis.get('dominant_visual_style'),
            'style_confidence': visual_analysis.get('style_confidence', 0.7),
            'trend_relevance': visual_analysis.get('trend_relevance_score', 0.5),
            'professionalism': visual_analysis.get('professionalism_score', 0.5),
            'visual_type_hint': visual_type_hint
        }

    return result


def _combine_text_and_visual_classification(
    text_class: str,
    text_conf: float,
    visual_hint: Optional[str],
    visual_analysis: Optional[Dict]
) -> Tuple[str, float]:
    """
    텍스트 분류와 이미지 분석 결과를 통합하여 최종 분류 결정

    원칙:
    1. 텍스트 분류 신뢰도가 높으면(0.7 이상) 텍스트 결과 우선
    2. 텍스트가 Unknown이면 이미지 분석 결과 사용
    3. 둘 다 있고 불일치하면 가중 평균
    """
    if not visual_hint or not visual_analysis:
        return text_class, text_conf

    visual_trend_score = visual_analysis.get('trend_relevance_score', 0.5)
    visual_prof_score = visual_analysis.get('professionalism_score', 0.5)

    # 텍스트에서 Unknown이면 이미지 분석 결과를 따름
    if text_class == 'Unknown':
        return visual_hint, 0.6

    # 텍스트 분류 신뢰도가 높으면 텍스트 결과 우선 (Expert 키워드가 많이 발견된 경우)
    if text_conf >= 0.7:
        # 텍스트와 이미지가 일치하면 신뢰도 상승
        if text_class == visual_hint:
            boosted_conf = min(1.0, text_conf + 0.1)
            return text_class, boosted_conf
        # 불일치해도 텍스트 결과 유지 (약간의 신뢰도 감소)
        return text_class, text_conf * 0.95

    # 텍스트 분류 신뢰도가 중간일 때 (0.5 ~ 0.7)
    # 텍스트와 이미지가 일치하면 신뢰도 상승
    if text_class == visual_hint:
        boosted_conf = min(1.0, text_conf + 0.15)
        return text_class, boosted_conf

    # 불일치 시: 텍스트 신뢰도가 낮을 때만 이미지 분석 고려
    if text_conf < 0.5:
        if visual_hint == 'Trendsetter' and visual_trend_score > 0.7:
            combined_conf = (text_conf * 0.4) + (visual_trend_score * 0.6)
            return 'Trendsetter', combined_conf

        if visual_hint == 'Expert' and visual_prof_score > 0.7:
            combined_conf = (text_conf * 0.4) + (visual_prof_score * 0.6)
            return 'Expert', combined_conf

    # 기본적으로 텍스트 분류 결과 유지
    return text_class, text_conf


def batch_classify(influencers: List[Dict]) -> List[Dict]:
    """
    여러 인플루언서를 일괄 분류합니다.

    Args:
        influencers: 인플루언서 데이터 리스트

    Returns:
        분류 결과 리스트
    """
    results = []
    for influencer in influencers:
        result = analyze_influencer(influencer)
        results.append(result)
    return results


# 테스트용 코드
if __name__ == "__main__":
    # 테스트 데이터
    test_influencer = {
        "username": "test_hair_master",
        "bio": "청담동 헤어살롱 원장 | 15년차 미용사 | 염색 & 펌 전문",
        "recent_posts": [
            {"caption": "오늘의 시술 - 염색 레시피 공개! #미용사일상 #살롱"},
            {"caption": "C컬 펌 시술 과정! #펌 #시술영상"}
        ]
    }

    result = analyze_influencer(test_influencer)
    print(f"Username: {result['username']}")
    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Role Vector: {result['role_vector']}")
    print(f"Expert Keywords: {result['analysis']['expert_keywords_found']}")
    print(f"Trend Keywords: {result['analysis']['trend_keywords_found']}")
