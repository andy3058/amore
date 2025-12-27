"""
Matcher Module - 브랜드-인플루언서 매칭 알고리즘
코사인 유사도 기반 최적 매칭 및 추천
"""

from typing import Dict, List, Tuple, Optional
import math
import json

from .taxonomy import analyze_influencer, get_role_vector
from .fis_engine import calculate_fis_score
from .brand_analyzer import analyze_brand, create_brand_vector, get_matching_criteria


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    두 벡터 간의 코사인 유사도를 계산합니다.

    Args:
        vec1: 첫 번째 벡터
        vec2: 두 번째 벡터

    Returns:
        코사인 유사도 (-1 ~ 1)
    """
    if len(vec1) != len(vec2):
        # 벡터 길이가 다르면 짧은 쪽을 0으로 패딩
        max_len = max(len(vec1), len(vec2))
        vec1 = vec1 + [0] * (max_len - len(vec1))
        vec2 = vec2 + [0] * (max_len - len(vec2))

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def create_influencer_vector(influencer_data: Dict, taxonomy_result: Dict) -> List[float]:
    """
    인플루언서 데이터를 벡터로 변환합니다.
    이미지 분석 데이터가 있으면 활용합니다.

    벡터 구성 (브랜드 벡터와 동일한 차원):
    [luxury_score, professional_score, expert_score, trendsetter_score,
     colorfulness, natural_score, modern_score]

    Args:
        influencer_data: 인플루언서 데이터
        taxonomy_result: 분류 결과

    Returns:
        인플루언서 벡터
    """
    # 역할 벡터 (Expert/Trendsetter)
    role_vector = taxonomy_result.get('role_vector', [0.5, 0.5])
    expert_score = role_vector[0]
    trendsetter_score = role_vector[1]

    # 이미지 분석 데이터 확인
    image_analysis = influencer_data.get('image_analysis', {})

    # Bio 기반 럭셔리 점수 추정
    bio = influencer_data.get('bio', '')
    luxury_keywords = ['프리미엄', '럭셔리', '고급', 'VIP', '청담', '압구정', '프레스티지']
    mass_keywords = ['가성비', '저렴', '다이소', '올리브영', '저렴이']

    luxury_count = sum(1 for kw in luxury_keywords if kw in bio)
    mass_count = sum(1 for kw in mass_keywords if kw in bio)

    if luxury_count > mass_count:
        luxury_score = 0.7 + (luxury_count * 0.1)
    elif mass_count > luxury_count:
        luxury_score = 0.3 - (mass_count * 0.05)
    else:
        luxury_score = 0.5

    # 이미지 분석에서 스타일 추출
    if image_analysis:
        dominant_style = image_analysis.get('dominant_style', '')
        aesthetic_tags = image_analysis.get('aesthetic_tags', [])

        # 이미지 분석 기반 럭셔리 점수 조정
        if dominant_style == 'luxury' or '럭셔리' in aesthetic_tags or '하이엔드' in aesthetic_tags:
            luxury_score = max(luxury_score, 0.85)
        elif dominant_style == 'minimal' or '미니멀' in aesthetic_tags:
            luxury_score = max(luxury_score, 0.6)
        elif '가성비' in aesthetic_tags or '저렴이' in aesthetic_tags:
            luxury_score = min(luxury_score, 0.35)

    luxury_score = max(0, min(1, luxury_score))

    # 전문성 점수 (Expert일수록 높음)
    professional_score = expert_score * 0.8 + 0.1
    if image_analysis:
        prof_from_image = image_analysis.get('professionalism_score', 0.5)
        professional_score = (professional_score + prof_from_image) / 2

    # 스타일 점수 (캡션 분석 + 이미지 분석)
    posts = influencer_data.get('recent_posts', [])
    captions = ' '.join([post.get('caption', '') for post in posts])

    colorful_keywords = ['화려', '컬러풀', '비비드', '핑크', '블루', '레드']
    natural_keywords = ['자연', '내추럴', '비건', '유기농', '친환경', '힐링']
    modern_keywords = ['트렌드', '트렌디', 'MZ', '힙', '모던', 'Y2K', '스트릿']

    colorfulness = sum(1 for kw in colorful_keywords if kw in captions) * 0.2
    natural_score = sum(1 for kw in natural_keywords if kw in captions) * 0.2
    modern_score = sum(1 for kw in modern_keywords if kw in captions) * 0.2

    # 이미지 분석에서 스타일 점수 보완
    if image_analysis:
        dominant_style = image_analysis.get('dominant_style', '')
        aesthetic_tags = image_analysis.get('aesthetic_tags', [])

        # 컬러풀
        if dominant_style == 'colorful' or '컬러풀' in aesthetic_tags or '키치' in aesthetic_tags:
            colorfulness = max(colorfulness, 0.8)

        # 내추럴
        if dominant_style == 'natural' or '내추럴' in aesthetic_tags or '힐링' in aesthetic_tags:
            natural_score = max(natural_score, 0.8)

        # 모던/트렌디
        if dominant_style == 'trendy' or '트렌디' in aesthetic_tags or 'Y2K' in aesthetic_tags:
            modern_score = max(modern_score, 0.8)

        # trend_relevance_score 반영
        trend_relevance = image_analysis.get('trend_relevance_score', 0.5)
        modern_score = (modern_score + trend_relevance) / 2

    # 정규화
    colorfulness = min(1, colorfulness + 0.3)
    natural_score = min(1, natural_score + 0.3)
    modern_score = min(1, modern_score + 0.3)

    vector = [
        luxury_score,
        professional_score,
        expert_score,
        trendsetter_score,
        colorfulness,
        natural_score,
        modern_score
    ]

    # L2 정규화
    magnitude = math.sqrt(sum(v ** 2 for v in vector))
    if magnitude > 0:
        vector = [v / magnitude for v in vector]

    return vector


def calculate_match_score(
    brand_vector: List[float],
    influencer_vector: List[float],
    fis_score: float,
    role_weight: float = 1.0
) -> float:
    """
    브랜드-인플루언서 매칭 점수를 계산합니다.

    Final Score = Similarity × FIS_normalized × Role_weight (0-1 범위로 클램핑)

    Args:
        brand_vector: 브랜드 벡터
        influencer_vector: 인플루언서 벡터
        fis_score: FIS 점수 (0-100)
        role_weight: 역할 가중치 (1.0 기준, 최대 1.1로 제한)

    Returns:
        최종 매칭 점수 (0-1, 절대 1을 초과하지 않음)
    """
    # 코사인 유사도 (0-1로 정규화)
    similarity = cosine_similarity(brand_vector, influencer_vector)
    similarity_normalized = (similarity + 1) / 2  # -1~1 -> 0~1

    # FIS 정규화 (0-1로 클램핑)
    fis_normalized = max(0.0, min(1.0, fis_score / 100))

    # role_weight 범위 제한 (0.8 ~ 1.1)
    role_weight = max(0.8, min(1.1, role_weight))

    # 최종 점수 계산
    raw_score = similarity_normalized * fis_normalized * role_weight

    # 최종 점수를 0-1 범위로 클램핑 (100% 초과 방지)
    final_score = max(0.0, min(1.0, raw_score))

    return final_score


def match_influencers(
    brand_data: Dict,
    influencers: List[Dict],
    top_k: int = 5,
    min_fis: float = 60.0,
    ensure_diversity: bool = True
) -> List[Dict]:
    """
    브랜드에 맞는 인플루언서를 매칭합니다.
    다양성을 보장하여 Expert와 Trendsetter가 균형있게 추천됩니다.

    Args:
        brand_data: 브랜드 데이터
        influencers: 인플루언서 리스트
        top_k: 반환할 상위 K명
        min_fis: 최소 FIS 점수
        ensure_diversity: True이면 Expert와 Trendsetter를 균형있게 포함

    Returns:
        매칭 결과 리스트
    """
    # 브랜드 분석
    brand_analysis = analyze_brand(brand_data)
    brand_vector = brand_analysis['brand_vector']
    matching_criteria = get_matching_criteria(brand_analysis)

    # 선호 인플루언서 타입
    preferred_type = matching_criteria['preferred_type']

    all_results = []

    for influencer in influencers:
        # 분류 분석
        taxonomy_result = analyze_influencer(influencer)

        # FIS 점수 계산
        fis_result = calculate_fis_score(influencer)
        fis_score = fis_result['fis_score']

        # FIS 필터링
        if fis_score < min_fis:
            continue

        # 인플루언서 벡터 생성
        influencer_vector = create_influencer_vector(influencer, taxonomy_result)

        # 역할 가중치 계산 (범위 조정: 0.9 ~ 1.1로 제한하여 100% 초과 방지)
        classification = taxonomy_result['classification']
        if classification == 'Unknown':
            # Unknown인 경우 이미지 분석 결과로 재분류 시도
            image_analysis = influencer.get('image_analysis', {})
            if image_analysis:
                trend_score = image_analysis.get('trend_relevance_score', 0.5)
                prof_score = image_analysis.get('professionalism_score', 0.5)
                if trend_score > prof_score:
                    classification = 'Trendsetter'
                else:
                    classification = 'Expert'

        if preferred_type == 'Both':
            role_weight = 1.0
        elif preferred_type == classification:
            role_weight = 1.08  # 선호 타입 보너스 (기존 1.2 -> 1.08로 축소)
        else:
            role_weight = 0.92  # 비선호 타입 패널티 (기존 0.8 -> 0.92로 완화)

        # 매칭 점수 계산
        match_score = calculate_match_score(
            brand_vector, influencer_vector, fis_score, role_weight
        )

        # 결과 저장
        all_results.append({
            'username': influencer.get('username', ''),
            'followers': influencer.get('followers', 0),
            'classification': classification,
            'confidence': taxonomy_result['confidence'],
            'fis_score': fis_score,
            'fis_verdict': fis_result['verdict'],
            'match_score': round(match_score, 4),
            'similarity': round(cosine_similarity(brand_vector, influencer_vector), 4),
            'influencer_vector': influencer_vector,
            'bio': influencer.get('bio', ''),
            'taxonomy_analysis': taxonomy_result['analysis'],
            'fis_breakdown': fis_result['breakdown'],
            'image_analysis': influencer.get('image_analysis', {})
        })

    # 다양성 보장 로직
    if ensure_diversity and len(all_results) >= top_k:
        return _select_diverse_results(all_results, top_k, preferred_type)

    # 매칭 점수로 정렬
    all_results.sort(key=lambda x: x['match_score'], reverse=True)
    return all_results[:top_k]


def _select_diverse_results(
    results: List[Dict],
    top_k: int,
    preferred_type: str
) -> List[Dict]:
    """
    다양성을 보장하면서 상위 K명을 선택합니다.
    Expert와 Trendsetter가 균형있게 포함되도록 합니다.

    Args:
        results: 전체 매칭 결과
        top_k: 선택할 인원 수
        preferred_type: 선호하는 인플루언서 타입

    Returns:
        다양성이 보장된 결과 리스트
    """
    # 타입별로 분류
    experts = [r for r in results if r['classification'] == 'Expert']
    trendsetters = [r for r in results if r['classification'] in ['Trendsetter', 'Unknown']]

    # 각 그룹을 점수순으로 정렬
    experts.sort(key=lambda x: x['match_score'], reverse=True)
    trendsetters.sort(key=lambda x: x['match_score'], reverse=True)

    # 균형있는 선택 비율 결정
    if preferred_type == 'Expert':
        # Expert 선호: 60% Expert, 40% Trendsetter
        expert_count = min(len(experts), max(1, int(top_k * 0.6)))
        trend_count = min(len(trendsetters), top_k - expert_count)
    elif preferred_type == 'Trendsetter':
        # Trendsetter 선호: 40% Expert, 60% Trendsetter
        trend_count = min(len(trendsetters), max(1, int(top_k * 0.6)))
        expert_count = min(len(experts), top_k - trend_count)
    else:
        # Both: 50% 균등 분배
        expert_count = min(len(experts), max(1, top_k // 2))
        trend_count = min(len(trendsetters), top_k - expert_count)

    # 부족한 경우 다른 그룹에서 채우기
    if expert_count + trend_count < top_k:
        remaining = top_k - expert_count - trend_count
        if len(experts) > expert_count:
            additional = min(len(experts) - expert_count, remaining)
            expert_count += additional
            remaining -= additional
        if remaining > 0 and len(trendsetters) > trend_count:
            trend_count += min(len(trendsetters) - trend_count, remaining)

    # 선택된 결과 조합
    selected = experts[:expert_count] + trendsetters[:trend_count]

    # 최종 점수순 정렬
    selected.sort(key=lambda x: x['match_score'], reverse=True)

    return selected[:top_k]


def generate_recommendation_reason(
    brand_data: Dict,
    influencer_result: Dict
) -> str:
    """
    인플루언서의 실제 특성을 반영한 구체적인 추천 사유를 생성합니다.

    Args:
        brand_data: 브랜드 데이터
        influencer_result: 인플루언서 매칭 결과 (bio, image_analysis 포함)

    Returns:
        구체적이고 개인화된 추천 사유 문자열
    """
    username = influencer_result['username']
    classification = influencer_result['classification']
    match_score = influencer_result['match_score']
    followers = influencer_result['followers']
    bio = influencer_result.get('bio', '')
    image_analysis = influencer_result.get('image_analysis', {})

    brand_name = brand_data.get('brand_name', '브랜드')
    aesthetic_style = brand_data.get('aesthetic_style', '')
    product_type = brand_data.get('product_type', '')

    # 매칭 점수
    match_percent = int(match_score * 100)

    # 1. 인플루언서 스타일/성격 설명 생성
    style_desc = _generate_style_description(bio, image_analysis, classification)

    # 2. 콘텐츠 특성 설명
    content_desc = _generate_content_description(bio, image_analysis, classification)

    # 3. 팔로워 규모 설명
    follower_desc = _format_follower_count(followers)

    # 4. 브랜드 적합성 설명
    fit_desc = _generate_brand_fit_description(brand_name, aesthetic_style, product_type, image_analysis, match_percent)

    # 최종 추천 사유 조합
    reason = f"@{username}은 {style_desc}. {content_desc} {fit_desc} ({follower_desc})"

    return reason


def _generate_style_description(bio: str, image_analysis: Dict, classification: str) -> str:
    """인플루언서의 스타일과 성격을 설명하는 문구 생성"""

    # image_analysis에서 vibe 추출
    vibe = image_analysis.get('vibe', '')
    dominant_style = image_analysis.get('dominant_style', '')
    aesthetic_tags = image_analysis.get('aesthetic_tags', [])

    # vibe가 있으면 그대로 활용
    if vibe:
        return vibe

    # bio에서 핵심 정보 추출
    if bio:
        # 전문가 관련 키워드
        if any(kw in bio for kw in ['원장', '디렉터', '전문', '경력', '자격증']):
            experience_match = None
            if '년' in bio:
                import re
                exp_match = re.search(r'(\d+)년', bio)
                if exp_match:
                    experience_match = exp_match.group(1) + '년차'

            specialty = []
            if '염색' in bio: specialty.append('염색')
            if '펌' in bio: specialty.append('펌')
            if '탈모' in bio or '두피' in bio: specialty.append('두피케어')
            if '발레아쥬' in bio: specialty.append('발레아쥬')

            spec_text = '/'.join(specialty[:2]) if specialty else '헤어 시술'
            exp_text = f" {experience_match}" if experience_match else ""

            if '청담' in bio or '압구정' in bio or '프리미엄' in bio:
                return f"청담/압구정에서 활동하는{exp_text} {spec_text} 전문 헤어 아티스트"
            elif '홍대' in bio:
                return f"홍대에서 활동하는{exp_text} {spec_text} 전문 미용사"
            else:
                return f"{exp_text} {spec_text} 전문가로서 기술적인 노하우를 공유하는 헤어 크리에이터"

        # 셀프케어/리뷰어 관련
        if any(kw in bio for kw in ['리뷰', '홈케어', '셀프']):
            if '가성비' in bio or '저렴' in bio:
                return "가성비 좋은 제품을 발굴하여 솔직하게 리뷰하는 실용주의 뷰티 크리에이터"
            elif '비건' in bio or '친환경' in bio or '자연' in bio:
                return "친환경적이고 자연스러운 뷰티 라이프스타일을 추구하는 헤어케어 전문 크리에이터"
            else:
                return "집에서 할 수 있는 셀프 헤어케어 팁을 공유하는 실용적인 뷰티 크리에이터"

        # 육아맘
        if '육아' in bio or '엄마' in bio:
            return "바쁜 일상 속에서도 간편한 헤어 스타일링 팁을 공유하는 육아맘 뷰티 크리에이터"

        # K-Beauty
        if 'k-beauty' in bio.lower() or '해외' in bio:
            return "한국 헤어 트렌드를 글로벌하게 소개하는 K-Beauty 전문 크리에이터"

        # 트렌드세터
        if '트렌드' in bio or '패션' in bio:
            return "최신 헤어 트렌드를 선도하며 스타일링 콘텐츠를 제작하는 패션 인플루언서"

    # aesthetic_tags 기반 설명
    if aesthetic_tags:
        style_map = {
            '럭셔리': '고급스럽고 세련된 감성의',
            '글램': '화려하고 글래머러스한 스타일의',
            '미니멀': '정제되고 심플한 미니멀 스타일의',
            '스트릿': '힙하고 자유로운 스트릿 감성의',
            'Y2K': 'Y2K 레트로 무드를 즐기는',
            '빈티지': '빈티지한 감성과 레트로 무드를 사랑하는',
            '내추럴': '자연스럽고 편안한 분위기의',
            '로맨틱': '청순하고 여성스러운 로맨틱 스타일의',
            '엣지': '대담하고 개성있는 엣지 스타일의',
            '클래식': '클래식하고 우아한 정통 스타일의',
            '시크': '도시적이고 세련된 시크 무드의',
            '컬러풀': '화려한 컬러감으로 개성을 표현하는',
            '힐링': '편안하고 힐링되는 분위기의',
        }

        for tag in aesthetic_tags[:3]:
            if tag in style_map:
                return f"{style_map[tag]} 라이프스타일 인플루언서"

    # dominant_style 기반 기본 설명
    style_default_map = {
        'luxury': '고급스럽고 세련된 감성을 가진 프리미엄 뷰티 인플루언서',
        'trendy': '최신 트렌드를 빠르게 포착하는 스타일 인플루언서',
        'natural': '자연스럽고 편안한 라이프스타일을 추구하는 인플루언서',
        'minimal': '깔끔하고 정제된 미니멀 스타일의 인플루언서',
        'colorful': '화려하고 개성있는 컬러풀 스타일의 인플루언서'
    }

    if dominant_style in style_default_map:
        return style_default_map[dominant_style]

    # 기본값
    if classification == 'Expert':
        return "전문적인 헤어 시술과 기술을 공유하는 헤어 전문가"
    else:
        return "트렌디한 스타일링 콘텐츠를 제작하는 패션 인플루언서"


def _generate_content_description(bio: str, image_analysis: Dict, classification: str) -> str:
    """인플루언서가 주로 제작하는 콘텐츠 유형 설명"""

    hair_style_tags = image_analysis.get('hair_style_tags', [])
    aesthetic_tags = image_analysis.get('aesthetic_tags', [])

    # 헤어 스타일 태그가 있으면 활용
    if hair_style_tags:
        hair_styles = ', '.join(hair_style_tags[:3])

        if classification == 'Expert':
            return f"주로 {hair_styles} 등의 시술 과정과 전문 노하우를 담은 교육적 콘텐츠를 제작합니다."
        else:
            return f"{hair_styles} 스타일을 활용한 데일리 룩과 스타일링 콘텐츠로 팔로워들의 큰 호응을 얻고 있습니다."

    # bio 기반 콘텐츠 유형 추론
    if bio:
        content_types = []
        if '시술' in bio: content_types.append('시술 과정')
        if '리뷰' in bio: content_types.append('제품 리뷰')
        if '팁' in bio or '꿀팁' in bio: content_types.append('실용 팁')
        if '튜토리얼' in bio or 'tutorial' in bio.lower(): content_types.append('튜토리얼')
        if '일상' in bio or 'daily' in bio.lower(): content_types.append('일상 브이로그')

        if content_types:
            return f"{', '.join(content_types[:2])} 콘텐츠를 주로 제작하며 높은 참여도를 보입니다."

    # aesthetic_tags 기반 콘텐츠 추론
    if aesthetic_tags:
        if any(tag in aesthetic_tags for tag in ['OOTD', '데일리', '캐주얼']):
            return "데일리 스타일링과 OOTD 콘텐츠로 일상적인 영감을 제공합니다."
        elif any(tag in aesthetic_tags for tag in ['파티', '글램', '럭셔리']):
            return "특별한 날을 위한 글래머러스한 스타일링 콘텐츠를 선보입니다."
        elif any(tag in aesthetic_tags for tag in ['힐링', '여행', '카페']):
            return "감성적인 라이프스타일과 자연스러운 뷰티 콘텐츠를 공유합니다."

    # 기본 설명
    if classification == 'Expert':
        return "전문적인 헤어 시술 과정과 케어 팁을 담은 콘텐츠로 신뢰를 얻고 있습니다."
    else:
        return "트렌디한 스타일링 콘텐츠로 MZ세대에게 높은 인기를 얻고 있습니다."


def _format_follower_count(followers: int) -> str:
    """팔로워 수를 읽기 쉬운 형태로 포맷"""
    if followers >= 100000:
        return f"팔로워 {followers/10000:.1f}만명"
    elif followers >= 10000:
        return f"팔로워 {followers/10000:.1f}만명"
    else:
        return f"팔로워 {followers:,}명"


def _generate_brand_fit_description(brand_name: str, aesthetic_style: str, product_type: str, image_analysis: Dict, match_percent: int) -> str:
    """브랜드와의 적합성 설명 생성"""

    dominant_style = image_analysis.get('dominant_style', '')
    aesthetic_tags = image_analysis.get('aesthetic_tags', [])

    # 스타일 매칭 설명
    style_match_reasons = []

    # 럭셔리 브랜드 매칭
    if aesthetic_style and 'luxury' in aesthetic_style.lower():
        if dominant_style == 'luxury' or any(tag in aesthetic_tags for tag in ['럭셔리', '하이엔드', '명품']):
            style_match_reasons.append("고급스러운 이미지")

    # 트렌디 브랜드 매칭
    if aesthetic_style and 'trendy' in aesthetic_style.lower():
        if dominant_style == 'trendy' or any(tag in aesthetic_tags for tag in ['트렌디', 'Y2K', '스트릿']):
            style_match_reasons.append("트렌디한 감성")

    # 내추럴 브랜드 매칭
    if aesthetic_style and 'natural' in aesthetic_style.lower():
        if dominant_style == 'natural' or any(tag in aesthetic_tags for tag in ['내추럴', '친환경', '힐링']):
            style_match_reasons.append("자연스러운 이미지")

    # 제품 유형별 적합성
    product_fit = ""
    if product_type:
        if '샴푸' in product_type or '케어' in product_type:
            product_fit = f"{product_type} 제품 홍보"
        elif '염색' in product_type or '컬러' in product_type:
            product_fit = f"{product_type} 마케팅"
        elif '스타일링' in product_type:
            product_fit = f"스타일링 제품 협업"
        else:
            product_fit = f"{product_type} 캠페인"

    # 최종 조합
    if style_match_reasons and product_fit:
        return f"{', '.join(style_match_reasons)}가 '{brand_name}' 브랜드와 {match_percent}% 일치하여 {product_fit}에 최적입니다."
    elif product_fit:
        return f"'{brand_name}' 브랜드 이미지와 {match_percent}% 부합하며 {product_fit}에 적합합니다."
    else:
        return f"'{brand_name}'이 추구하는 무드와 {match_percent}% 일치하여 브랜드 캠페인에 효과적입니다."


def get_full_recommendations(
    brand_data: Dict,
    influencers: List[Dict],
    top_k: int = 5,
    min_fis: float = 60.0,
    expert_count: int = None,
    trendsetter_count: int = None
) -> Dict:
    """
    전체 추천 결과를 생성합니다.

    Args:
        brand_data: 브랜드 데이터
        influencers: 인플루언서 리스트
        top_k: 반환할 상위 K명
        min_fis: 최소 FIS 점수
        expert_count: 전문가 추천 수 (None이면 균등 분배)
        trendsetter_count: 트렌드세터 추천 수 (None이면 균등 분배)

    Returns:
        전체 추천 결과
    """
    # 매칭 수행 (다양성 보장 없이 전체 후보 확보)
    matched = match_influencers(brand_data, influencers, len(influencers), min_fis, ensure_diversity=False)

    # 브랜드 분석
    brand_analysis = analyze_brand(brand_data)

    # 타입별 분리 및 점수순 정렬
    experts = sorted([m for m in matched if m['classification'] == 'Expert'],
                     key=lambda x: x['match_score'], reverse=True)
    trendsetters = sorted([m for m in matched if m['classification'] == 'Trendsetter'],
                          key=lambda x: x['match_score'], reverse=True)

    # 요청된 수만큼 선택
    if expert_count is not None and trendsetter_count is not None:
        selected_experts = experts[:expert_count]
        selected_trendsetters = trendsetters[:trendsetter_count]
    else:
        # 기존 방식: 균등 분배
        half = top_k // 2
        selected_experts = experts[:half]
        selected_trendsetters = trendsetters[:top_k - half]

    # 합치고 매칭 점수로 정렬
    final_matched = selected_experts + selected_trendsetters
    final_matched.sort(key=lambda x: x['match_score'], reverse=True)

    # 추천 사유 생성
    recommendations = []
    for i, result in enumerate(final_matched, 1):
        reason = generate_recommendation_reason(brand_data, result)
        # match_score를 0-100 범위로 클램핑 (100% 초과 방지)
        match_score_percent = min(100.0, max(0.0, round(result['match_score'] * 100, 1)))
        recommendations.append({
            'rank': i,
            'username': result['username'],
            'followers': result['followers'],
            'type': result['classification'],
            'match_score': match_score_percent,
            'fis_score': result.get('fis_score', 0),
            'reason': reason,
            'details': {
                'confidence': result['confidence'],
                'similarity': result['similarity'],
                'bio': result['bio'],
                'fis_verdict': result.get('fis_verdict', '신뢰 계정')
            }
        })

    return {
        'brand_info': {
            'name': brand_data.get('brand_name', ''),
            'style': brand_analysis['style_analysis']['detected_style'],
            'product': brand_data.get('product_type', ''),
            'recommended_type': brand_analysis['recommendation']['influencer_type']
        },
        'total_analyzed': len(influencers),
        'total_passed_fis': len(matched),
        'recommendations': recommendations
    }


# 테스트용 코드
if __name__ == "__main__":
    import os

    # 테스트 데이터 로드
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'influencers_data.json')

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    influencers = data['influencers']

    # 테스트 브랜드
    test_brand = {
        "brand_name": "설화수",
        "slogan": "시간의 지혜를 담다",
        "core_values": ["전통", "고급스러움", "한방"],
        "target_audience": "30-50대 여성, 프리미엄 뷰티 소비자",
        "product_type": "헤어 에센스",
        "aesthetic_style": "Luxury"
    }

    # 매칭 수행
    results = get_full_recommendations(test_brand, influencers, top_k=5)

    print(f"\n=== {results['brand_info']['name']} 인플루언서 추천 ===")
    print(f"분석 대상: {results['total_analyzed']}명")
    print(f"FIS 통과: {results['total_passed_fis']}명")
    print(f"추천 타입: {results['brand_info']['recommended_type']}")

    print("\n--- 추천 리스트 ---")
    for rec in results['recommendations']:
        print(f"\n{rec['rank']}. @{rec['username']}")
        print(f"   팔로워: {rec['followers']:,}")
        print(f"   타입: {rec['type']}")
        print(f"   매칭: {rec['match_score']}%")
        print(f"   사유: {rec['reason'][:100]}...")
