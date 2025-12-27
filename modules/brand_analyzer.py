"""
Brand Analyzer Module - 브랜드/제품 분석 모듈
브랜드 페르소나 벡터화 및 특성 추출
"""

from typing import Dict, List, Tuple, Optional
import math


# 미적 스타일 정의
AESTHETIC_STYLES = {
    'Luxury': {
        'keywords': ['프리미엄', '럭셔리', '고급', '명품', 'VIP', '하이엔드', '프레스티지'],
        'score': 1.0
    },
    'Natural': {
        'keywords': ['자연', '내추럴', '유기농', '비건', '친환경', '순수', '클린뷰티'],
        'score': 0.7
    },
    'Trendy': {
        'keywords': ['트렌디', '힙', '모던', '영', '젊은', 'MZ', '감각적'],
        'score': 0.5
    },
    'Classic': {
        'keywords': ['클래식', '전통', '한방', '시간', '역사', '헤리티지'],
        'score': 0.8
    },
    'Mass': {
        'keywords': ['대중적', '가성비', '데일리', '일상', '실용', '합리적'],
        'score': 0.3
    },
    'Minimal': {
        'keywords': ['미니멀', '심플', '깔끔', '베이직', '에센셜'],
        'score': 0.6
    },
    'Colorful': {
        'keywords': ['컬러풀', '화려', '비비드', '팝', '펀', '플레이풀'],
        'score': 0.4
    }
}

# 제품 유형별 전문성 지수
PRODUCT_PROFESSIONAL_INDEX = {
    # === 전문가용 (살롱/미용사 타겟) ===
    # 염색/컬러
    '염색약': 0.9,
    '산화제': 0.95,
    '탈색제': 0.9,
    '컬러차트': 0.85,
    '염색볼/브러시': 0.8,

    # 펌/웨이브
    '펌제': 0.95,
    '펌로드': 0.9,
    '중화제': 0.9,
    '셋팅제': 0.75,

    # 클리닉/케어
    '클리닉': 0.9,
    '두피케어': 0.8,
    '두피스케일러': 0.75,
    '두피앰플': 0.8,
    '모발클리닉': 0.85,
    '손상모케어': 0.7,
    '단백질트리트먼트': 0.75,

    # 살롱 전용
    '전문기기': 1.0,
    '살롱전용': 1.0,
    '미용가위': 0.95,
    '바리깡': 0.85,
    '헤어아이롱(전문가용)': 0.85,

    # === 일반 소비자용 (홈케어) ===
    # 세정
    '샴푸': 0.3,
    '린스': 0.25,
    '컨디셔너': 0.25,
    '클렌징폼': 0.25,
    '드라이샴푸': 0.3,
    '쿨링샴푸': 0.3,
    '약산성샴푸': 0.35,

    # 트리트먼트/마스크
    '트리트먼트': 0.4,
    '헤어팩': 0.4,
    '헤어마스크': 0.45,
    '리브인트리트먼트': 0.35,
    '나이트케어': 0.35,

    # 에센스/오일/세럼
    '헤어에센스': 0.4,
    '헤어오일': 0.35,
    '헤어세럼': 0.4,
    '헤어앰플': 0.45,
    '헤어미스트': 0.3,
    '헤어워터': 0.3,
    '헤어밤': 0.35,
    '헤어버터': 0.35,

    # 스타일링
    '헤어스프레이': 0.3,
    '왁스': 0.35,
    '무스': 0.3,
    '젤': 0.3,
    '포마드': 0.35,
    '헤어크림': 0.3,
    '볼륨파우더': 0.3,
    '헤어쿠션': 0.3,
    '새치커버': 0.4,

    # 열기구
    '고데기': 0.4,
    '드라이기': 0.35,
    '헤어롤': 0.3,
    '매직기': 0.4,
    '열보호제': 0.35,

    # 염색 (홈케어)
    '셀프염색': 0.45,
    '염색샴푸': 0.4,
    '컬러트리트먼트': 0.45,
    '뿌리염색': 0.45,
    '헤어틴트': 0.4,
    '헤어초크': 0.3,
    '헤어마스카라': 0.35,

    # 탈모/두피 (홈케어)
    '탈모샴푸': 0.55,
    '탈모케어': 0.55,
    '두피토닉': 0.5,
    '두피세럼': 0.5,
    '탈모앰플': 0.55,
    '육모제': 0.6,

    # 악세서리
    '헤어브러시': 0.25,
    '빗': 0.2,
    '헤어밴드': 0.2,
    '헤어핀': 0.2,
    '헤어캡': 0.2,

    # === 기타 ===
    '기타': 0.5
}

# 제품 유형 카테고리 (UI 표시용)
PRODUCT_CATEGORIES = {
    '세정': ['샴푸', '린스', '컨디셔너', '드라이샴푸', '쿨링샴푸', '약산성샴푸'],
    '트리트먼트': ['트리트먼트', '헤어팩', '헤어마스크', '리브인트리트먼트', '나이트케어', '단백질트리트먼트'],
    '에센스/오일': ['헤어에센스', '헤어오일', '헤어세럼', '헤어앰플', '헤어미스트', '헤어워터', '헤어밤'],
    '스타일링': ['헤어스프레이', '왁스', '무스', '젤', '포마드', '헤어크림', '볼륨파우더'],
    '염색(홈케어)': ['셀프염색', '염색샴푸', '컬러트리트먼트', '뿌리염색', '헤어틴트', '새치커버'],
    '염색(전문가용)': ['염색약', '산화제', '탈색제', '컬러차트'],
    '펌(전문가용)': ['펌제', '펌로드', '중화제', '셋팅제'],
    '두피/탈모케어': ['두피케어', '두피토닉', '두피세럼', '탈모샴푸', '탈모케어', '탈모앰플', '육모제'],
    '열기구': ['고데기', '드라이기', '매직기', '열보호제'],
    '클리닉(전문가용)': ['클리닉', '모발클리닉', '손상모케어', '두피앰플'],
    '기타': ['기타']
}

# 타겟 고객층 분석
TARGET_AUDIENCE_MAPPING = {
    '전문가': {'professional_weight': 0.9, 'age_range': '25-55'},
    '미용사': {'professional_weight': 1.0, 'age_range': '20-50'},
    '살롱': {'professional_weight': 0.95, 'age_range': '25-50'},
    '20대': {'professional_weight': 0.3, 'age_range': '20-29'},
    '30대': {'professional_weight': 0.4, 'age_range': '30-39'},
    '40대': {'professional_weight': 0.5, 'age_range': '40-49'},
    '50대': {'professional_weight': 0.6, 'age_range': '50-59'},
    '여성': {'professional_weight': 0.4, 'age_range': 'all'},
    '남성': {'professional_weight': 0.45, 'age_range': 'all'},
    '프리미엄': {'professional_weight': 0.6, 'age_range': '30-55'},
    '대학생': {'professional_weight': 0.2, 'age_range': '18-25'},
    'MZ': {'professional_weight': 0.25, 'age_range': '18-35'},
    '일반': {'professional_weight': 0.35, 'age_range': 'all'}
}


def analyze_aesthetic_style(brand_data: Dict) -> Dict:
    """
    브랜드의 미적 스타일을 분석합니다.

    Args:
        brand_data: 브랜드 데이터

    Returns:
        스타일 분석 결과
    """
    aesthetic_style = brand_data.get('aesthetic_style', '')
    slogan = brand_data.get('slogan', '')
    core_values = brand_data.get('core_values', [])

    # 전체 텍스트 결합
    full_text = f"{aesthetic_style} {slogan} {' '.join(core_values)}"

    style_scores = {}
    detected_style = None
    max_score = 0

    for style_name, style_info in AESTHETIC_STYLES.items():
        # 명시적 스타일 매칭
        if style_name.lower() == aesthetic_style.lower():
            style_scores[style_name] = 1.0
            detected_style = style_name
            max_score = 1.0
        else:
            # 키워드 기반 매칭
            keywords = style_info['keywords']
            match_count = sum(1 for kw in keywords if kw in full_text)
            score = match_count / len(keywords) if keywords else 0

            style_scores[style_name] = round(score, 3)

            if score > max_score:
                max_score = score
                detected_style = style_name

    # 기본 스타일이 없으면 가장 높은 점수의 스타일 사용
    if detected_style is None:
        detected_style = 'Mass'  # 기본값

    luxury_score = style_scores.get('Luxury', 0)
    if detected_style == 'Luxury':
        luxury_score = max(luxury_score, 0.8)

    return {
        'detected_style': detected_style,
        'style_scores': style_scores,
        'luxury_score': luxury_score,
        'is_luxury': luxury_score >= 0.5
    }


def analyze_product_type(brand_data: Dict) -> Dict:
    """
    제품 유형을 분석하여 전문성 지수를 산출합니다.

    Args:
        brand_data: 브랜드 데이터

    Returns:
        제품 분석 결과
    """
    product_type = brand_data.get('product_type', '')

    # 정확한 매칭 시도
    professional_index = PRODUCT_PROFESSIONAL_INDEX.get(product_type, None)

    if professional_index is None:
        # 부분 매칭 시도
        for prod_name, index in PRODUCT_PROFESSIONAL_INDEX.items():
            if prod_name in product_type or product_type in prod_name:
                professional_index = index
                break

    if professional_index is None:
        professional_index = 0.5  # 기본값

    # Expert/Trendsetter 선호도 결정
    if professional_index >= 0.7:
        preferred_type = 'Expert'
        type_weight = professional_index
    elif professional_index <= 0.4:
        preferred_type = 'Trendsetter'
        type_weight = 1 - professional_index
    else:
        preferred_type = 'Both'
        type_weight = 0.5

    return {
        'product_type': product_type,
        'professional_index': professional_index,
        'preferred_influencer_type': preferred_type,
        'type_weight': type_weight
    }


def analyze_target_audience(brand_data: Dict) -> Dict:
    """
    타겟 고객층을 분석합니다.

    Args:
        brand_data: 브랜드 데이터

    Returns:
        타겟 고객층 분석 결과
    """
    target = brand_data.get('target_audience', '')

    detected_segments = []
    total_weight = 0
    count = 0

    for segment, info in TARGET_AUDIENCE_MAPPING.items():
        if segment in target:
            detected_segments.append(segment)
            total_weight += info['professional_weight']
            count += 1

    if count == 0:
        avg_weight = 0.4  # 기본값
    else:
        avg_weight = total_weight / count

    return {
        'target_audience': target,
        'detected_segments': detected_segments,
        'professional_weight': avg_weight
    }


def analyze_campaign_style(brand_data: Dict) -> Dict:
    """
    캠페인 설명에서 추가 스타일 요소를 분석합니다.
    브랜드가 Luxury라도 제품/캠페인은 Trendy할 수 있음.

    Args:
        brand_data: 브랜드 데이터

    Returns:
        캠페인 스타일 분석 결과
    """
    campaign_desc = brand_data.get('campaign_description', '')

    # 캠페인 스타일 키워드
    trendy_keywords = ['트렌디', '힙', 'MZ', '젊은', 'Y2K', '스트릿', '캐주얼', '발랄']
    luxury_keywords = ['프리미엄', '고급', '럭셔리', 'VIP', '하이엔드']
    natural_keywords = ['자연', '내추럴', '순한', '친환경', '비건', '힐링']
    colorful_keywords = ['화려', '컬러풀', '비비드', '팝', '펀']
    minimal_keywords = ['미니멀', '심플', '깔끔', '모던']

    # 점수 계산
    trendy_score = sum(1 for kw in trendy_keywords if kw in campaign_desc) * 0.15
    luxury_score = sum(1 for kw in luxury_keywords if kw in campaign_desc) * 0.15
    natural_score = sum(1 for kw in natural_keywords if kw in campaign_desc) * 0.15
    colorful_score = sum(1 for kw in colorful_keywords if kw in campaign_desc) * 0.15
    minimal_score = sum(1 for kw in minimal_keywords if kw in campaign_desc) * 0.15

    return {
        'trendy': min(1, trendy_score),
        'luxury': min(1, luxury_score),
        'natural': min(1, natural_score),
        'colorful': min(1, colorful_score),
        'minimal': min(1, minimal_score),
        'has_campaign_style': any([trendy_score, luxury_score, natural_score, colorful_score, minimal_score])
    }


def create_brand_vector(brand_data: Dict) -> List[float]:
    """
    브랜드 데이터를 벡터로 변환합니다.
    브랜드 고유 스타일과 캠페인/제품 스타일을 조합합니다.

    벡터 구성:
    [luxury_score, professional_index, expert_preference, trendsetter_preference,
     colorfulness, natural_score, modern_score]

    Args:
        brand_data: 브랜드 데이터

    Returns:
        브랜드 벡터 (정규화됨)
    """
    style_analysis = analyze_aesthetic_style(brand_data)
    product_analysis = analyze_product_type(brand_data)
    audience_analysis = analyze_target_audience(brand_data)
    campaign_style = analyze_campaign_style(brand_data)

    # 벡터 구성 요소 - 브랜드 기본 스타일
    base_luxury_score = style_analysis['luxury_score']
    professional_index = product_analysis['professional_index']

    # 캠페인 스타일이 있으면 조합 (브랜드 40% + 캠페인 60%)
    if campaign_style['has_campaign_style']:
        # 럭셔리 브랜드가 트렌디한 제품을 내는 경우
        luxury_score = base_luxury_score * 0.4 + campaign_style['luxury'] * 0.6
        if campaign_style['trendy'] > 0:
            # 트렌디한 캠페인이면 럭셔리 점수 일부 유지하면서 트렌디함 추가
            luxury_score = max(0.3, luxury_score * 0.7)
    else:
        luxury_score = base_luxury_score

    # Expert/Trendsetter 선호도 - 캠페인 스타일 고려
    if product_analysis['preferred_influencer_type'] == 'Expert':
        expert_pref = 0.8
        trend_pref = 0.2
    elif product_analysis['preferred_influencer_type'] == 'Trendsetter':
        expert_pref = 0.2
        trend_pref = 0.8
    else:
        expert_pref = 0.5
        trend_pref = 0.5

    # 캠페인이 트렌디하면 트렌드세터 선호도 높임
    if campaign_style['trendy'] > 0.3:
        trend_pref = min(1, trend_pref + campaign_style['trendy'] * 0.4)
        expert_pref = max(0.1, expert_pref - campaign_style['trendy'] * 0.2)

    # 스타일 점수들 - 브랜드 + 캠페인 조합
    base_colorfulness = style_analysis['style_scores'].get('Colorful', 0.3)
    base_natural_score = style_analysis['style_scores'].get('Natural', 0.3)
    base_modern_score = style_analysis['style_scores'].get('Trendy', 0.3)

    # 캠페인 스타일 반영
    if campaign_style['has_campaign_style']:
        colorfulness = max(base_colorfulness, campaign_style['colorful'])
        natural_score = max(base_natural_score, campaign_style['natural'])
        modern_score = max(base_modern_score, campaign_style['trendy'])
    else:
        colorfulness = base_colorfulness
        natural_score = base_natural_score
        modern_score = base_modern_score

    # 벡터 생성
    vector = [
        luxury_score,
        professional_index,
        expert_pref,
        trend_pref,
        colorfulness,
        natural_score,
        modern_score
    ]

    # L2 정규화
    magnitude = math.sqrt(sum(v ** 2 for v in vector))
    if magnitude > 0:
        vector = [v / magnitude for v in vector]

    return vector


def analyze_brand(brand_data: Dict) -> Dict:
    """
    브랜드 전체 분석을 수행합니다.

    Args:
        brand_data: 브랜드 데이터

    Returns:
        종합 분석 결과
    """
    style_analysis = analyze_aesthetic_style(brand_data)
    product_analysis = analyze_product_type(brand_data)
    audience_analysis = analyze_target_audience(brand_data)
    brand_vector = create_brand_vector(brand_data)

    # 추천 인플루언서 타입 결정
    if product_analysis['professional_index'] >= 0.7:
        recommended_type = 'Expert'
        type_reason = '전문 시술/제품으로 미용 전문가와의 협업이 효과적'
    elif product_analysis['professional_index'] <= 0.35:
        recommended_type = 'Trendsetter'
        type_reason = '일반 소비자 타겟 제품으로 트렌드세터와의 협업이 효과적'
    else:
        recommended_type = 'Both'
        type_reason = '전문가와 트렌드세터 모두와 협업 가능'

    return {
        'brand_name': brand_data.get('brand_name', ''),
        'brand_vector': brand_vector,
        'style_analysis': style_analysis,
        'product_analysis': product_analysis,
        'audience_analysis': audience_analysis,
        'recommendation': {
            'influencer_type': recommended_type,
            'reason': type_reason,
            'luxury_match': style_analysis['is_luxury'],
            'professional_match': product_analysis['professional_index'] >= 0.6
        }
    }


def get_matching_criteria(brand_analysis: Dict) -> Dict:
    """
    브랜드 분석 결과를 기반으로 매칭 기준을 생성합니다.

    Args:
        brand_analysis: 브랜드 분석 결과

    Returns:
        매칭 기준
    """
    style = brand_analysis['style_analysis']
    product = brand_analysis['product_analysis']
    rec = brand_analysis['recommendation']

    # 최소 FIS 점수 설정
    if style['is_luxury']:
        min_fis = 75  # 럭셔리 브랜드는 더 높은 기준
    else:
        min_fis = 60

    # 팔로워 범위 설정
    if style['is_luxury']:
        follower_range = (30000, 200000)  # 중형~대형
    elif product['professional_index'] >= 0.7:
        follower_range = (10000, 100000)  # 소형~중형 (전문가)
    else:
        follower_range = (50000, 500000)  # 중형~대형 (일반)

    return {
        'preferred_type': rec['influencer_type'],
        'min_fis_score': min_fis,
        'follower_range': follower_range,
        'luxury_preference': style['luxury_score'],
        'professional_preference': product['professional_index']
    }


# 테스트용 코드
if __name__ == "__main__":
    # 테스트 브랜드 데이터
    test_brand = {
        "brand_name": "설화수",
        "slogan": "시간의 지혜를 담다",
        "core_values": ["전통", "고급스러움", "한방"],
        "target_audience": "30-50대 여성, 프리미엄 뷰티 소비자",
        "product_type": "헤어 에센스",
        "aesthetic_style": "Luxury"
    }

    result = analyze_brand(test_brand)

    print(f"Brand: {result['brand_name']}")
    print(f"Brand Vector: {[round(v, 3) for v in result['brand_vector']]}")
    print(f"Detected Style: {result['style_analysis']['detected_style']}")
    print(f"Luxury Score: {result['style_analysis']['luxury_score']}")
    print(f"Professional Index: {result['product_analysis']['professional_index']}")
    print(f"Recommended Type: {result['recommendation']['influencer_type']}")
    print(f"Reason: {result['recommendation']['reason']}")

    criteria = get_matching_criteria(result)
    print(f"\nMatching Criteria:")
    print(f"  Min FIS: {criteria['min_fis_score']}")
    print(f"  Follower Range: {criteria['follower_range']}")
