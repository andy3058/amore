"""
FIS Engine - Fake Integrity Score 허수 필터링 모듈
봇 계정 및 어뷰징 계정 탐지를 위한 통계적 검증 모델
"""

from typing import Dict, List, Tuple
import math
from collections import Counter


def calculate_view_variability_score(posts: List[Dict]) -> Tuple[float, Dict]:
    """
    조회수 변동성 점수 (V_score) 계산
    표준편차/평균 (CV - Coefficient of Variation)
    변동계수가 0.1 미만이면 뷰봇 구매 의심

    Args:
        posts: 게시물 리스트

    Returns:
        (점수 0-100, 상세 분석)
    """
    views = [post.get('views', 0) for post in posts if post.get('views', 0) > 0]

    if len(views) < 2:
        return 50.0, {'status': 'insufficient_data', 'views': views}

    mean_views = sum(views) / len(views)
    if mean_views == 0:
        return 0.0, {'status': 'zero_mean', 'views': views}

    # 표준편차 계산
    variance = sum((v - mean_views) ** 2 for v in views) / len(views)
    std_dev = math.sqrt(variance)

    # 변동계수 (CV)
    cv = std_dev / mean_views

    # 점수 변환 (CV가 0.05 미만이면 의심, 0.08~0.5가 정상)
    if cv < 0.03:
        score = 30.0  # 매우 의심 (거의 동일한 조회수)
    elif cv < 0.05:
        score = 55.0  # 의심
    elif cv < 0.08:
        score = 75.0  # 약간 균일하지만 허용
    elif cv < 0.50:
        score = 95.0  # 정상
    else:
        score = 80.0  # 변동이 크지만 정상 범위

    analysis = {
        'mean_views': round(mean_views, 2),
        'std_dev': round(std_dev, 2),
        'cv': round(cv, 4),
        'views': views,
        'status': 'normal' if 0.10 <= cv <= 0.50 else 'suspicious'
    }

    return score, analysis


def calculate_engagement_asymmetry_score(posts: List[Dict]) -> Tuple[float, Dict]:
    """
    참여 비대칭성 점수 (A_score) 계산
    좋아요/조회수 비율이 2%~15% 범위 이탈 시 이상치

    Args:
        posts: 게시물 리스트

    Returns:
        (점수 0-100, 상세 분석)
    """
    ratios = []
    for post in posts:
        views = post.get('views', 0)
        likes = post.get('likes', 0)
        if views > 0:
            ratio = likes / views
            ratios.append(ratio)

    if len(ratios) == 0:
        return 50.0, {'status': 'insufficient_data'}

    avg_ratio = sum(ratios) / len(ratios)

    # 정상 범위: 2% ~ 15%
    if 0.02 <= avg_ratio <= 0.15:
        score = 90.0
        status = 'normal'
    elif 0.01 <= avg_ratio < 0.02:
        score = 70.0
        status = 'low_engagement'
    elif 0.15 < avg_ratio <= 0.25:
        score = 70.0
        status = 'high_engagement'
    elif avg_ratio < 0.01:
        score = 30.0  # 좋아요가 너무 적음 (뷰봇 의심)
        status = 'suspicious_low'
    else:
        score = 40.0  # 좋아요가 너무 많음 (좋아요 구매 의심)
        status = 'suspicious_high'

    analysis = {
        'avg_ratio': round(avg_ratio * 100, 2),
        'ratios': [round(r * 100, 2) for r in ratios],
        'status': status
    }

    return score, analysis


def calculate_comment_entropy_score(posts: List[Dict]) -> Tuple[float, Dict]:
    """
    댓글 엔트로피 점수 (E_score) 계산
    섀넌 엔트로피로 어휘 다양성 측정

    Note: 실제 구현에서는 댓글 내용이 필요하지만,
    여기서는 댓글 수와 참여율로 간접 측정

    Args:
        posts: 게시물 리스트

    Returns:
        (점수 0-100, 상세 분석)
    """
    comment_counts = [post.get('comments', 0) for post in posts]
    view_counts = [post.get('views', 0) for post in posts if post.get('views', 0) > 0]

    if len(comment_counts) == 0 or len(view_counts) == 0:
        return 50.0, {'status': 'insufficient_data'}

    # 댓글/조회수 비율
    comment_ratios = []
    for post in posts:
        views = post.get('views', 0)
        comments = post.get('comments', 0)
        if views > 0:
            comment_ratios.append(comments / views)

    if len(comment_ratios) == 0:
        return 50.0, {'status': 'no_ratio_data'}

    avg_comment_ratio = sum(comment_ratios) / len(comment_ratios)

    # 정상 댓글 비율: 0.1% ~ 2%
    if 0.001 <= avg_comment_ratio <= 0.02:
        score = 90.0
        status = 'normal'
    elif 0.0005 <= avg_comment_ratio < 0.001:
        score = 70.0
        status = 'low_comments'
    elif 0.02 < avg_comment_ratio <= 0.05:
        score = 75.0
        status = 'high_comments'
    elif avg_comment_ratio < 0.0005:
        score = 40.0  # 댓글이 너무 적음
        status = 'suspicious_low'
    else:
        score = 50.0  # 댓글이 너무 많음 (봇 댓글 의심)
        status = 'suspicious_high'

    # 댓글 수의 변동성 체크 (너무 균일하면 의심)
    if len(comment_counts) >= 2:
        mean_comments = sum(comment_counts) / len(comment_counts)
        if mean_comments > 0:
            variance = sum((c - mean_comments) ** 2 for c in comment_counts) / len(comment_counts)
            cv = math.sqrt(variance) / mean_comments
            if cv < 0.1:
                score -= 20  # 댓글 수가 너무 균일함
                status = 'suspicious_uniform'

    analysis = {
        'avg_comment_ratio': round(avg_comment_ratio * 100, 3),
        'comment_counts': comment_counts,
        'status': status
    }

    return max(0, score), analysis


def calculate_activity_stability_score(influencer_data: Dict) -> Tuple[float, Dict]:
    """
    활동 안정성 점수 (ACS_score) 계산
    업로드 간격의 표준편차 및 극단값 분석

    Args:
        influencer_data: 인플루언서 데이터

    Returns:
        (점수 0-100, 상세 분석)
    """
    avg_interval = influencer_data.get('avg_upload_interval_days', 0)

    if avg_interval == 0:
        return 50.0, {'status': 'no_interval_data'}

    # 정상 업로드 간격: 1~7일
    if 1 <= avg_interval <= 7:
        score = 90.0
        status = 'normal'
    elif 0.5 <= avg_interval < 1:
        score = 75.0  # 너무 자주 올림
        status = 'very_frequent'
    elif 7 < avg_interval <= 14:
        score = 80.0
        status = 'moderate'
    elif avg_interval < 0.5:
        score = 40.0  # 비정상적으로 자주 올림 (봇 의심)
        status = 'suspicious_frequent'
    else:
        score = 60.0  # 너무 뜸함
        status = 'infrequent'

    analysis = {
        'avg_interval_days': avg_interval,
        'status': status
    }

    return score, analysis


def calculate_geographic_consistency_score(influencer_data: Dict) -> Tuple[float, Dict]:
    """
    지리적 정합성 점수 (D_score) 계산
    콘텐츠 언어와 팔로워 국가 일치도

    Args:
        influencer_data: 인플루언서 데이터

    Returns:
        (점수 0-100, 상세 분석)
    """
    audience = influencer_data.get('audience_countries', {})
    bio = influencer_data.get('bio', '')

    if not audience:
        return 80.0, {'status': 'no_audience_data'}

    # 한국 팔로워 비율
    kr_ratio = audience.get('KR', 0)

    # Bio가 한국어인지 확인 (간단한 휴리스틱)
    has_korean = any('\uac00' <= char <= '\ud7a3' for char in bio)

    # 캡션에서 한국어 체크 (릴스 인플루언서는 bio가 없거나 짧음)
    posts = influencer_data.get('recent_posts', [])
    has_korean_content = has_korean
    for post in posts:
        caption = post.get('caption', '')
        if any('\uac00' <= char <= '\ud7a3' for char in caption):
            has_korean_content = True
            break

    # 한국 타겟 인플루언서 판정 (bio 또는 캡션에 한국어가 있거나, 한국 팔로워가 50% 이상)
    is_korean_target = has_korean_content or kr_ratio >= 0.50

    if is_korean_target:
        # 한국 타겟 인플루언서
        if kr_ratio >= 0.70:
            score = 95.0
            status = 'excellent'
        elif kr_ratio >= 0.50:
            score = 90.0
            status = 'good'
        elif kr_ratio >= 0.35:
            score = 80.0
            status = 'moderate'
        else:
            score = 65.0  # 해외 팔로워가 많지만 의심까지는 아님
            status = 'international_mix'
    else:
        # 글로벌 인플루언서 (bio가 영어이거나 없음)
        if kr_ratio >= 0.30:
            score = 90.0
            status = 'global_with_kr'
        else:
            score = 75.0
            status = 'global'

    analysis = {
        'kr_ratio': kr_ratio,
        'has_korean_bio': has_korean,
        'has_korean_content': has_korean_content,
        'audience_distribution': audience,
        'status': status
    }

    return score, analysis


def calculate_fis_score(influencer_data: Dict) -> Dict:
    """
    최종 FIS (Fake Integrity Score) 계산
    FIS = (w1×V + w2×E + w3×A + w4×(100-ACS)) × D_score/100

    Args:
        influencer_data: 인플루언서 데이터

    Returns:
        FIS 점수 및 상세 분석
    """
    posts = influencer_data.get('recent_posts', [])

    # 각 지표 계산
    v_score, v_analysis = calculate_view_variability_score(posts)
    e_score, e_analysis = calculate_comment_entropy_score(posts)
    a_score, a_analysis = calculate_engagement_asymmetry_score(posts)
    acs_score, acs_analysis = calculate_activity_stability_score(influencer_data)
    d_score, d_analysis = calculate_geographic_consistency_score(influencer_data)

    # 가중치 설정
    w1 = 0.25  # 조회수 변동성
    w2 = 0.20  # 댓글 엔트로피
    w3 = 0.25  # 참여 비대칭성
    w4 = 0.15  # 활동 안정성
    w5 = 0.15  # 지리적 정합성 (별도 적용)

    # 기본 점수 계산
    base_score = (w1 * v_score + w2 * e_score + w3 * a_score + w4 * acs_score)

    # 지리적 정합성 반영
    final_score = base_score * (d_score / 100) + (w5 * d_score)

    # 0-100 범위로 정규화
    final_score = max(0, min(100, final_score))

    # 판정
    if final_score >= 80:
        verdict = '신뢰 계정'
        action = '추천 대상'
    elif final_score >= 60:
        verdict = '주의 필요'
        action = '수동 검토 후 결정'
    else:
        verdict = '허수 의심'
        action = '자동 제외'

    return {
        'username': influencer_data.get('username', ''),
        'fis_score': round(final_score, 1),
        'verdict': verdict,
        'action': action,
        'breakdown': {
            'view_variability': {'score': round(v_score, 1), 'weight': w1, 'analysis': v_analysis},
            'comment_entropy': {'score': round(e_score, 1), 'weight': w2, 'analysis': e_analysis},
            'engagement_asymmetry': {'score': round(a_score, 1), 'weight': w3, 'analysis': a_analysis},
            'activity_stability': {'score': round(acs_score, 1), 'weight': w4, 'analysis': acs_analysis},
            'geographic_consistency': {'score': round(d_score, 1), 'weight': w5, 'analysis': d_analysis}
        }
    }


def batch_calculate_fis(influencers: List[Dict]) -> List[Dict]:
    """
    여러 인플루언서의 FIS 점수를 일괄 계산합니다.

    Args:
        influencers: 인플루언서 데이터 리스트

    Returns:
        FIS 결과 리스트
    """
    results = []
    for influencer in influencers:
        result = calculate_fis_score(influencer)
        results.append(result)
    return results


def filter_by_fis(influencers: List[Dict], min_score: float = 60.0) -> List[Dict]:
    """
    FIS 점수로 인플루언서를 필터링합니다.

    Args:
        influencers: 인플루언서 데이터 리스트
        min_score: 최소 FIS 점수

    Returns:
        필터링된 인플루언서 리스트와 FIS 결과
    """
    filtered = []
    for influencer in influencers:
        fis_result = calculate_fis_score(influencer)
        if fis_result['fis_score'] >= min_score:
            influencer['fis_result'] = fis_result
            filtered.append(influencer)
    return filtered


# 테스트용 코드
if __name__ == "__main__":
    # 정상 계정 테스트
    normal_account = {
        "username": "normal_test",
        "bio": "헤어 전문가 | 살롱 원장",
        "recent_posts": [
            {"views": 45000, "likes": 3200, "comments": 89},
            {"views": 38000, "likes": 2800, "comments": 72},
            {"views": 32000, "likes": 2100, "comments": 56}
        ],
        "audience_countries": {"KR": 0.92, "US": 0.03, "JP": 0.02, "OTHER": 0.03},
        "avg_upload_interval_days": 3.2
    }

    # 의심 계정 테스트
    suspicious_account = {
        "username": "fake_test",
        "bio": "Hair | Beauty",
        "recent_posts": [
            {"views": 150000, "likes": 15000, "comments": 45},
            {"views": 149000, "likes": 14800, "comments": 42},
            {"views": 151000, "likes": 15200, "comments": 48}
        ],
        "audience_countries": {"KR": 0.25, "IN": 0.30, "BR": 0.20, "OTHER": 0.25},
        "avg_upload_interval_days": 1.0
    }

    print("=== 정상 계정 ===")
    result = calculate_fis_score(normal_account)
    print(f"FIS Score: {result['fis_score']}")
    print(f"Verdict: {result['verdict']}")
    print(f"Action: {result['action']}")

    print("\n=== 의심 계정 ===")
    result = calculate_fis_score(suspicious_account)
    print(f"FIS Score: {result['fis_score']}")
    print(f"Verdict: {result['verdict']}")
    print(f"Action: {result['action']}")
