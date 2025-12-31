"""
API 라우터 - 인플루언서 추천 API (RAG 기반)
==========================================

엔드포인트:
- GET  /brands: 브랜드 목록
- GET  /brands/{name}: 브랜드 상세
- POST /recommend: 인플루언서 추천 (RAG 기반)
- GET  /product-categories: 제품 카테고리
- GET  /influencers: 인플루언서 목록
- POST /rag/analyze: 인플루언서 분석 및 인덱싱
- GET  /rag/status: RAG 시스템 상태
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging

from config.products import PRODUCT_CATEGORIES

logger = logging.getLogger(__name__)

router = APIRouter()


# ============== Pydantic 모델 ==============

class RecommendRequest(BaseModel):
    """추천 요청 (RAG 기반)"""
    brand_name: str
    product_type: Optional[str] = None
    product_line: Optional[str] = None
    description: Optional[str] = None  # 캠페인 설명
    target_gender: Optional[str] = None  # female/male/unisex
    expert_count: Optional[int] = 2  # 전문가 추천 수
    trendsetter_count: Optional[int] = 3  # 트렌드세터 추천 수


# ============== 의존성 ==============

_influencers: List[Dict] = None
_brand_db: Dict = None
_rag_manager = None


def init_routes(brand_db: Dict, influencers: List[Dict]):
    """라우터 초기화"""
    global _influencers, _brand_db, _rag_manager

    _brand_db = brand_db
    _influencers = influencers

    # RAG 매니저 초기화
    from pipeline import RAG_AVAILABLE
    if RAG_AVAILABLE:
        from pipeline import InfluencerAnalysisManager
        _rag_manager = InfluencerAnalysisManager()
        logger.info("RAG 시스템 초기화 완료")


# ============== 브랜드 API ==============

@router.get("/brands")
async def get_brands():
    """브랜드 목록"""
    hair_brands = _brand_db.get("hair_brands", [])
    brands_detail = {}

    for name in hair_brands:
        info = _brand_db.get("brands", {}).get(name, {})
        brands_detail[name] = {
            "slogan": info.get("slogan", ""),
            "aesthetic_style": info.get("aesthetic_style", ""),
            "core_values": info.get("core_values", [])
        }

    return {
        "brands": list(_brand_db.get("brands", {}).keys()),
        "hair_brands": hair_brands,
        "brands_detail": brands_detail
    }


@router.get("/brands/{brand_name}")
async def get_brand_info(brand_name: str):
    """브랜드 상세 정보"""
    info = _brand_db.get("brands", {}).get(brand_name)
    if not info:
        raise HTTPException(status_code=404, detail="브랜드를 찾을 수 없습니다")
    return info


# ============== 추천 API (RAG 기반) ==============

@router.post("/recommend")
async def recommend_influencers(request: RecommendRequest):
    """
    인플루언서 추천 (RAG 기반)

    파이프라인:
    1. LLM Vision으로 인플루언서 이미지 분석 (사전 인덱싱)
    2. 브랜드 + 제품 + 캠페인 설명으로 쿼리 생성
    3. ChromaDB 벡터 유사도 검색 (전문가/트렌드세터 비율 적용)
    4. 성별 필터링 적용
    5. LLM으로 추천 사유 생성
    """
    from pipeline import RAG_AVAILABLE

    if not RAG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="RAG 시스템이 사용 불가능합니다. ChromaDB를 설치해주세요: pip install chromadb"
        )

    brand_info = _brand_db.get("brands", {}).get(request.brand_name, {})

    if not brand_info:
        raise HTTPException(
            status_code=404,
            detail=f"브랜드 '{request.brand_name}'을(를) 찾을 수 없습니다"
        )

    expert_count = request.expert_count or 0
    trendsetter_count = request.trendsetter_count or 0
    total_count = expert_count + trendsetter_count

    if total_count == 0:
        return {
            'brand_info': {'name': request.brand_name},
            'recommendations': [],
            'total_results': 0
        }

    # 검색 배수 설정 (필터링 손실을 고려하여 넉넉하게)
    SEARCH_MULTIPLIER = 5

    # 전문가 검색
    expert_results = []
    if expert_count > 0:
        expert_results = _rag_manager.search_influencers(
            brand_name=request.brand_name,
            product_type=request.product_type or request.product_line,
            campaign_description=request.description,
            target_gender=request.target_gender,
            influencer_type='expert',
            top_k=expert_count * SEARCH_MULTIPLIER
        )

    # 트렌드세터 검색
    trendsetter_results = []
    if trendsetter_count > 0:
        trendsetter_results = _rag_manager.search_influencers(
            brand_name=request.brand_name,
            product_type=request.product_type or request.product_line,
            campaign_description=request.description,
            target_gender=request.target_gender,
            influencer_type='trendsetter',
            top_k=trendsetter_count * SEARCH_MULTIPLIER
        )

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
        if not target_gender or target_gender == 'unisex':
            return results

        filtered = []
        for r in results:
            inf_gender = r.get('metadata', {}).get('target_gender', 'unisex')

            if strict:
                # 트렌드세터: 타겟과 정확히 일치하거나 unisex여야 함
                if inf_gender == target_gender or inf_gender == 'unisex':
                    filtered.append(r)
            else:
                # 전문가: 반대 성별 전용만 제외
                if target_gender == 'female' and inf_gender == 'male':
                    continue
                if target_gender == 'male' and inf_gender == 'female':
                    continue
                filtered.append(r)

        return filtered

    def extract_target_ages(description):
        """설명에서 타겟 연령대 추출 (예: 30,40대 -> [30, 40])"""
        if not description:
            return []
        import re
        ages = []
        # 30,40대 패턴
        match = re.search(r'(\d{2})[,\s]*(\d{2})대', description)
        if match:
            ages = [int(match.group(1)), int(match.group(2))]
        else:
            # 단일 연령대 패턴
            matches = re.findall(r'(\d{2})대', description)
            ages = [int(m) for m in matches]
        return ages

    def filter_trendsetter_by_age(results, description):
        """
        트렌드세터 연령대 필터링 (마케팅 관점)

        트렌드세터는 본인이 광고 모델이므로:
        - 타겟 연령대와 정확히 일치하는 인플루언서가 최우선
        - 인접 연령대(±10년)는 차선책
        - 연령대 정보 없는 경우는 최후순위

        반환: (정확일치, 인접연령, 미상) 순으로 정렬된 리스트
        """
        target_ages = extract_target_ages(description)
        if not target_ages:
            return results  # 타겟 연령 미지정시 필터 안함

        exact_match = []      # 정확히 일치 (30대 타겟 → 30대 인플루언서)
        adjacent_match = []   # 인접 연령대 (30대 타겟 → 20대, 40대)
        unknown_age = []      # 연령대 정보 없음

        for r in results:
            inf_age_str = r.get('metadata', {}).get('target_age', '')

            if not inf_age_str:
                unknown_age.append(r)
                continue

            import re
            inf_age_match = re.search(r'(\d{2})', inf_age_str)
            if not inf_age_match:
                unknown_age.append(r)
                continue

            inf_age = int(inf_age_match.group(1))

            # 정확히 일치
            if inf_age in target_ages:
                exact_match.append(r)
            # 인접 연령대 (±10년 이내)
            elif any(abs(inf_age - t) <= 10 for t in target_ages):
                adjacent_match.append(r)
            # 그 외는 제외 (너무 동떨어진 연령대)

        # 우선순위: 정확일치 > 인접연령 > 미상
        return exact_match + adjacent_match + unknown_age

    def ensure_count(results, target_count, fallback_results):
        """
        목표 인원수 보장
        부족하면 fallback_results에서 중복 없이 추가
        """
        if len(results) >= target_count:
            return results[:target_count]

        existing_usernames = {r['username'] for r in results}
        for r in fallback_results:
            if r['username'] not in existing_usernames:
                results.append(r)
                existing_usernames.add(r['username'])
                if len(results) >= target_count:
                    break

        return results[:target_count]

    # ============================================================
    # 전문가 필터링
    # ============================================================
    # - 연령대 필터 없음 (모든 연령 고객 시술 가능)
    # - 성별 필터도 느슨하게 (반대 성별 '전용'만 제외)
    expert_filtered = filter_by_gender(expert_results, request.target_gender, strict=False)
    expert_results = ensure_count(expert_filtered, expert_count, expert_results)

    # ============================================================
    # 트렌드세터 필터링 (엄격)
    # ============================================================
    # - 성별 필터 엄격 (타겟과 일치하거나 unisex만)
    # - 연령대 필터 적용 (정확일치 > 인접 > 미상 순)
    trendsetter_gender_filtered = filter_by_gender(
        trendsetter_results, request.target_gender, strict=True
    )
    trendsetter_age_filtered = filter_trendsetter_by_age(
        trendsetter_gender_filtered, request.description
    )
    trendsetter_results = ensure_count(
        trendsetter_age_filtered, trendsetter_count, trendsetter_gender_filtered
    )

    # 결과 병합 및 응답 구성
    all_results = expert_results + trendsetter_results
    product_type = request.product_type or request.product_line or '헤어케어'

    # 모든 결과를 리스트로 변환
    recommendations = []
    for r in all_results:
        username = r['username']
        influencer = next((inf for inf in _influencers if inf["username"] == username), None)
        metadata = r.get('metadata', {})
        inf_type = metadata.get('influencer_type', 'trendsetter')

        # 매칭 점수 스케일 조정 (Multi-Signal Hybrid Scoring 기반)
        #
        # 학술적 기반:
        # - Reciprocal Rank Fusion (RRF): 순위 기반 점수 융합
        # - Temperature Scaling: 점수 분포 캘리브레이션
        # - Min-Max Normalization: 정규화로 스케일 조정
        #
        # 점수 범위: 55% ~ 98% (43%p 범위)
        # - 기존: 65% ~ 98% (33%p) → 새 방식: 더 넓은 분포
        #
        raw_score = r.get('score', 0.5)  # Temperature Scaled score
        hybrid_score = r.get('hybrid_score', raw_score)  # 원본 Hybrid score

        # 최종 점수 = 55 + hybrid_score * 43
        # hybrid_score 범위가 0~1이므로 결과는 55~98%
        match_score = min(98, max(55, 55 + hybrid_score * 43))

        # persona 생성 (LLM 페르소나 우선 사용)
        rag_profile = metadata.copy()
        if 'persona' in rag_profile:
            content_type = metadata.get('content_type', '')
            main_mood = metadata.get('main_mood', '')
            target_age = metadata.get('target_age', '')

            # LLM 생성 페르소나가 있으면 우선 사용
            llm_persona = metadata.get('llm_persona', '')
            if llm_persona:
                # LLM 페르소나 사용 (예: "청담 컬러 마스터", "데일리 뷰티 크리에이터")
                rag_profile['persona'] = f"{llm_persona} | {main_mood}"
            else:
                # 폴백: 기존 규칙 기반 페르소나 생성
                # 콘텐츠 타입 한글화
                content_map = {
                    '시술결과': '시술 결과',
                    '전후비교': 'B&A',
                    '헤어팁': '헤어팁',
                    '튜토리얼': '튜토리얼',
                    '일상브이로그': '일상 브이로그',
                    '뷰티리뷰': '뷰티 리뷰'
                }
                content_desc = content_map.get(content_type, content_type)

                # 복합형 페르소나 생성: 시술 특화 + 콘텐츠 유형
                # 예시: "손상모 B&A 전문가", "염색 튜토리얼 전문가", "볼륨펌 시술 전문가"
                if inf_type == 'expert':
                    # 전문가: best_categories에서 세부 특화 분야 추출 + content_type 조합
                    # best_categories 예시: "트리트먼트-손상복구, 에센스-윤기"
                    best_cat = metadata.get('best_categories', '')

                    # 세부 특화 분야 추출 (대분류-세부분류에서 세부분류 우선)
                    specialty = ''
                    if best_cat:
                        first_item = best_cat.split(',')[0].strip()
                        if '-' in first_item:
                            # "트리트먼트-손상복구" → "손상복구"
                            parts = first_item.split('-')
                            specialty = parts[1].strip() if len(parts) > 1 else parts[0].strip()
                        else:
                            specialty = first_item

                    # 세부 분야 한글 표시 정리
                    specialty_display_map = {
                        # 케어 관련
                        '손상복구': '손상모 복구',
                        '손상모': '손상모 케어',
                        '손상케어': '손상모 케어',
                        '윤기': '윤기 케어',
                        '두피': '두피 케어',
                        '두피케어': '두피 케어',
                        '볼륨': '볼륨',
                        '탈모': '탈모 케어',
                        '보습': '보습 케어',

                        # 컬러 관련
                        '발레아쥬': '발레아쥬',
                        '하이라이트': '하이라이트',
                        '옴브레': '옴브레',
                        '블리치': '탈색',
                        '염색': '염색',
                        '탈색': '탈색',

                        # 스타일링 관련
                        '볼륨펌': '볼륨펌',
                        '웨이브': '웨이브펌',
                        '셋팅': '셋팅',
                        '드라이': '드라이 스타일링',
                        '커트': '커트',
                        '컷': '커트',

                        # 전문 분야
                        '전문가용': '프로페셔널',
                        '살롱급': '살롱 스타일',
                    }

                    # 콘텐츠 타입 → 전문가 유형 매핑
                    content_type_map = {
                        '전후비교': 'B&A',
                        '시술결과': '시술',
                        '튜토리얼': '튜토리얼',
                        '헤어팁': '팁',
                        '뷰티리뷰': '리뷰',
                        '일상브이로그': '브이로그',
                    }

                    specialty_display = specialty_display_map.get(specialty, specialty)
                    content_suffix = content_type_map.get(content_type, '시술')

                    if specialty_display:
                        # "손상모 복구 B&A 전문가", "발레아쥬 시술 전문가"
                        rag_profile['persona'] = f"{specialty_display} {content_suffix} 전문가 | {main_mood}"
                    else:
                        rag_profile['persona'] = f"헤어 {content_suffix} 전문가 | {main_mood}"
                else:
                    # 트렌드세터: "20대 여성 | 트렌디한 일상 브이로그"
                    rag_profile['persona'] = f"{target_age} 타겟 | {main_mood} {content_desc}"

        recommendations.append({
            'username': username,
            'match_score': round(match_score, 1),
            'rag_profile': rag_profile,
            'match_reason': _generate_recommendation_reason(
                metadata, brand_info, product_type, inf_type, request.description
            ),
            'influencer_data': influencer
        })

    # 매칭 점수 순으로 정렬
    recommendations.sort(key=lambda x: x['match_score'], reverse=True)

    # 순위 부여
    for i, rec in enumerate(recommendations):
        rec['rank'] = i + 1

    # 브랜드 분석 텍스트 생성
    brand_analysis = _generate_brand_analysis(
        request.brand_name,
        brand_info,
        product_type
    )

    # 제품 설명 요약
    description_summary = _summarize_description(request.description) if request.description else None

    return {
        'brand_info': {
            'name': request.brand_name,
            'slogan': brand_info.get('slogan', ''),
            'aesthetic_style': brand_info.get('aesthetic_style', ''),
            'core_values': brand_info.get('core_values', [])
        },
        'brand_analysis': brand_analysis,
        'description_summary': description_summary,
        'query': {
            'brand_name': request.brand_name,
            'product_type': request.product_type,
            'product_line': request.product_line,
            'description': request.description,
            'target_gender': request.target_gender,
            'expert_count': expert_count,
            'trendsetter_count': trendsetter_count
        },
        'total_results': len(recommendations),
        'recommendations': recommendations
    }


def _generate_recommendation_reason(metadata: Dict, brand_info: Dict, product_type: str, inf_type: str, campaign_description: str = None) -> str:
    """추천 사유 생성 - 인플루언서 특성에 맞게 상세하게 다각화"""
    import hashlib
    import re

    username = metadata.get('username', '')
    target_age = metadata.get('target_age', '')
    target_gender = metadata.get('target_gender', '')
    main_mood = metadata.get('main_mood', '')
    ad_approach = metadata.get('ad_approach', '')
    content_type = metadata.get('content_type', '')
    best_categories = metadata.get('best_categories', '')
    campaigns = metadata.get('campaigns', '')

    brand_name = brand_info.get('name', '')
    brand_style = brand_info.get('aesthetic_style', '')
    brand_values = brand_info.get('core_values', [])

    # 캠페인 설명에서 타겟 연령대 추출
    campaign_target_age = ''
    if campaign_description:
        age_match = re.search(r'(\d{2})[,\s]*(\d{2})?대', campaign_description)
        if age_match:
            if age_match.group(2):
                campaign_target_age = f"{age_match.group(1)},{age_match.group(2)}대"
            else:
                campaign_target_age = f"{age_match.group(1)}대"

    display_age = campaign_target_age if campaign_target_age else target_age
    gender_kr = {'female': '여성', 'male': '남성', 'unisex': ''}.get(target_gender, '')

    # 전문 분야 추출
    specialty = best_categories.split(',')[0].split('-')[0] if best_categories else '헤어케어'

    # 브랜드 스타일 한글화
    style_kr = {'Natural': '자연주의', 'Trendy': '트렌디', 'Luxury': '프리미엄', 'Classic': '클래식'}.get(brand_style, '')

    # 전문가 인플루언서 - 콘텐츠 타입 & 광고 접근법에 따라 다른 추천 사유
    if inf_type == 'expert':
        # 콘텐츠 타입별 차별화
        if content_type == '시술결과' or ad_approach == '비포애프터':
            reasons = [
                f"시술 전후 비교 콘텐츠를 전문적으로 제작하는 헤어 전문가입니다. {specialty} 분야에서 다년간 경험을 쌓으며 {display_age} 고객층에게 높은 신뢰를 받고 있습니다. {product_type} 사용 전후의 변화를 시각적으로 보여주는 콘텐츠로 '실제로 이렇게 달라진다'는 확신을 줄 수 있어 구매 전환율이 높습니다. 특히 {main_mood} 분위기의 촬영 스타일이 {brand_name}의 {style_kr} 이미지와 잘 어울립니다.",
                f"Before & After 콘텐츠 전문 헤어 디자이너로, 시술 결과물의 퀄리티가 매우 높습니다. {display_age} 고객을 주로 시술하며 해당 연령대의 모발 고민과 니즈를 정확히 파악하고 있습니다. {product_type}의 효과를 실제 시술 과정에서 자연스럽게 보여줄 수 있어, 단순 광고가 아닌 '전문가의 선택'으로 인식되는 것이 강점입니다. 팔로워들이 시술 예약 문의를 할 정도로 신뢰도가 높습니다.",
                f"{specialty} 분야 시술 결과를 꾸준히 공유하며 전문성을 인정받은 헤어 전문가입니다. 매 게시물마다 시술 전후 비교 사진과 함께 사용한 제품, 시술 과정을 상세히 설명해 교육적 가치가 높습니다. {product_type}을 활용한 시술 과정을 콘텐츠로 제작하면 '이 제품으로 이런 결과가 나온다'는 구체적인 효과 입증이 가능합니다. {brand_name}의 전문성 이미지 강화에 크게 기여할 수 있습니다."
            ]
        elif content_type == '튜토리얼' or ad_approach == '튜토리얼형':
            reasons = [
                f"헤어 케어 튜토리얼 콘텐츠로 높은 인기를 얻고 있는 전문가입니다. 단순히 제품을 보여주는 것이 아니라 '왜 이 제품이 좋은지', '어떻게 사용해야 효과적인지'를 단계별로 상세히 설명합니다. {product_type}의 올바른 사용법과 기대 효과를 교육적으로 전달할 수 있어, {display_age} 타겟층이 제품을 구매한 후에도 만족도가 높습니다. 댓글에서 사용법 질문이 많이 달려 자연스러운 2차 홍보 효과도 기대됩니다.",
                f"헤어 케어 노하우를 친절하게 공유하는 것으로 유명한 전문가입니다. {specialty} 관련 튜토리얼 영상이 평균 조회수가 높으며, 특히 {display_age} 시청자층에서 반응이 좋습니다. {product_type}을 활용한 홈케어 방법이나 살롱 케어 팁을 콘텐츠로 제작하면, 제품의 가치를 높이면서 실용적인 정보를 전달할 수 있습니다. {brand_name}의 '전문성'과 '신뢰성' 이미지에 부합합니다.",
                f"단계별 사용법 안내 콘텐츠로 팔로워들의 신뢰를 받는 헤어 전문가입니다. 복잡한 헤어 케어 과정을 쉽게 따라할 수 있도록 설명하는 능력이 뛰어납니다. {product_type}의 정확한 사용법과 함께 프로만 아는 팁을 공유하면 '전문가가 알려주는 꿀팁' 형태의 콘텐츠가 가능합니다. {display_age} 타겟층이 '나도 따라해봐야지'라는 반응을 보이며 실제 구매로 이어질 확률이 높습니다."
            ]
        elif ad_approach == '전문가추천':
            reasons = [
                f"살롱에서 실제로 사용하는 제품만 추천하는 것으로 유명한 헤어 전문가입니다. '내 살롱에서 직접 쓰는 제품'이라는 메시지가 강력한 신뢰를 주며, {display_age} 고객층에게 '전문가가 선택한 제품'이라는 프리미엄 이미지를 심어줍니다. {product_type}을 살롱에서 실제 시술에 활용하는 모습을 보여주면 단순 광고를 넘어 '검증된 제품'으로 인식됩니다. {brand_name}의 전문 브랜드 포지셔닝에 적합합니다.",
                f"미용 전문가로서 제품 추천에 까다롭기로 유명합니다. 그래서 이 인플루언서가 추천하는 제품은 팔로워들 사이에서 '진짜 좋은 제품'으로 인정받습니다. {product_type}의 성분, 효과, 가성비를 전문가 관점에서 분석하는 콘텐츠가 가능하며, {display_age} 타겟층에게 높은 설득력을 가집니다. 특히 {main_mood} 분위기의 콘텐츠가 {brand_name} 브랜드 톤앤매너와 잘 맞습니다.",
                f"다년간의 현장 경험을 바탕으로 고객 상담을 많이 해온 헤어 전문가입니다. {display_age} 고객들이 어떤 고민을 가지고 있는지, 어떤 제품이 실제로 효과가 있는지 누구보다 잘 알고 있습니다. '고객들에게 이 제품을 추천하는 이유'라는 형태의 콘텐츠로 {product_type}을 소개하면 진정성 있는 추천이 됩니다. 댓글에서 추가 질문이 많이 달려 자연스러운 소통이 이뤄집니다."
            ]
        else:
            reasons = [
                f"{main_mood} 분위기의 살롱 콘텐츠를 제작하는 헤어 전문가입니다. 피드 전체가 일관된 톤앤매너를 유지하고 있어 {brand_name}의 {style_kr} 브랜드 이미지와 자연스럽게 어울립니다. {product_type}을 살롱 인테리어와 함께 고급스럽게 노출하면 제품의 프리미엄 이미지가 강화됩니다. {display_age} 고객층을 주로 시술하며 해당 연령대의 니즈를 잘 파악하고 있습니다.",
                f"{specialty} 분야에서 전문성을 인정받은 헤어 디자이너입니다. {campaigns.split(',')[0] if campaigns else '다양한 브랜드 캠페인'}에 참여한 경험이 있어 광고 콘텐츠 제작에 능숙합니다. {product_type}의 효과를 {display_age} 고객 시술 결과와 함께 보여주면 타겟층에게 직접적으로 어필할 수 있습니다. {main_mood} 분위기의 콘텐츠 스타일이 브랜드 캠페인에 적합합니다.",
                f"살롱 현장에서 직접 제품을 사용하는 모습을 보여주는 콘텐츠로 신뢰를 얻고 있는 전문가입니다. '실제로 효과가 있어서 쓴다'는 메시지가 팔로워들에게 강하게 전달됩니다. {product_type}을 시술 과정에 자연스럽게 녹여 노출하면 PPL 느낌 없이 제품력을 입증할 수 있습니다. {brand_name}과의 장기적인 파트너십도 기대해볼 만합니다."
            ]
    else:
        # 트렌드세터 - 무드 & 콘텐츠 타입에 따라 다른 추천 사유
        if content_type == '일상브이로그':
            reasons = [
                f"일상 브이로그로 팔로워들과 친밀한 관계를 형성하고 있는 인플루언서입니다. 아침 루틴, 외출 준비 등 일상 콘텐츠에서 자연스럽게 {product_type}을 노출하면 광고처럼 느껴지지 않아 거부감이 적습니다. {display_age} {gender_kr} 팔로워들이 '나도 저렇게 해봐야지'라는 공감을 하며 자연스럽게 구매 욕구가 생깁니다. 댓글에서 '이거 어디 제품이에요?'라는 질문이 자주 달려 바이럴 효과가 큽니다.",
                f"데일리 루틴 콘텐츠로 {display_age} 팔로워들에게 높은 공감을 얻고 있습니다. 출근 전 헤어 스타일링, 샤워 후 헤어 케어 등 일상적인 상황에서 {product_type}을 사용하는 모습이 자연스럽게 녹아들 수 있습니다. '이 제품 쓰고 나서 머릿결이 달라졌어요' 같은 진정성 있는 후기가 가능해 팔로워들의 신뢰를 얻습니다. {main_mood} 감성의 피드가 {brand_name}과 잘 어울립니다.",
                f"일상 속 자연스러운 모습을 보여주며 팔로워들과 소통하는 인플루언서입니다. {display_age} {gender_kr} 타겟층과 비슷한 라이프스타일을 공유해 공감대 형성이 쉽습니다. {product_type}을 '요즘 매일 쓰는 아이템'으로 소개하면 일상에서 자연스럽게 스며드는 느낌을 줄 수 있습니다. 스토리나 릴스에서 사용 후기를 공유하면 즉각적인 반응과 함께 DM 문의도 많이 들어옵니다."
            ]
        elif content_type == '뷰티리뷰':
            reasons = [
                f"뷰티 제품 리뷰로 {display_age} 팔로워들에게 '믿고 보는 리뷰어'로 인정받고 있습니다. 제품의 장단점을 솔직하게 이야기해 신뢰도가 높으며, 이 인플루언서가 추천하면 실제 구매로 이어지는 비율이 높습니다. {product_type}의 사용감, 향, 효과를 상세히 리뷰하면 댓글에서 활발한 반응이 예상됩니다. {brand_name} 제품을 처음 접하는 소비자들에게 좋은 첫인상을 심어줄 수 있습니다.",
                f"뷰티 리뷰 콘텐츠 전문으로 제품 분석 능력이 뛰어난 인플루언서입니다. 단순히 '좋다'가 아니라 '어떤 점이 좋고, 어떤 타입에게 맞는지'까지 구체적으로 설명합니다. {product_type}의 텍스처, 향, 지속력, 가성비 등을 꼼꼼히 리뷰하면 {display_age} 타겟층의 구매 결정에 직접적인 영향을 줍니다. 리뷰 저장 수가 높아 장기적인 노출 효과도 기대됩니다.",
                f"화장품과 헤어 제품 리뷰로 {display_age} {gender_kr} 팔로워들의 신뢰를 받는 인플루언서입니다. 리뷰 콘텐츠마다 댓글이 활발하게 달리며, 팔로워들이 구매 전 이 인플루언서의 리뷰를 찾아볼 정도입니다. {product_type}을 일주일 사용 후기 형태로 리뷰하면 진정성 있는 추천이 가능합니다. {brand_name}의 신제품 론칭이나 리뉴얼 제품 홍보에 특히 적합합니다."
            ]
        elif main_mood in ['시크한', '모던한', '미니멀']:
            reasons = [
                f"{main_mood} 감성의 세련된 피드를 운영하며 {display_age} {gender_kr} 팔로워들에게 '워너비' 이미지를 가진 인플루언서입니다. 깔끔하고 정돈된 피드에 {product_type}을 노출하면 제품 자체도 고급스럽게 인식됩니다. {brand_name}의 {style_kr} 브랜드 톤과 완벽하게 맞아 캠페인 비주얼의 통일성을 높일 수 있습니다. 팔로워들이 '저 제품 뭐예요?'라고 물어볼 만큼 자연스러운 노출이 가능합니다.",
                f"깔끔하고 미니멀한 피드 구성으로 {display_age} {gender_kr} 타겟층에게 어필하는 인플루언서입니다. 불필요한 꾸밈 없이 제품 자체의 매력을 보여주는 콘텐츠 스타일이 {product_type}의 본질적인 효과를 강조하기에 적합합니다. {brand_name}의 브랜드 이미지와 일관성 있는 비주얼을 유지하면서 자연스러운 홍보가 가능합니다. 저장과 공유 수가 높아 도달률도 좋습니다.",
                f"{main_mood} 무드의 콘텐츠로 프리미엄 감성을 전달하는 인플루언서입니다. {product_type}을 일상 속 '작은 사치' 또는 '나를 위한 투자' 아이템으로 포지셔닝하면 {display_age} 타겟층의 구매 욕구를 자극할 수 있습니다. {brand_name}의 고급 이미지 강화에 기여하며, 브랜드 캠페인의 전체적인 톤앤매너를 높여줍니다."
            ]
        elif main_mood in ['트렌디한', '힙한', '캐주얼']:
            reasons = [
                f"MZ세대가 좋아하는 {main_mood} 콘텐츠로 {display_age} 팔로워들에게 트렌드세터로 인식되는 인플루언서입니다. 릴스, 숏폼 등 SNS 트렌드에 맞는 포맷으로 콘텐츠를 제작해 높은 조회수와 공유율을 기록합니다. {product_type}을 '요즘 핫한 아이템'으로 소개하면 자연스러운 바이럴 효과가 기대됩니다. {brand_name}을 젊은 층에게 어필하고 싶다면 최적의 선택입니다.",
                f"{main_mood} 감성으로 {display_age} 팔로워들과 활발히 소통하는 인플루언서입니다. 댓글과 DM 반응률이 높아 콘텐츠 업로드 후 즉각적인 반응을 확인할 수 있습니다. {product_type}을 '찐템'이나 '요즘 매일 쓰는 것'으로 소개하면 팔로워들의 관심을 끌기 좋습니다. 스토리 멘션과 태그가 활발해 2차 바이럴 효과도 큽니다.",
                f"트렌디한 콘텐츠 스타일로 젊은 층의 관심을 끄는 인플루언서입니다. {display_age} 타겟층이 따라하고 싶어하는 스타일과 라이프스타일을 보여주며, {product_type}을 이 맥락에 자연스럽게 녹이면 '나도 저거 써봐야겠다'는 반응을 이끌어냅니다. {brand_name}의 젊고 역동적인 이미지를 강화하기에 적합하며, 신규 고객 유입에 효과적입니다."
            ]
        else:
            reasons = [
                f"{main_mood} 분위기의 피드로 {display_age} 팔로워들과 꾸준히 소통하는 인플루언서입니다. {brand_name}의 브랜드 감성과 시너지를 낼 수 있는 콘텐츠 스타일을 가지고 있습니다. {product_type}을 일상 속에서 자연스럽게 사용하는 모습을 보여주면 광고 같지 않은 진정성 있는 홍보가 가능합니다. 팔로워들의 댓글 반응이 좋아 브랜드에 대한 긍정적인 인식을 형성하는 데 도움이 됩니다.",
                f"팔로워들과의 소통이 활발해 콘텐츠 업로드 시 빠른 반응을 얻는 인플루언서입니다. {product_type} 사용 후기를 공유하면 댓글에서 추가 질문과 함께 2차 바이럴이 일어납니다. {display_age} 타겟층과 라이프스타일이 비슷해 공감대 형성이 쉬우며, '이 사람이 쓰면 나도 써볼까?'라는 반응을 이끌어냅니다. {brand_name}의 인지도와 호감도를 동시에 높일 수 있습니다.",
                f"진정성 있는 콘텐츠로 팔로워들의 신뢰를 받는 인플루언서입니다. 광고가 아닌 진짜 추천처럼 느껴지는 것이 가장 큰 강점입니다. {product_type}을 '요즘 진짜 좋아서 쓰는 것'으로 자연스럽게 소개하면 {display_age} {gender_kr} 타겟층의 구매 욕구를 자극합니다. {brand_name}과 장기적인 앰버서더 관계도 기대해볼 수 있습니다."
            ]

    # 해시로 템플릿 선택 (같은 인플루언서는 항상 같은 템플릿)
    hash_val = int(hashlib.md5(username.encode()).hexdigest(), 16) % len(reasons)
    reason = reasons[hash_val]

    # 빈 값 정리
    reason = reason.replace('  ', ' ')
    return reason


def _generate_brand_analysis(brand_name: str, brand_info: Dict, product_type: str) -> str:
    """브랜드 분석 텍스트 생성"""
    style = brand_info.get("aesthetic_style", "Trendy")
    slogan = brand_info.get("slogan", "")
    core_values = brand_info.get("core_values", [])

    style_desc_map = {
        'Luxury': '프리미엄과 고급스러움을 추구하는',
        'Natural': '자연친화적이고 건강한 이미지의',
        'Trendy': '트렌디하고 젊은 감성의',
        'Classic': '클래식하고 전통적인 가치를 중시하는',
        'Minimal': '심플하고 세련된 미니멀리즘의',
        'Colorful': '화려하고 개성있는'
    }
    style_text = style_desc_map.get(style, '다양한 매력을 가진')

    analysis = f"'{brand_name}'은 {style_text} 브랜드입니다."

    if slogan:
        analysis += f" '{slogan}'이라는 슬로건 아래,"

    if core_values:
        analysis += f" '{', '.join(core_values[:3])}'을 핵심 가치로 삼고 있습니다."

    return analysis


def _summarize_description(description: str) -> str:
    """제품 설명을 요약"""
    if not description or len(description) < 30:
        return description

    # 핵심 키워드 추출하여 요약
    keywords = []

    # 타겟층 추출
    if '여성' in description:
        keywords.append('여성 타겟')
    if '남성' in description:
        keywords.append('남성 타겟')

    # 연령대 추출
    import re
    age_match = re.search(r'(\d{2})[,\s]*(\d{2})?대', description)
    if age_match:
        if age_match.group(2):
            keywords.append(f"{age_match.group(1)}-{age_match.group(2)}대")
        else:
            keywords.append(f"{age_match.group(1)}대")

    # 제품 특성 추출
    product_keywords = ['탈모', '손상', '볼륨', '윤기', '두피', '케어', '완화', '예방', '보습', '영양']
    found_product = [k for k in product_keywords if k in description]
    keywords.extend(found_product[:3])

    # 홍보 목적 추출
    if '출시' in description or '신제품' in description:
        keywords.append('신제품 런칭')
    if '효과' in description:
        keywords.append('효과 입증')
    if '홍보' in description:
        keywords.append('제품 홍보')

    if keywords:
        return ' / '.join(keywords)
    else:
        # 키워드 추출 실패 시 앞부분만
        return description[:50] + '...' if len(description) > 50 else description


# ============== 제품 API ==============

@router.get("/product-categories")
async def get_product_categories():
    """제품 카테고리 목록"""
    categories = []
    for name, info in PRODUCT_CATEGORIES.items():
        categories.append({
            'name': name,
            'description': info['description'],
            'icon': info['icon'],
            'target': info['target'],
            'product_count': len(info['products'])
        })
    return {"categories": categories, "total_categories": len(categories)}


@router.get("/product-categories/{category_name}")
async def get_products_by_category(category_name: str):
    """카테고리별 제품 목록"""
    if category_name not in PRODUCT_CATEGORIES:
        raise HTTPException(
            status_code=404,
            detail=f"카테고리 '{category_name}'을(를) 찾을 수 없습니다"
        )

    info = PRODUCT_CATEGORIES[category_name]
    return {
        "category": category_name,
        "description": info['description'],
        "icon": info['icon'],
        "target": info['target'],
        "products": info['products'],
        "product_count": len(info['products'])
    }


@router.get("/product-types")
async def get_product_types():
    """전체 제품 유형 목록"""
    categories = {name: info['products'] for name, info in PRODUCT_CATEGORIES.items()}
    all_products = []
    for info in PRODUCT_CATEGORIES.values():
        all_products.extend(info['products'])

    return {
        "categories": categories,
        "all_products": all_products,
        "total": len(all_products)
    }


# ============== 인플루언서 API ==============

@router.get("/influencers")
async def get_influencers():
    """인플루언서 목록"""
    return {"influencers": _influencers, "total": len(_influencers)}


@router.get("/influencers/{username}")
async def get_influencer_detail(username: str):
    """인플루언서 상세 정보"""
    from pipeline import FISCalculator, InfluencerClassifier

    influencer = next((inf for inf in _influencers if inf["username"] == username), None)
    if not influencer:
        raise HTTPException(status_code=404, detail="인플루언서를 찾을 수 없습니다")

    fis_calc = FISCalculator()
    classifier = InfluencerClassifier()

    return {
        "influencer": influencer,
        "classification": classifier.classify(influencer),
        "fis": fis_calc.calculate(influencer)
    }


# ============== RAG 관리 API ==============

class RAGAnalyzeRequest(BaseModel):
    """인플루언서 분석 요청"""
    usernames: Optional[List[str]] = None  # None이면 전체 분석
    force_reanalyze: Optional[bool] = False


@router.post("/rag/analyze")
async def rag_analyze_influencers(request: RAGAnalyzeRequest):
    """
    인플루언서 이미지 분석 및 RAG 인덱싱

    LLM Vision으로 인플루언서의 이미지를 분석하고
    결과를 벡터 DB에 저장합니다.
    """
    from pipeline import RAG_AVAILABLE

    if not RAG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="RAG 시스템이 사용 불가능합니다. ChromaDB를 설치해주세요: pip install chromadb"
        )

    from pipeline import InfluencerAnalysisManager

    manager = InfluencerAnalysisManager()

    # 분석 대상 결정
    if request.usernames:
        targets = [inf for inf in _influencers if inf['username'] in request.usernames]
    else:
        targets = _influencers

    # 분석 및 인덱싱 수행
    stats = manager.analyze_and_index_all(targets, force_reanalyze=request.force_reanalyze)

    return {
        'status': 'completed',
        'stats': stats
    }


@router.get("/rag/status")
async def rag_status():
    """RAG 시스템 상태 확인"""
    from pipeline import RAG_AVAILABLE

    if not RAG_AVAILABLE:
        return {
            'available': False,
            'message': 'ChromaDB not installed'
        }

    from pipeline import InfluencerRAG

    try:
        rag = InfluencerRAG()
        indexed_usernames = rag.get_all_usernames()

        return {
            'available': True,
            'indexed_count': len(indexed_usernames),
            'total_influencers': len(_influencers),
            'indexed_usernames': indexed_usernames[:10]  # 샘플만
        }
    except Exception as e:
        return {
            'available': False,
            'error': str(e)
        }


@router.get("/rag/influencer/{username}")
async def rag_get_influencer_profile(username: str):
    """RAG에 저장된 인플루언서 프로필 조회"""
    from pipeline import RAG_AVAILABLE

    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG 시스템 사용 불가")

    from pipeline import InfluencerRAG

    rag = InfluencerRAG()
    profile = rag.get_influencer(username)

    if not profile:
        raise HTTPException(status_code=404, detail="인플루언서 프로필이 없습니다. 먼저 분석을 실행하세요.")

    return profile
