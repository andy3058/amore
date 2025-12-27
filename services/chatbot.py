"""
챗봇 서비스 - 자연어 처리 및 대화 관리
"""

from typing import Optional, Dict, List
import re

from config.products import PRODUCT_KEYWORDS
from modules.matcher import get_full_recommendations


class ChatBot:
    """자연어 입력 기반 챗봇"""

    def __init__(self, brand_db: Dict, influencers: List[Dict]):
        self.sessions = {}
        self.brand_db = brand_db
        self.influencers = influencers

    def get_or_create_session(self, session_id: str) -> Dict:
        """세션 생성 또는 가져오기"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "state": "initial",
                "brand_name": None,
                "target_audience": None,
                "product_type": None,
                "aesthetic_style": None,
                "history": []
            }
        return self.sessions[session_id]

    def extract_brand_from_message(self, message: str) -> Optional[str]:
        """메시지에서 브랜드명 추출"""
        brands = list(self.brand_db.get("brands", {}).keys())
        message_lower = message.lower()

        for brand in brands:
            if brand.lower() in message_lower or brand in message:
                return brand

        for brand, info in self.brand_db.get("brands", {}).items():
            en_name = info.get("brand_name_en", "").lower()
            if en_name and en_name in message_lower:
                return brand

        return None

    def extract_product_type(self, message: str) -> Optional[str]:
        """메시지에서 제품 유형 추출"""
        message_lower = message.lower()
        for product, keywords in PRODUCT_KEYWORDS.items():
            for kw in keywords:
                if kw in message_lower:
                    return product
        return None

    def extract_target_audience(self, message: str) -> Optional[str]:
        """메시지에서 타겟 고객층 추출"""
        age_patterns = [r"(\d{2})대", r"(\d{2})~(\d{2})대", r"(\d{2})-(\d{2})대"]

        for pattern in age_patterns:
            match = re.search(pattern, message)
            if match:
                return match.group(0)

        audience_keywords = {
            "MZ세대": ["mz", "엠제트", "젊은"],
            "20대": ["20대", "이십대"],
            "30대": ["30대", "삼십대"],
            "40대": ["40대", "사십대"],
            "50대": ["50대", "오십대"],
            "전문가": ["전문가", "미용사", "살롱"],
            "일반 소비자": ["일반", "대중", "소비자"]
        }

        message_lower = message.lower()
        for audience, keywords in audience_keywords.items():
            for kw in keywords:
                if kw in message_lower:
                    return audience
        return None

    def extract_aesthetic_style(self, message: str) -> Optional[str]:
        """메시지에서 비주얼 스타일 추출"""
        style_keywords = {
            "Luxury": ["럭셔리", "고급", "프리미엄", "luxury", "premium"],
            "Natural": ["자연", "내추럴", "친환경", "natural", "clean"],
            "Trendy": ["트렌디", "힙", "mz", "trendy", "modern"],
            "Classic": ["클래식", "전통", "한방", "classic"],
            "Mass": ["대중적", "가성비", "데일리", "mass"],
            "Minimal": ["미니멀", "심플", "minimal", "simple"],
            "Colorful": ["컬러풀", "화려", "colorful", "vivid"]
        }

        message_lower = message.lower()
        for style, keywords in style_keywords.items():
            for kw in keywords:
                if kw in message_lower:
                    return style
        return None

    def process_message(self, message: str, session_id: str) -> Dict:
        """메시지 처리 및 응답 생성"""
        session = self.get_or_create_session(session_id)
        session["history"].append({"role": "user", "content": message})

        # 정보 추출
        brand_name = self.extract_brand_from_message(message)
        if brand_name:
            session["brand_name"] = brand_name

        product_type = self.extract_product_type(message)
        if product_type:
            session["product_type"] = product_type

        target_audience = self.extract_target_audience(message)
        if target_audience:
            session["target_audience"] = target_audience

        aesthetic_style = self.extract_aesthetic_style(message)
        if aesthetic_style:
            session["aesthetic_style"] = aesthetic_style

        # 응답 생성
        response, recommendations, brand_info = self._generate_response(session, message)
        session["history"].append({"role": "assistant", "content": response})

        return {
            "response": response,
            "recommendations": recommendations,
            "brand_info": brand_info,
            "session_data": {
                "brand_name": session["brand_name"],
                "target_audience": session["target_audience"],
                "product_type": session["product_type"],
                "aesthetic_style": session["aesthetic_style"]
            }
        }

    def _generate_response(self, session: Dict, message: str) -> tuple:
        """응답 생성 로직"""
        brand_name = session.get("brand_name")
        target_audience = session.get("target_audience")
        product_type = session.get("product_type")
        aesthetic_style = session.get("aesthetic_style")

        # 인사 메시지
        greetings = ["안녕", "hello", "hi", "시작", "도움"]
        if any(g in message.lower() for g in greetings):
            return self._get_welcome_message(), None, None

        # 추천 요청 감지
        recommend_keywords = ["추천", "찾아", "알려", "매칭", "인플루언서", "추천해"]
        is_recommend_request = any(kw in message for kw in recommend_keywords)

        # 브랜드 정보만 요청
        if brand_name and not is_recommend_request:
            brand_info = self.brand_db.get("brands", {}).get(brand_name)
            if brand_info:
                response = f"**{brand_name}** 브랜드 정보입니다:\n\n"
                response += f"- **슬로건**: {brand_info.get('slogan', 'N/A')}\n"
                response += f"- **핵심 가치**: {', '.join(brand_info.get('core_values', []))}\n"
                response += f"- **스타일**: {brand_info.get('aesthetic_style', 'N/A')}\n\n"
                response += "인플루언서 추천을 원하시면 **타겟 고객층**과 **제품 유형**을 알려주세요."
                return response, None, brand_info

        # 추천 실행
        if is_recommend_request or (brand_name and (target_audience or product_type)):
            return self._execute_recommendation(session)

        # 정보 부족
        missing = []
        if not brand_name:
            missing.append("브랜드명")
        if not target_audience:
            missing.append("타겟 고객층")
        if not product_type:
            missing.append("제품 유형")

        if missing:
            response = f"인플루언서 추천을 위해 다음 정보가 필요합니다:\n"
            for item in missing:
                response += f"- {item}\n"
            response += "\n예시: '려 브랜드로 30대 여성 대상 탈모샴푸 마케팅을 위한 인플루언서 추천해줘'"
            return response, None, None

        return self._get_welcome_message(), None, None

    def _execute_recommendation(self, session: Dict) -> tuple:
        """추천 실행"""
        brand_name = session.get("brand_name")
        brand_info = self.brand_db.get("brands", {}).get(brand_name, {})

        brand_data = {
            "brand_name": brand_name or "Unknown",
            "slogan": brand_info.get("slogan", ""),
            "core_values": brand_info.get("core_values", []),
            "target_audience": session.get("target_audience") or brand_info.get("age_target", ""),
            "product_type": session.get("product_type") or brand_info.get("product_categories", ["샴푸"])[0],
            "aesthetic_style": session.get("aesthetic_style") or brand_info.get("aesthetic_style", "Trendy")
        }

        try:
            results = get_full_recommendations(brand_data, self.influencers, top_k=5, min_fis=60)

            response = f"## {brand_name} 브랜드를 위한 인플루언서 추천 결과\n\n"
            response += f"**타겟**: {brand_data['target_audience']} | **제품**: {brand_data['product_type']}\n\n"
            response += f"분석: {results['total_analyzed']}명 → FIS 통과: {results['total_passed_fis']}명\n\n---\n\n"

            for rec in results["recommendations"]:
                response += f"### {rec['rank']}. @{rec['username']}\n"
                response += f"- 팔로워: {rec['followers']:,}명 | 타입: {rec['type']}\n"
                response += f"- FIS: {rec['fis_score']:.0f}점 | 매칭: {rec['match_score']:.1f}%\n"
                response += f"- 추천 사유: {rec['reason']}\n\n"

            return response, results["recommendations"], brand_info
        except Exception as e:
            return f"추천 처리 중 오류: {str(e)}", None, None

    def _get_welcome_message(self) -> str:
        """환영 메시지"""
        hair_brands = self.brand_db.get("hair_brands", [])
        return f"""# AI 헤어 인플루언서 큐레이션 에이전트

브랜드에 최적화된 헤어 인플루언서를 추천해드립니다.

## 사용 예시
- "려 브랜드로 30대 여성 대상 탈모샴푸 마케팅 인플루언서 추천해줘"
- "미쟝센 트렌디한 스타일의 인플루언서 찾아줘"

## 지원 브랜드
{', '.join(hair_brands)}

무엇을 도와드릴까요?
"""