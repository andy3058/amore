"""
MVP용 인플루언서 샘플 데이터 생성 스크립트 (확장판)
====================================================

두 가지 형태의 데이터 생성:
1. influencers_raw.json: 크롤러에서 수집한 형태 (분류/분석 없음)
2. influencers_data.json: Processor에서 처리된 형태 (분류/분석 완료)

Instagram Graph API 스키마에 맞는 150명의 샘플 데이터 생성
- Expert (헤어 전문가): 70명 (다양한 연령대, 성별 전문성)
- Trendsetter (패션/라이프스타일): 80명 (다양한 연령대, 스타일)

FIS 점수 분포:
- 신뢰 가능 (80-98): 약 60%
- 주의 필요 (60-79): 약 25%
- 위험 (40-59): 약 15%

각 인플루언서당 최대 10개의 릴스(VIDEO) 게시물 포함
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path


# ============================================================
# 데이터 생성용 상수
# ============================================================

# Expert용 데이터 (120명 - 다양한 전문 분야)
# 동적 생성을 위한 베이스 이름들
EXPERT_USERNAME_BASES = [
    # 여성 타겟 전문가 (염색/펌) - 40명
    "hair_master", "salon_beauty", "color_specialist", "perm_artist",
    "cheongdam_hair", "gangnam_stylist", "hair_clinic", "beauty_director",
    "styling_pro", "color_queen", "hair_doctor", "salon_style",
    "premium_hair", "hair_healing", "style_creator", "hair_lab",
    "beauty_hair", "salon_director", "hair_specialist", "color_master",
    "perm_pro", "hair_artist", "beauty_expert", "salon_master",
    "hair_designer", "color_artist", "perm_specialist", "hair_studio",
    "beauty_lab", "salon_expert", "hair_pro", "style_expert",
    "hair_queen", "color_pro", "perm_master", "beauty_studio",
    "salon_pro", "hair_center", "style_master", "beauty_center",
    # 남성 타겟 전문가 - 25명
    "mens_hair", "barber_master", "mens_cut", "gentleman_salon",
    "barber_shop", "mens_style", "male_hair", "barber_artist",
    "mens_grooming", "gentleman_barber", "mens_salon", "barber_pro",
    "male_grooming", "mens_designer", "barber_studio", "gentleman_style",
    "mens_expert", "barber_lab", "male_stylist", "mens_clinic",
    "barber_center", "gentleman_cut", "mens_master", "barber_expert",
    "male_barber",
    # 중년/시니어 전문가 - 20명
    "mature_hair", "senior_beauty", "ageless_salon", "midlife_hair",
    "silver_hair", "classic_beauty", "elegant_hair", "timeless_style",
    "graceful_hair", "premium_age", "mature_style", "senior_salon",
    "ageless_beauty", "midlife_style", "silver_salon", "classic_hair",
    "elegant_beauty", "timeless_hair", "graceful_salon", "premium_mature",
    # 두피/탈모 전문가 - 20명
    "scalp_healing", "hair_loss", "trichology", "scalp_doctor",
    "hair_regrowth", "alopecia_care", "healthy_scalp", "hair_restoration",
    "scalp_care", "hair_health", "scalp_clinic", "hair_therapy",
    "scalp_expert", "hair_recovery", "scalp_pro", "hair_solution",
    "scalp_master", "hair_renewal", "scalp_lab", "hair_revive",
    # 웨딩/특수 헤어 - 15명
    "wedding_hair", "bridal_beauty", "special_occasion", "event_stylist",
    "celebrity_hair", "photoshoot_hair", "wedding_style", "bridal_salon",
    "occasion_hair", "event_hair", "celeb_stylist", "studio_hair",
    "wedding_pro", "bridal_expert", "special_hair",
]

def generate_expert_usernames(count: int) -> list:
    """Expert 유저네임 동적 생성"""
    usernames = []
    suffixes = ["_kim", "_lee", "_park", "_cho", "_jung", "_oh", "_han", "_yoon",
                "_seoul", "_korea", "_pro", "_lab", "_studio", "_center", "_k", "_j", "_m", "_y"]
    for i, base in enumerate(EXPERT_USERNAME_BASES):
        if len(usernames) >= count:
            break
        suffix = suffixes[i % len(suffixes)]
        usernames.append(f"{base}{suffix}")
    # 부족하면 번호 붙여서 추가
    while len(usernames) < count:
        idx = len(usernames)
        base = EXPERT_USERNAME_BASES[idx % len(EXPERT_USERNAME_BASES)]
        usernames.append(f"{base}_{idx}")
    return usernames[:count]

EXPERT_USERNAMES = generate_expert_usernames(120)

# 전문가 바이오 - 타겟 연령대/성별별 분류
EXPERT_BIOS_FEMALE_YOUNG = [  # 20대 여성 타겟
    "청담동 헤어살롱 원장 | 15년차 미용사 | 염색 & 펌 전문 | 예약문의 DM",
    "강남 프리미엄 헤어숍 | 컬러리스트 | 손상모 복구 전문 | 카카오톡 예약",
    "홍대 스타일리스트 | 10년 경력 | 트렌디한 염색 | 예약 링크 ⬇️",
    "압구정 살롱 디렉터 | 펌 전문가 | 자연스러운 웨이브 | DM 예약",
    "신사동 헤어살롱 | 12년차 디자이너 | 볼륨펌 전문 | 예약문의 카톡",
    "청담 컬러 전문숍 | 하이톤 염색 | 블리치 전문 | 예약 DM",
    "서초 프리미엄살롱 | 헤어클리닉 | 손상모 트리트먼트 | DM 상담",
    "마포 스타일리스트 | 염색 전문 | 애쉬계열 컬러 | 예약문의",
    "용산 헤어아티스트 | C컬펌 전문 | 자연스러운 볼륨 | DM 예약",
    "성수 트렌디살롱 | MZ 감성 | 레이어드컷 전문 | DM 예약",
]

EXPERT_BIOS_FEMALE_MATURE = [  # 30~40대 여성 타겟
    "강남 프리미엄살롱 | 30대 여성 전문 | 자연스러운 볼륨펌 | 예약 DM",
    "목동 살롱원장 | 20년 경력 | 중년 헤어 전문 | 상담예약 DM",
    "분당 프리미엄헤어 | 15년차 원장 | VIP 고객 전담 | 예약 DM",
    "판교 헤어살롱 | 40대 여성 맞춤 | 커버 그레이 전문 | 카카오 예약",
    "일산 뷰티살롱 | 30~40대 전문 | 우아한 스타일링 | DM 상담",
    "용인 프리미엄헤어 | 중년 여성 헤어 | 볼륨 & 윤기 | 예약문의",
    "수원 살롱디렉터 | 18년 경력 | 세련된 중년 스타일 | 카톡 예약",
    "송파 헤어클리닉 | 30대 직장인 전문 | 관리 쉬운 스타일 | DM",
    "서초 뷰티스튜디오 | 40대 맞춤 컬러 | 새치 케어 | 예약 DM",
    "강동 프리미엄살롱 | 중년 볼륨펌 | 자연스러운 컬 | 상담예약",
]

EXPERT_BIOS_MALE = [  # 남성 타겟
    "강남역 헤어스튜디오 | 8년차 미용사 | 남성 커트 전문 | 예약문의",
    "홍대 바버샵 | 남성 전문 | 투블럭 & 페이드 | DM 예약",
    "청담 맨즈헤어 | 남성 스타일리스트 | 비즈니스룩 전문 | 카카오톡",
    "강남 바버마스터 | 10년차 | 남성 그루밍 전문 | 예약 DM",
    "서초 맨즈살롱 | 남성 펌 전문가 | 볼륨 & 다운펌 | 상담문의",
    "역삼 바버샵 | 남성 두피케어 | 탈모 예방 전문 | DM 예약",
    "삼성 맨즈스튜디오 | 직장인 남성 전문 | 깔끔한 스타일 | 예약",
    "잠실 남성헤어 | 20대 남성 전문 | 트렌디한 커트 | 카톡 예약",
    "송파 바버클럽 | 남성 그루밍 | 면도 & 헤어 | DM 상담",
    "강서 맨즈헤어랩 | 30대 남성 맞춤 | 볼륨감 있는 스타일 | 예약",
]

EXPERT_BIOS_SCALP = [  # 두피/탈모 전문
    "성수동 헤어랩 | 두피케어 전문 | 탈모 예방 클리닉 | 상담문의 DM",
    "송파 헤어살롱 | 두피관리 | 탈모케어 전문 | 카카오 예약",
    "강남 두피클리닉 | 탈모 전문 | 모발 이식 상담 | 예약 DM",
    "서초 트리콜로지 | 두피 진단 전문 | 맞춤 케어 | 상담문의",
    "분당 헤어클리닉 | 탈모 예방 | 두피 스케일링 | DM 예약",
    "일산 두피센터 | 여성 탈모 전문 | 볼륨 케어 | 카카오 상담",
    "수원 모발클리닉 | 남성 탈모 | M자 케어 전문 | 예약문의",
    "용인 두피힐링 | 지성 두피 전문 | 비듬 케어 | DM 상담",
]

EXPERT_BIOS_WEDDING = [  # 웨딩/특수 헤어
    "잠실 헤어디자이너 | 웨딩헤어 전문 | 업스타일 | 예약 카카오톡",
    "청담 브라이덜 | 웨딩 전문 | 신부 헤어메이크업 | 상담 DM",
    "강남 웨딩스튜디오 | 특별한 날 전문 | 이벤트 헤어 | 예약문의",
    "압구정 브라이덜헤어 | 웨딩촬영 전문 | 업스타일 | 카카오톡",
    "서초 파티헤어 | 행사 헤어 전문 | 연예인 스타일 | DM 예약",
    "목동 웨딩살롱 | 결혼식 헤어 | 하객 스타일링 | 상담문의",
]

# 기본 바이오 (혼합)
EXPERT_BIOS = (EXPERT_BIOS_FEMALE_YOUNG + EXPERT_BIOS_FEMALE_MATURE +
              EXPERT_BIOS_MALE + EXPERT_BIOS_SCALP + EXPERT_BIOS_WEDDING)

EXPERT_CAPTIONS = [
    "오늘의 시술 - 웜톤 고객님께 어울리는 가을 염색 레시피 공개! #염색약 #헤어컬러 #미용사일상",
    "C컬 펌 시술 과정 풀영상! 자연스러운 볼륨감 살리는 비법 #펌 #시술영상 #살롱",
    "손상모 케어 전후 비교! 클리닉 트리트먼트 효과 #헤어클리닉 #손상모케어",
    "애쉬브라운 염색 시술 🎨 쿨톤 고객님 맞춤 컬러 #애쉬브라운 #염색전문",
    "볼륨펌 시술 완료! 뿌리볼륨 살리는 테크닉 공개 #볼륨펌 #펌전문",
    "블리치 없이 하이톤 염색하기 💫 손상 최소화 비법 #하이톤염색 #컬러리스트",
    "레이어드컷 시술 영상 ✂️ 얼굴형에 맞는 커트라인 #레이어드컷 #커트전문",
    "두피 스케일링 전후 비교! 건강한 두피 만들기 #두피케어 #탈모예방",
    "웨딩헤어 업스타일 시술 👰 신부님 헤어 완성 #웨딩헤어 #업스타일",
    "남성 투블럭 커트 시술 💇‍♂️ 깔끔한 라인 정리 #남성커트 #투블럭",
    "염색약 조색 과정 공개! 맞춤 컬러 만들기 #조색 #컬러레시피",
    "히피펌 시술 완료 🌊 자연스러운 웨이브 연출 #히피펌 #웨이브펌",
    "탈색 후 톤다운 염색 시술 🎨 손상 케어 포함 #톤다운 #염색시술",
    "숱치기 없이 볼륨감 살리는 커트 테크닉 ✂️ #볼륨커트 #시술영상",
    "클리닉 트리트먼트 시술 과정 💆‍♀️ 손상모 집중 케어 #헤어클리닉 #트리트먼트",
]

# Trendsetter용 데이터 (180명 - 다양한 연령대, 스타일)
TRENDSETTER_USERNAME_BASES = [
    # 20대 여성 (MZ세대) - 70명
    "haru_style", "minjung_daily", "yuna_ootd", "seo_fashion", "jin_lookbook",
    "hyun_daily", "sua_style", "minji_look", "yeonhee_ootd", "jiwon_fashion",
    "hana_daily", "sooyeon_style", "eunji_look", "dahyun_ootd", "chaeyoung_life",
    "nayeon_style", "jihyo_daily", "momo_look", "sana_ootd", "tzuyu_fashion",
    "rose_style", "jennie_daily", "lisa_look", "jisoo_ootd", "irene_fashion",
    "seulgi_style", "wendy_daily", "joy_look", "yeri_ootd", "winter_fashion",
    "karina_style", "giselle_daily", "ningning_look", "yujin_ive", "wonyoung_ootd",
    "leeseo_style", "gaeul_daily", "rei_look", "liz_ootd", "kazuha_fashion",
    "sakura_style", "chaewon_daily", "yunjin_look", "eunchae_ootd", "sullyoon_fashion",
    "haewon_style", "bae_daily", "jiwoo_look", "lily_ootd", "kyujin_fashion",
    "yeji_style", "lia_daily", "ryujin_look", "chaeryeong_ootd", "yuna_fashion",
    "minju_style", "yujin_daily", "wonyoung_look", "gaeul_ootd", "rei_fashion",
    "sieun_style", "yoon_daily", "sumin_look", "isa_ootd", "jiyeon_fashion",
    "yeonjung_style", "seola_daily", "bona_look", "exy_ootd", "soobin_fashion",
    # 30대 여성 - 40명
    "worklife_soyeon", "office_style_j", "career_woman_kim", "elegant_jihye",
    "chic_soojin", "modern_lady_lee", "classy_mirae", "refined_yoona",
    "sophisticated_hana", "professional_beauty", "city_girl_seoul", "urban_style_k",
    "business_chic_j", "polished_look_m", "smart_casual_y", "office_chic_kim",
    "career_style_lee", "working_mom_j", "modern_office_k", "elegant_30s_m",
    "chic_career_y", "professional_look_h", "city_style_seoul", "urban_chic_k",
    "business_look_j", "polished_style_m", "smart_office_y", "classy_30s_kim",
    "refined_career_lee", "sophisticated_office_j", "modern_working_k", "elegant_business_m",
    "chic_professional_y", "career_chic_h", "city_elegant_seoul", "urban_business_k",
    "office_elegant_j", "working_chic_m", "career_modern_y", "professional_30s_kim",
    # 40대 여성 - 30명
    "graceful_40s", "timeless_beauty_k", "ageless_style_j", "classic_elegance",
    "mature_chic_lee", "forever_young_kim", "elegant_midlife", "stylish_40plus",
    "refined_beauty_m", "sophisticated_40s", "graceful_lady_j", "timeless_style_k",
    "ageless_beauty_m", "classic_chic_y", "mature_elegance_h", "forever_style_seoul",
    "elegant_40plus_k", "stylish_mature_j", "refined_40s_m", "sophisticated_lady_y",
    "graceful_chic_h", "timeless_elegance_seoul", "ageless_chic_k", "classic_style_j",
    "mature_refined_m", "forever_elegant_y", "elegant_classic_h", "stylish_ageless_seoul",
    "refined_mature_k", "sophisticated_timeless_j",
    # 20대 남성 - 25명
    "street_boy_kim", "urban_mens_style", "cool_guy_j", "trendy_man_lee",
    "fashion_bro_k", "style_guy_seoul", "mens_daily_look", "dapper_dude_j",
    "casual_mens_m", "hip_hop_style_k", "street_style_lee", "urban_cool_j",
    "trendy_boy_k", "fashion_guy_m", "style_dude_y", "mens_street_h",
    "dapper_style_seoul", "casual_cool_k", "hip_style_j", "street_fashion_m",
    "urban_dude_y", "trendy_cool_h", "fashion_street_seoul", "style_hip_k",
    "mens_trendy_j",
    # 30대 남성 - 15명
    "gentleman_style_k", "business_man_look", "smart_casual_m", "modern_man_j",
    "mature_mens_style", "classy_guy_lee", "gentleman_look_h", "business_style_seoul",
    "smart_man_k", "modern_gentleman_j", "mature_style_m", "classy_business_y",
    "gentleman_chic_h", "business_casual_seoul", "smart_gentleman_k",
]

def generate_trendsetter_usernames(count: int) -> list:
    """Trendsetter 유저네임 동적 생성"""
    usernames = list(TRENDSETTER_USERNAME_BASES)
    # 부족하면 번호 붙여서 추가
    idx = 0
    while len(usernames) < count:
        base = TRENDSETTER_USERNAME_BASES[idx % len(TRENDSETTER_USERNAME_BASES)]
        usernames.append(f"{base}_{len(usernames)}")
        idx += 1
    return usernames[:count]

TRENDSETTER_USERNAMES = generate_trendsetter_usernames(180)

# 트렌드세터 바이오 - 연령대/성별별
TRENDSETTER_BIOS_FEMALE_20 = [  # 20대 여성
    "fashion | daily",
    "ootd 📸",
    "seoul 🇰🇷 | 20s",
    "style ✨ MZ감성",
    "fashion lover 💕",
    "daily look | 대학생",
    "📍서울 | Y2K style",
    "✉️ DM for collab",
    "lifestyle | 취준생",
    "fashion & beauty 🌸",
    "ootd diary 📓",
    "minimal style",
    "힙한 감성 ✌️",
    "스트릿 패션 🔥",
    "캠퍼스 룩 📚",
]

TRENDSETTER_BIOS_FEMALE_30 = [  # 30대 여성
    "30대 직장인 | daily",
    "워킹맘 일상 💼",
    "오피스룩 전문 👔",
    "30s fashion | 서울",
    "career woman style",
    "modern & chic ✨",
    "직장인 데일리 👩‍💼",
    "30대 여자의 패션 🌷",
    "세련된 일상 🏙️",
    "출근룩 | 퇴근룩 👠",
    "비즈니스 캐주얼 💄",
    "30대 맞팔환영 🤝",
]

TRENDSETTER_BIOS_FEMALE_40 = [  # 40대 여성
    "40대의 품격있는 일상",
    "elegant style | 40s",
    "timeless beauty ✨",
    "classic fashion 🌹",
    "graceful 40s | seoul",
    "40대 여자의 멋 💎",
    "우아한 일상 🎀",
    "세월을 이기는 스타일",
    "품위있는 패션 👗",
    "에이지리스 뷰티 💫",
]

TRENDSETTER_BIOS_MALE_20 = [  # 20대 남성
    "mens fashion 🔥",
    "street style | 20s",
    "힙합 감성 🎤",
    "남자 데일리 👟",
    "urban style | seoul",
    "스트릿 패션 🛹",
    "20대 남자 ootd",
    "casual & cool 😎",
]

TRENDSETTER_BIOS_MALE_30 = [  # 30대 남성
    "30대 남자 패션 👔",
    "젠틀맨 스타일 🎩",
    "비즈니스 캐주얼 💼",
    "modern gentleman",
    "30s mens style",
    "클래식 & 모던 🖤",
]

# 기본 바이오 (혼합)
TRENDSETTER_BIOS = (TRENDSETTER_BIOS_FEMALE_20 + TRENDSETTER_BIOS_FEMALE_30 +
                   TRENDSETTER_BIOS_FEMALE_40 + TRENDSETTER_BIOS_MALE_20 +
                   TRENDSETTER_BIOS_MALE_30)

TRENDSETTER_CAPTIONS = [
    "",
    "#ootd",
    "✨",
    "#dailylook",
    "",
    "🖤",
    "#fashion",
    "",
    "#style",
    "",
]

# 스타일 및 분석 데이터
DOMINANT_STYLES = ["luxury", "natural", "trendy", "colorful", "minimal"]
SUB_STYLES = ["modern", "classic", "casual", "street", "feminine", "chic", "bohemian", "preppy"]
COLOR_PALETTES = ["warm_gold", "neutral_warm", "neutral_cool", "monochrome", "pastel_pop", "earth_tone", "black_gold"]
AESTHETIC_TAGS = [
    "스트릿패션", "Y2K", "캐주얼", "레이어드", "데님", "미니멀", "오버사이즈",
    "크롭탑", "와이드팬츠", "플리츠", "니트", "자켓", "코트룩", "원피스",
    "블레이저", "하이웨이스트", "빈티지", "모던시크", "페미닌", "보헤미안"
]
HAIR_STYLE_TAGS = [
    "웨이브", "레이어드컷", "내추럴브라운", "히피펌", "C컬", "볼륨펌",
    "애쉬브라운", "허쉬컷", "롱헤어", "단발", "염색", "하이라이트"
]
# 연령대/성별별 VIBES
VIBES_FEMALE_20 = [
    "힙하고 트렌디한 MZ세대 패션 인플루언서",
    "컬러풀하고 개성 넘치는 Y2K 감성 크리에이터",
    "미니멀하고 깔끔한 모던 시크 스타일",
    "스트릿과 하이패션을 넘나드는 트렌드세터",
    "대학생 감성의 캠퍼스 패션 인플루언서",
    "SNS 트렌드를 선도하는 20대 여성 크리에이터",
]

VIBES_FEMALE_30 = [
    "세련된 30대 직장인의 오피스 스타일리스트",
    "일과 삶의 균형을 보여주는 워킹맘 인플루언서",
    "프로페셔널하면서도 트렌디한 30대 패셔니스타",
    "30대 여성의 세련된 일상을 공유하는 크리에이터",
    "비즈니스 캐주얼의 정석을 보여주는 인플루언서",
    "커리어와 스타일을 동시에 잡은 30대 여성",
]

VIBES_FEMALE_40 = [
    "고급스럽고 세련된 럭셔리 무드의 스타일리스트",
    "자연스럽고 편안한 데일리룩을 선보이는 인플루언서",
    "품격있는 40대의 우아한 스타일을 보여주는 크리에이터",
    "클래식과 현대를 믹스한 에이지리스 패셔니스타",
    "세월을 이기는 아름다움을 보여주는 40대 인플루언서",
    "우아하고 단정한 중년 여성 패션 리더",
]

VIBES_MALE_20 = [
    "스트릿 패션을 선도하는 20대 남성 인플루언서",
    "힙합 감성의 트렌디한 남성 크리에이터",
    "캐주얼하면서도 세련된 20대 남자 스타일리스트",
    "MZ세대 남성의 데일리룩을 보여주는 인플루언서",
]

VIBES_MALE_30 = [
    "젠틀맨 스타일의 30대 남성 패셔니스타",
    "비즈니스 캐주얼의 정석을 보여주는 직장인 인플루언서",
    "클래식하면서도 모던한 30대 남성 스타일리스트",
    "프로페셔널한 이미지의 30대 남성 크리에이터",
]

VIBES = VIBES_FEMALE_20 + VIBES_FEMALE_30 + VIBES_FEMALE_40 + VIBES_MALE_20 + VIBES_MALE_30

# Expert 이미지 분석 데이터
SPECIALTIES = ["염색", "펌", "커트", "클리닉", "두피케어", "웨딩헤어", "남성커트", "탈모케어"]
TECHNIQUES = ["C컬펌", "히피펌", "볼륨펌", "레이어드컷", "애쉬염색", "하이톤염색", "클리닉트리트먼트", "두피스케일링"]
CLIENT_HAIR_TYPES = ["웨이브", "스트레이트", "볼륨펌", "C컬", "히피펌", "레이어드", "단발", "롱헤어"]
COLOR_SPECIALTIES_LIST = ["애쉬", "브라운", "하이톤", "로우톤", "그레이", "핑크", "레드", "베이지"]
WORK_ENVIRONMENTS = ["salon", "home_salon", "freelance", "academy"]


def generate_post_id():
    """Instagram 스타일 게시물 ID 생성"""
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
    return "".join(random.choices(chars, k=11))


def generate_permalink(username: str, post_id: str) -> str:
    """Instagram permalink 생성"""
    return f"https://www.instagram.com/reel/{post_id}/"


def generate_media_url(post_id: str) -> str:
    """Instagram media URL 생성 (실제로는 CDN URL)"""
    return f"https://scontent.cdninstagram.com/v/t51.2885-15/{post_id}.jpg"


def generate_timestamp(days_ago: int) -> str:
    """ISO 8601 형식 타임스탬프 생성"""
    dt = datetime.now() - timedelta(days=days_ago)
    return dt.strftime("%Y-%m-%dT%H:%M:%S+0000")


def generate_expert_posts(num_posts: int = 10, is_fake: bool = False) -> list:
    """
    Expert형 인플루언서의 릴스 게시물 생성

    Args:
        num_posts: 게시물 수
        is_fake: 허수 계정 여부 (views 대비 likes/comments 비율 조작)

    Returns:
        게시물 리스트 (views 필드 포함)
    """
    posts = []
    for i in range(num_posts):
        post_id = generate_post_id()

        # 조회수 기반으로 좋아요/댓글 생성 (정상 비율: likes 2-8%, comments 0.1-1%)
        views = random.randint(10000, 80000)

        if is_fake:
            # 허수 계정: 비정상적으로 높은 좋아요 비율 (15-30%)
            likes = int(views * random.uniform(0.15, 0.30))
            comments = int(views * random.uniform(0.03, 0.08))
        else:
            # 정상 계정: 적정 비율
            likes = int(views * random.uniform(0.02, 0.08))
            comments = int(views * random.uniform(0.001, 0.01))

        posts.append({
            "caption": random.choice(EXPERT_CAPTIONS),
            "views": views,
            "likes": likes,
            "comments": comments,
            "media_type": "VIDEO",  # 릴스
            "timestamp": generate_timestamp(i * random.randint(2, 5)),
            "media_url": generate_media_url(post_id),
            "permalink": generate_permalink("expert", post_id)
        })
    return posts


def generate_trendsetter_posts(num_posts: int = 10, is_fake: bool = False, is_viewbot: bool = False) -> list:
    """
    Trendsetter형 인플루언서의 릴스 게시물 생성

    Args:
        num_posts: 게시물 수
        is_fake: 허수 계정 여부 (좋아요 구매 의심)
        is_viewbot: 뷰봇 계정 여부 (조회수 대비 참여율 극히 낮음)

    Returns:
        게시물 리스트 (views 필드 포함)
    """
    posts = []
    for i in range(num_posts):
        post_id = generate_post_id()

        # 트렌드세터는 조회수가 높음
        views = random.randint(50000, 500000)

        if is_viewbot:
            # 뷰봇 계정: 조회수는 높지만 참여율 극히 낮음 (likes < 1%)
            likes = int(views * random.uniform(0.001, 0.008))
            comments = int(views * random.uniform(0.0001, 0.0005))
        elif is_fake:
            # 좋아요 구매 계정: 비정상적으로 높은 좋아요 비율 (20-35%)
            likes = int(views * random.uniform(0.20, 0.35))
            comments = int(views * random.uniform(0.04, 0.10))
        else:
            # 정상 계정: 트렌드세터 적정 비율 (likes 3-12%, comments 0.3-2%)
            likes = int(views * random.uniform(0.03, 0.12))
            comments = int(views * random.uniform(0.003, 0.02))

        posts.append({
            "caption": random.choice(TRENDSETTER_CAPTIONS),
            "views": views,
            "likes": likes,
            "comments": comments,
            "media_type": "VIDEO",  # 릴스
            "timestamp": generate_timestamp(i * random.randint(1, 3)),
            "media_url": generate_media_url(post_id),
            "permalink": generate_permalink("trendsetter", post_id)
        })
    return posts


def generate_expert_text_analysis(bio: str, captions: list) -> dict:
    """
    Expert형 텍스트 분석 결과 생성 (Primary 분석)

    Expert는 bio와 caption에 정보가 풍부하므로 텍스트 분석이 핵심.
    - bio에서 자격증, 경력, 전문 분야 추출
    - caption에서 시술 키워드, 레시피, 기법 추출
    """
    # bio에서 전문 분야 추출
    specialties_from_bio = []
    certifications = []

    bio_text = bio.lower()
    for specialty in SPECIALTIES:
        if specialty in bio_text or specialty in bio:
            specialties_from_bio.append(specialty)

    # 자격증/경력 키워드
    cert_keywords = ["원장", "디렉터", "년차", "경력", "자격증", "교육", "아카데미"]
    for kw in cert_keywords:
        if kw in bio:
            certifications.append(kw)

    # caption에서 시술 키워드 추출
    techniques_from_caption = []
    all_captions = " ".join(captions)
    for tech in TECHNIQUES:
        if tech in all_captions:
            techniques_from_caption.append(tech)

    if not techniques_from_caption:
        techniques_from_caption = random.sample(TECHNIQUES, k=random.randint(2, 4))

    return {
        "analysis_type": "text_primary",  # 텍스트 분석이 주력
        "specialties_from_bio": specialties_from_bio if specialties_from_bio else random.sample(SPECIALTIES, k=2),
        "certifications_detected": certifications,
        "techniques_from_caption": techniques_from_caption,
        "caption_detail_level": "high",  # Expert는 caption이 상세함
        "text_confidence": round(random.uniform(0.8, 0.95), 2)
    }


def generate_expert_image_analysis(bio: str) -> dict:
    """
    Expert형 이미지 분석 결과 생성 (Secondary 분석 - 검증/보완용)

    Expert는 텍스트 정보가 풍부하므로 이미지 분석은 보조적 역할:
    - bio에서 언급된 전문 분야가 실제 시술 이미지와 일치하는지 검증
    - 텍스트에 없는 추가 전문 분야 발견
    """
    # bio에서 키워드 추출하여 verified_specialties 생성
    verified = []
    additional = []

    bio_lower = bio.lower()
    for specialty in SPECIALTIES:
        if specialty in bio_lower or specialty in bio:
            verified.append(specialty)
        elif random.random() < 0.3:
            additional.append(specialty)

    if not verified:
        verified = random.sample(SPECIALTIES, k=random.randint(1, 3))
    if not additional:
        additional = random.sample([s for s in SPECIALTIES if s not in verified], k=random.randint(0, 2))

    return {
        "analysis_type": "image_secondary",  # 이미지 분석은 보조
        "verified_specialties": verified[:3],  # bio 정보 검증됨
        "additional_specialties": additional[:2],  # 이미지에서 추가 발견
        "signature_techniques": random.sample(TECHNIQUES, k=random.randint(2, 4)),
        "client_hair_types": random.sample(CLIENT_HAIR_TYPES, k=random.randint(2, 4)),
        "color_specialties": random.sample(COLOR_SPECIALTIES_LIST, k=random.randint(2, 3)),
        "work_environment": random.choice(WORK_ENVIRONMENTS),
        "content_quality_score": round(random.uniform(0.7, 0.95), 2),
        "expertise_confidence": round(random.uniform(0.7, 0.95), 2)
    }


def generate_trendsetter_text_analysis(bio: str, captions: list) -> dict:
    """
    Trendsetter형 텍스트 분석 결과 생성 (Secondary 분석 - 보조)

    Trendsetter는 bio와 caption이 거의 비어있으므로 텍스트 분석은 보조적 역할:
    - bio가 간단하므로 추출 가능한 정보 제한적
    - caption도 해시태그 위주로 간략함
    - 텍스트에서 얻을 수 있는 정보가 적어 신뢰도 낮음
    """
    # bio에서 추출 가능한 키워드 (매우 제한적)
    keywords_from_bio = []
    style_hints = []

    bio_lower = bio.lower()
    style_keywords = ["fashion", "style", "ootd", "daily", "minimal", "lifestyle"]
    for kw in style_keywords:
        if kw in bio_lower:
            keywords_from_bio.append(kw)

    # caption에서 해시태그 추출 (대부분 간단함)
    hashtags = []
    for caption in captions:
        if "#" in caption:
            tags = [word.strip() for word in caption.split() if word.startswith("#")]
            hashtags.extend(tags)

    return {
        "analysis_type": "text_secondary",  # 텍스트 분석은 보조
        "keywords_from_bio": keywords_from_bio if keywords_from_bio else ["lifestyle"],
        "hashtags_from_caption": list(set(hashtags))[:5],
        "caption_detail_level": "low",  # Trendsetter는 caption이 간략함
        "extractable_info": "minimal",  # 추출 가능한 정보 제한적
        "text_confidence": round(random.uniform(0.2, 0.5), 2)  # 낮은 신뢰도
    }


def generate_trendsetter_image_analysis() -> dict:
    """
    Trendsetter형 이미지 분석 결과 생성 (Primary 분석 - 핵심)

    Trendsetter는 텍스트 정보가 부족하므로 이미지 분석이 핵심:
    - 스타일, 컬러, 미학적 태그는 이미지에서 직접 추출
    - 헤어 스타일도 이미지 분석으로만 파악 가능
    - 브랜드 매칭을 위한 모든 핵심 정보가 이미지에서 도출됨
    """
    return {
        "analysis_type": "image_primary",  # 이미지 분석이 주력
        "dominant_style": random.choice(DOMINANT_STYLES),
        "sub_styles": random.sample(SUB_STYLES, k=2),
        "color_palette": random.choice(COLOR_PALETTES),
        "aesthetic_tags": random.sample(AESTHETIC_TAGS, k=5),
        "hair_style_tags": random.sample(HAIR_STYLE_TAGS, k=random.randint(2, 4)),
        "vibe": random.choice(VIBES),
        "professionalism_score": round(random.uniform(0.3, 0.6), 2),
        "trend_relevance_score": round(random.uniform(0.8, 0.95), 2),
        "image_confidence": round(random.uniform(0.85, 0.98), 2)  # 높은 신뢰도
    }


def generate_expert_influencer(username: str, index: int) -> dict:
    """
    Expert형 인플루언서 데이터 생성

    분석 전략:
    - text_analysis: PRIMARY (bio/caption이 풍부하므로 핵심 정보원)
    - image_analysis: SECONDARY (텍스트 정보 검증 및 보완용)
    """
    bio = EXPERT_BIOS[index % len(EXPERT_BIOS)]
    posts = generate_expert_posts(num_posts=10)
    captions = [post["caption"] for post in posts]

    return {
        "username": username,
        "influencer_type": "expert",
        "followers": random.randint(30000, 150000),
        "bio": bio,
        "media_count": random.randint(200, 800),
        "recent_posts": posts,
        "audience_countries": {
            "KR": round(random.uniform(0.85, 0.95), 2),
            "US": round(random.uniform(0.01, 0.05), 2),
            "JP": round(random.uniform(0.01, 0.05), 2),
            "OTHER": round(random.uniform(0.01, 0.05), 2)
        },
        "avg_upload_interval_days": round(random.uniform(2.0, 5.0), 1),
        "analysis_strategy": {
            "primary": "text",
            "secondary": "image",
            "reason": "Expert는 bio와 caption에 전문 정보가 풍부함"
        },
        "text_analysis": generate_expert_text_analysis(bio, captions),
        "image_analysis": generate_expert_image_analysis(bio)
    }


def generate_trendsetter_influencer(username: str, index: int) -> dict:
    """
    Trendsetter형 인플루언서 데이터 생성

    분석 전략:
    - image_analysis: PRIMARY (bio/caption이 비어있어 이미지에서 정보 추출)
    - text_analysis: SECONDARY (해시태그 등 보조 정보만 추출)
    """
    bio = TRENDSETTER_BIOS[index % len(TRENDSETTER_BIOS)]
    posts = generate_trendsetter_posts(num_posts=10)
    captions = [post["caption"] for post in posts]

    return {
        "username": username,
        "influencer_type": "trendsetter",
        "followers": random.randint(100000, 500000),
        "bio": bio,
        "media_count": random.randint(300, 1000),
        "recent_posts": posts,
        "audience_countries": {
            "KR": round(random.uniform(0.70, 0.85), 2),
            "US": round(random.uniform(0.05, 0.10), 2),
            "JP": round(random.uniform(0.03, 0.08), 2),
            "OTHER": round(random.uniform(0.05, 0.10), 2)
        },
        "avg_upload_interval_days": round(random.uniform(1.0, 3.0), 1),
        "analysis_strategy": {
            "primary": "image",
            "secondary": "text",
            "reason": "Trendsetter는 bio/caption이 간략하여 이미지 분석이 핵심"
        },
        "text_analysis": generate_trendsetter_text_analysis(bio, captions),
        "image_analysis": generate_trendsetter_image_analysis()
    }


def generate_raw_data(num_experts: int = 50, num_trendsetters: int = 50, fake_ratio: float = 0.1) -> dict:
    """
    크롤러 형식의 raw 데이터 생성 (분류/분석 없음)

    실제 Instagram API에서 수집하는 형태와 동일
    - influencer_type, analysis_strategy, text_analysis, image_analysis 없음
    - Processor에서 처리할 수 있는 형태

    Args:
        num_experts: Expert 인플루언서 수
        num_trendsetters: Trendsetter 인플루언서 수
        fake_ratio: 허수 계정 비율 (기본 10%)

    Returns:
        raw 인플루언서 데이터 딕셔너리
    """
    influencers = []

    # 허수 계정 수 계산
    num_fake_experts = int(num_experts * fake_ratio)
    num_fake_trendsetters = int(num_trendsetters * fake_ratio)
    num_viewbot_trendsetters = int(num_trendsetters * fake_ratio / 2)

    # Expert 생성 (일부 허수) - raw 형식
    for i, username in enumerate(EXPERT_USERNAMES[:num_experts]):
        is_fake = i < num_fake_experts
        bio = EXPERT_BIOS[i % len(EXPERT_BIOS)]
        posts = generate_expert_posts(num_posts=10, is_fake=is_fake)

        influencer = {
            "username": username,
            "followers": random.randint(30000, 150000),
            "bio": bio,
            "media_count": random.randint(200, 800),
            "recent_posts": posts,
            "audience_countries": {
                "KR": round(random.uniform(0.85, 0.95), 2),
                "US": round(random.uniform(0.01, 0.05), 2),
                "JP": round(random.uniform(0.01, 0.05), 2),
                "OTHER": round(random.uniform(0.01, 0.05), 2)
            },
            "avg_upload_interval_days": round(random.uniform(2.0, 5.0), 1)
        }
        if is_fake:
            influencer["_test_label"] = "fake_likes"
        influencers.append(influencer)

    # Trendsetter 생성 (일부 허수/뷰봇) - raw 형식
    for i, username in enumerate(TRENDSETTER_USERNAMES[:num_trendsetters]):
        is_viewbot = i < num_viewbot_trendsetters
        is_fake = num_viewbot_trendsetters <= i < (num_viewbot_trendsetters + num_fake_trendsetters)

        bio = TRENDSETTER_BIOS[i % len(TRENDSETTER_BIOS)]
        posts = generate_trendsetter_posts(num_posts=10, is_fake=is_fake, is_viewbot=is_viewbot)

        influencer = {
            "username": username,
            "followers": random.randint(100000, 500000),
            "bio": bio,
            "media_count": random.randint(300, 1000),
            "recent_posts": posts,
            "audience_countries": {
                "KR": round(random.uniform(0.70, 0.85), 2),
                "US": round(random.uniform(0.05, 0.10), 2),
                "JP": round(random.uniform(0.03, 0.08), 2),
                "OTHER": round(random.uniform(0.05, 0.10), 2)
            },
            "avg_upload_interval_days": round(random.uniform(1.0, 3.0), 1)
        }

        if is_viewbot:
            influencer["_test_label"] = "viewbot"
        elif is_fake:
            influencer["_test_label"] = "fake_likes"

        influencers.append(influencer)

    return {
        "influencers": influencers,
        "metadata": {
            "crawled_at": datetime.now().isoformat(),
            "total_count": len(influencers),
            "posts_per_influencer": 10,
            "status": "raw",
            "note": "크롤러에서 수집한 raw 데이터 (Processor에서 분류/분석 필요)"
        }
    }


def get_fis_score_and_verdict(category: str) -> tuple:
    """
    다양한 FIS 점수 분포 생성

    분포:
    - 신뢰 가능 (80-98): 60%
    - 주의 필요 (60-79): 25%
    - 위험 (40-59): 15%
    """
    if category == 'high':  # 신뢰 가능
        score = round(random.uniform(80, 98), 1)
        verdict = "신뢰 가능"
    elif category == 'medium':  # 주의 필요
        score = round(random.uniform(60, 79), 1)
        verdict = "주의 필요"
    else:  # low - 위험
        score = round(random.uniform(40, 59), 1)
        verdict = "위험"
    return score, verdict


def get_random_fis_category() -> str:
    """FIS 카테고리 랜덤 선택 (분포에 따라)"""
    r = random.random()
    if r < 0.60:
        return 'high'
    elif r < 0.85:
        return 'medium'
    else:
        return 'low'


def determine_target_demographics(index: int, total: int, inf_type: str) -> dict:
    """
    인플루언서의 타겟 연령대/성별 결정

    Expert:
    - 20대 여성 타겟: 30%
    - 30대 여성 타겟: 25%
    - 40대 여성 타겟: 15%
    - 남성 타겟: 20%
    - 유니섹스(두피/탈모): 10%

    Trendsetter:
    - 20대 여성: 35%
    - 30대 여성: 25%
    - 40대 여성: 15%
    - 20대 남성: 15%
    - 30대 남성: 10%
    """
    r = random.random()

    if inf_type == 'expert':
        if r < 0.30:
            return {'target_gender': 'female', 'target_age': '20대'}
        elif r < 0.55:
            return {'target_gender': 'female', 'target_age': '30대'}
        elif r < 0.70:
            return {'target_gender': 'female', 'target_age': '40대'}
        elif r < 0.90:
            return {'target_gender': 'male', 'target_age': random.choice(['20대', '30대'])}
        else:
            return {'target_gender': 'unisex', 'target_age': random.choice(['30대', '40대'])}
    else:  # trendsetter
        if r < 0.35:
            return {'target_gender': 'female', 'target_age': '20대'}
        elif r < 0.60:
            return {'target_gender': 'female', 'target_age': '30대'}
        elif r < 0.75:
            return {'target_gender': 'female', 'target_age': '40대'}
        elif r < 0.90:
            return {'target_gender': 'male', 'target_age': '20대'}
        else:
            return {'target_gender': 'male', 'target_age': '30대'}


def get_mood_for_demographics(target_gender: str, target_age: str) -> str:
    """타겟 인구통계에 맞는 무드 선택"""
    moods = {
        ('female', '20대'): ['트렌디한', '힙한', 'Y2K 감성의', '컬러풀한', '스트릿한', '캐주얼한'],
        ('female', '30대'): ['세련된', '모던한', '프로페셔널한', '우아한', '시크한', '클래시한'],
        ('female', '40대'): ['고급스러운', '우아한', '클래식한', '품격있는', '단정한', '세련된'],
        ('male', '20대'): ['힙한', '스트릿한', '캐주얼한', '트렌디한', '쿨한', '댄디한'],
        ('male', '30대'): ['댄디한', '프로페셔널한', '클래식한', '모던한', '세련된', '젠틀한'],
        ('unisex', '30대'): ['전문적인', '신뢰감 있는', '클리닉한', '케어 전문'],
        ('unisex', '40대'): ['전문적인', '케어 전문', '클리닉한', '신뢰감 있는'],
    }
    key = (target_gender, target_age)
    return random.choice(moods.get(key, ['트렌디한', '세련된', '고급스러운']))


def generate_processed_data(num_experts: int = 70, num_trendsetters: int = 80) -> dict:
    """
    Processor에서 처리된 형태의 데이터 생성 (분류/분석 완료)

    다양한 FIS 분포:
    - 신뢰 가능 (80-98): 60%
    - 주의 필요 (60-79): 25%
    - 위험 (40-59): 15%

    Args:
        num_experts: Expert 인플루언서 수 (기본 70명)
        num_trendsetters: Trendsetter 인플루언서 수 (기본 80명)

    Returns:
        처리된 인플루언서 데이터 딕셔너리
    """
    influencers = []

    # 통계 추적
    stats = {
        'fis_high': 0, 'fis_medium': 0, 'fis_low': 0,
        'female_20': 0, 'female_30': 0, 'female_40': 0,
        'male_20': 0, 'male_30': 0, 'unisex': 0
    }

    # Expert 생성 - 다양한 타겟과 FIS 분포
    for i, username in enumerate(EXPERT_USERNAMES[:num_experts]):
        # 타겟 인구통계 결정
        demographics = determine_target_demographics(i, num_experts, 'expert')
        target_gender = demographics['target_gender']
        target_age = demographics['target_age']

        # 통계 업데이트
        if target_gender == 'unisex':
            stats['unisex'] += 1
        else:
            stats[f'{target_gender}_{target_age[:2]}'] += 1

        # 바이오 선택 (타겟에 맞게)
        if target_gender == 'male':
            bio = random.choice(EXPERT_BIOS_MALE)
        elif target_age == '40대' or target_age == '30대':
            bio = random.choice(EXPERT_BIOS_FEMALE_MATURE)
        elif target_gender == 'unisex':
            bio = random.choice(EXPERT_BIOS_SCALP)
        else:
            bio = random.choice(EXPERT_BIOS_FEMALE_YOUNG)

        posts = generate_expert_posts(num_posts=10, is_fake=False)
        captions = [post["caption"] for post in posts]

        # 다양한 FIS 점수 분포
        fis_category = get_random_fis_category()
        fis_score, fis_verdict = get_fis_score_and_verdict(fis_category)
        stats[f'fis_{fis_category}'] += 1

        # 무드 결정
        main_mood = get_mood_for_demographics(target_gender, target_age)

        influencer = {
            "username": username,
            "influencer_type": "expert",
            "followers": random.randint(30000, 200000),
            "bio": bio,
            "classification_confidence": round(random.uniform(0.85, 1.0), 2),
            "analysis_strategy": {
                "primary": "text",
                "secondary": "image",
                "reason": "Expert는 bio와 caption에 전문 정보가 풍부함"
            },
            "text_analysis": generate_expert_text_analysis(bio, captions),
            "image_analysis": {
                **generate_expert_image_analysis(bio),
                "target_gender": target_gender,
                "target_age": target_age,
                "main_mood": main_mood,
            },
            "fis": {
                "score": fis_score,
                "verdict": fis_verdict
            }
        }
        influencers.append(influencer)

    # Trendsetter 생성 - 다양한 타겟과 FIS 분포
    for i, username in enumerate(TRENDSETTER_USERNAMES[:num_trendsetters]):
        # 타겟 인구통계 결정
        demographics = determine_target_demographics(i, num_trendsetters, 'trendsetter')
        target_gender = demographics['target_gender']
        target_age = demographics['target_age']

        # 통계 업데이트
        if target_gender == 'unisex':
            stats['unisex'] += 1
        else:
            key = f'{target_gender}_{target_age[:2]}'
            if key in stats:
                stats[key] += 1

        # 바이오 선택 (타겟에 맞게)
        if target_gender == 'male':
            if target_age == '30대':
                bio = random.choice(TRENDSETTER_BIOS_MALE_30)
            else:
                bio = random.choice(TRENDSETTER_BIOS_MALE_20)
        elif target_age == '40대':
            bio = random.choice(TRENDSETTER_BIOS_FEMALE_40)
        elif target_age == '30대':
            bio = random.choice(TRENDSETTER_BIOS_FEMALE_30)
        else:
            bio = random.choice(TRENDSETTER_BIOS_FEMALE_20)

        posts = generate_trendsetter_posts(num_posts=10, is_fake=False, is_viewbot=False)
        captions = [post["caption"] for post in posts]

        # 다양한 FIS 점수 분포
        fis_category = get_random_fis_category()
        fis_score, fis_verdict = get_fis_score_and_verdict(fis_category)
        stats[f'fis_{fis_category}'] += 1

        # 무드와 바이브 결정
        main_mood = get_mood_for_demographics(target_gender, target_age)

        # 바이브 선택 (연령대/성별에 맞게)
        if target_gender == 'male':
            if target_age == '30대':
                vibe = random.choice(VIBES_MALE_30)
            else:
                vibe = random.choice(VIBES_MALE_20)
        elif target_age == '40대':
            vibe = random.choice(VIBES_FEMALE_40)
        elif target_age == '30대':
            vibe = random.choice(VIBES_FEMALE_30)
        else:
            vibe = random.choice(VIBES_FEMALE_20)

        influencer = {
            "username": username,
            "influencer_type": "trendsetter",
            "followers": random.randint(50000, 500000),
            "bio": bio,
            "classification_confidence": round(random.uniform(0.85, 1.0), 2),
            "analysis_strategy": {
                "primary": "image",
                "secondary": "text",
                "reason": "Trendsetter는 bio/caption이 간략하여 이미지 분석이 핵심"
            },
            "text_analysis": generate_trendsetter_text_analysis(bio, captions),
            "image_analysis": {
                **generate_trendsetter_image_analysis(),
                "target_gender": target_gender,
                "target_age": target_age,
                "main_mood": main_mood,
                "vibe": vibe,
            },
            "fis": {
                "score": fis_score,
                "verdict": fis_verdict
            }
        }

        influencers.append(influencer)

    return {
        "influencers": influencers,
        "metadata": {
            "processed_at": datetime.now().isoformat(),
            "total_count": len(influencers),
            "expert_count": num_experts,
            "trendsetter_count": num_trendsetters,
            "status": "processed",
            "schema_version": "5.0",
            "note": "다양한 FIS 분포와 타겟 인구통계를 포함한 확장 데이터",
            "raw_data_ref": "influencers_raw.json",
            "statistics": stats
        }
    }


def main():
    """메인 실행"""
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 설정 (300명 데이터셋)
    NUM_EXPERTS = 120
    NUM_TRENDSETTERS = 180

    # 1. Raw 데이터 생성 (크롤러 형식)
    print("=" * 60)
    print("1. Raw 데이터 생성 (크롤러 형식)")
    print("=" * 60)

    raw_data = generate_raw_data(num_experts=NUM_EXPERTS, num_trendsetters=NUM_TRENDSETTERS)
    raw_path = data_dir / "influencers_raw.json"

    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Raw 데이터 생성 완료: {raw_path}")
    print(f"   - 총 인플루언서: {raw_data['metadata']['total_count']}명")
    print(f"   - 상태: {raw_data['metadata']['status']}")

    # 2. Processed 데이터 생성 (분류/분석 완료 형식)
    print("\n" + "=" * 60)
    print("2. Processed 데이터 생성 (분류/분석 완료)")
    print("=" * 60)

    processed_data = generate_processed_data(num_experts=NUM_EXPERTS, num_trendsetters=NUM_TRENDSETTERS)
    processed_path = data_dir / "influencers_data.json"

    with open(processed_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Processed 데이터 생성 완료: {processed_path}")
    print(f"   - 총 인플루언서: {processed_data['metadata']['total_count']}명")
    print(f"   - Expert: {processed_data['metadata']['expert_count']}명")
    print(f"   - Trendsetter: {processed_data['metadata']['trendsetter_count']}명")
    print(f"   - 스키마 버전: {processed_data['metadata']['schema_version']}")

    # 통계 출력
    stats = processed_data['metadata'].get('statistics', {})
    print("\n📊 FIS 점수 분포:")
    print(f"   - 신뢰 가능 (80-98): {stats.get('fis_high', 0)}명")
    print(f"   - 주의 필요 (60-79): {stats.get('fis_medium', 0)}명")
    print(f"   - 위험 (40-59): {stats.get('fis_low', 0)}명")

    print("\n👥 타겟 인구통계 분포:")
    print(f"   - 20대 여성: {stats.get('female_20', 0)}명")
    print(f"   - 30대 여성: {stats.get('female_30', 0)}명")
    print(f"   - 40대 여성: {stats.get('female_40', 0)}명")
    print(f"   - 20대 남성: {stats.get('male_20', 0)}명")
    print(f"   - 30대 남성: {stats.get('male_30', 0)}명")
    print(f"   - 유니섹스: {stats.get('unisex', 0)}명")

    # Expert 샘플 출력
    print("\n📌 Expert 샘플:")
    for i in [0, 20, 40]:
        if i < len(processed_data["influencers"]):
            expert = processed_data["influencers"][i]
            if expert['influencer_type'] == 'expert':
                img = expert['image_analysis']
                print(f"   @{expert['username']} | {img.get('target_gender', 'N/A')} {img.get('target_age', 'N/A')} | FIS: {expert['fis']['score']} ({expert['fis']['verdict']})")

    # Trendsetter 샘플 출력
    print("\n📌 Trendsetter 샘플:")
    for i in range(NUM_EXPERTS, min(NUM_EXPERTS + 10, len(processed_data["influencers"]))):
        trendsetter = processed_data["influencers"][i]
        if trendsetter['influencer_type'] == 'trendsetter':
            img = trendsetter['image_analysis']
            print(f"   @{trendsetter['username']} | {img.get('target_gender', 'N/A')} {img.get('target_age', 'N/A')} | FIS: {trendsetter['fis']['score']} ({trendsetter['fis']['verdict']})")


if __name__ == "__main__":
    main()
