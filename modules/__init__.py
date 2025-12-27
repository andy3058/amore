"""
AI Hair Influencer Curation Agent - Modules
"""

from .taxonomy import (
    analyze_influencer,
    classify_influencer,
    batch_classify,
    get_role_vector,
    extract_text_features
)

from .fis_engine import (
    calculate_fis_score,
    batch_calculate_fis,
    filter_by_fis,
    calculate_view_variability_score,
    calculate_engagement_asymmetry_score,
    calculate_comment_entropy_score,
    calculate_activity_stability_score,
    calculate_geographic_consistency_score
)

from .brand_analyzer import (
    analyze_brand,
    create_brand_vector,
    get_matching_criteria,
    analyze_aesthetic_style,
    analyze_product_type,
    analyze_target_audience
)

from .matcher import (
    match_influencers,
    get_full_recommendations,
    generate_recommendation_reason,
    calculate_match_score,
    cosine_similarity
)

from .image_analyzer import (
    ImageAnalyzer,
    analyze_influencer_style,
    get_visual_style_vector
)

__all__ = [
    # Taxonomy
    'analyze_influencer',
    'classify_influencer',
    'batch_classify',
    'get_role_vector',
    'extract_text_features',

    # FIS Engine
    'calculate_fis_score',
    'batch_calculate_fis',
    'filter_by_fis',
    'calculate_view_variability_score',
    'calculate_engagement_asymmetry_score',
    'calculate_comment_entropy_score',
    'calculate_activity_stability_score',
    'calculate_geographic_consistency_score',

    # Brand Analyzer
    'analyze_brand',
    'create_brand_vector',
    'get_matching_criteria',
    'analyze_aesthetic_style',
    'analyze_product_type',
    'analyze_target_audience',

    # Matcher
    'match_influencers',
    'get_full_recommendations',
    'generate_recommendation_reason',
    'calculate_match_score',
    'cosine_similarity',

    # Image Analyzer
    'ImageAnalyzer',
    'analyze_influencer_style',
    'get_visual_style_vector'
]
