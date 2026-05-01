PRIOR_CACHE_VERSION = "quality_v6_visual_v2_masks"
PACKAGE_CACHE_VERSION = "quality_tensor_v14_visual_v2_masks"
CACHE_VERSION_FILE = ".cache_version"

MODEL_INPUT_CHANNELS = 31
MODEL_OUTPUT_CHANNELS = 16

BASE_PACKAGE_KEYS = [
    "inputs",
    "base_prior",
    "visual_prior",
    "target",
    "reference",
    "input_rgb",
    "mask",
    "glare_mask",
    "reflection_area_mask",
]

SHADOW_MASK_KEYS = [
    "scene_reflection_mask",
    "dark_reflection_mask",
    "shadow_veil_mask",
    "lowfreq_reflection_mask",
    "foreground_structure_guard",
]

PACKAGE_REQUIRED_KEYS = BASE_PACKAGE_KEYS + SHADOW_MASK_KEYS

DEBUG_PANEL_EXTRA_FIELDS = [
    "StrongGlareCore",
    "TakeoverMask",
    "SceneReflectionMask",
    "DarkReflectionMask",
    "ShadowVeilMask",
    "LowfreqReflectionMask",
    "ForegroundStructureGuard",
    "ReflectionConf",
    "LineBandConf",
    "TextReflectionConf",
    "TintConf",
    "LowfreqReflectionConf",
    "DarkShadowConf",
]

METRIC_BASE_FIELDS = [
    "scene",
    "group",
    "reference_l1",
    "reference_focus_l1",
    "reference_psnr",
    "reference_ssim",
    "prior_l1",
    "prior_focus_l1",
    "prior_psnr",
    "prior_ssim",
    "direct_l1",
    "direct_focus_l1",
    "direct_psnr",
    "direct_ssim",
    "raw_l1",
    "raw_focus_l1",
    "raw_psnr",
    "raw_ssim",
    "visual_l1",
    "visual_focus_l1",
    "visual_psnr",
    "visual_ssim",
    "direct_vs_prior_focus_improvement",
    "raw_vs_prior_focus_improvement",
    "visual_vs_prior_focus_improvement",
]

METRIC_EXTRA_FIELDS = [
    "highlight_region_l1",
    "glare_suppression_ratio",
    "line_band_suppression_ratio",
    "lowfreq_reflection_l1",
    "dark_shadow_region_l1",
    "scene_reflection_l1",
    "shadow_veil_l1",
    "signed_delta_effect_ratio",
    "lowfreq_suppression_ratio",
    "dark_shadow_recovery_ratio",
    "foreground_damage_ratio",
    "raw_visualplus_diff_in_shadow",
]

METRIC_PATH_FIELDS = [
    "comparison_path",
    "raw_result_path",
    "result_path",
    "mask_path",
    "mask_gray_path",
    "glare_mask_gray_path",
    "strong_glare_core_gray_path",
    "takeover_mask_path",
    "reflection_mask_gray_path",
    "debug_mask_panel_path",
    "visual_extreme_result_path",
    "direct_clean_path",
    "raw_minus_prior_path",
    "direct_clean_minus_prior_path",
    "visual_plus_minus_prior_path",
    "blend_gate_path",
    "safe_gate_path",
    "final_mask_path",
    "prediction_lowfreq_path",
    "target_lowfreq_path",
]

METRIC_FIELDS = METRIC_BASE_FIELDS + METRIC_EXTRA_FIELDS + METRIC_PATH_FIELDS

HARD_CASE_FAILURE_TYPES = [
    "line_band",
    "horizontal_band",
    "diagonal_band",
    "text_reflection",
    "green_tint",
    "yellow_tint",
    "scene_reflection",
    "tree_shadow",
    "building_reflection",
    "dark_reflection",
    "lowfreq_reflection",
    "shadow_veil",
    "reflection_shadow_mix",
    "under_suppressed_shadow",
    "missed_glare",
    "over_smooth",
    "prior_like_output",
    "visual_equals_raw",
]
