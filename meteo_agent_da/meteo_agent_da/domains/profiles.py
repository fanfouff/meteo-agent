from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class DomainProfile:
    name: str
    description: str
    observation_types: List[str]
    model_context: List[str]
    assimilation_methods: List[str]
    diagnostics: List[str]
    default_tools: List[str]
    verifier_checks: List[str] = field(default_factory=list)


_PROFILES: Dict[str, DomainProfile] = {
    "pasnet_satellite": DomainProfile(
        name="pasnet_satellite",
        description="FY-3F MWTS-III / ERA5 / PASNet-DA temperature-profile correction workflow.",
        observation_types=["satellite_radiance", "mwts_iii", "brightness_temperature"],
        model_context=["era5", "temperature_profile", "npz_split"],
        assimilation_methods=["ml_da", "profile_correction", "neural_increment"],
        diagnostics=["rmse", "mae", "vertical_profile", "paper_table"],
        default_tools=["sanity_check", "data_indexer", "pasnet_runner", "evaluator", "plotter", "paper_writer"],
        verifier_checks=["path", "command", "metric", "artifact", "cost"],
    ),
    "wrf_3dvar": DomainProfile(
        name="wrf_3dvar",
        description="WRF 3DVar experiment planning profile for future executable tool coverage.",
        observation_types=["conventional_obs", "satellite_radiance", "radar"],
        model_context=["wrf", "cycle_time", "domain", "background_error"],
        assimilation_methods=["3dvar", "background_error_covariance", "observation_operator"],
        diagnostics=["innovation", "forecast_rmse", "case_study"],
        default_tools=["sanity_check", "paper_writer"],
        verifier_checks=["artifact", "cost"],
    ),
    "enkf_cycle": DomainProfile(
        name="enkf_cycle",
        description="Ensemble Kalman filter cycling profile for spread-skill and sensitivity studies.",
        observation_types=["conventional_obs", "satellite_radiance", "gnss_ro"],
        model_context=["ensemble_forecast", "cycle_time", "localization"],
        assimilation_methods=["enkf", "letkf", "inflation", "hybrid"],
        diagnostics=["spread_skill", "crps", "rank_histogram"],
        default_tools=["sanity_check", "paper_writer"],
        verifier_checks=["artifact", "cost"],
    ),
    "radar_da": DomainProfile(
        name="radar_da",
        description="Radar reflectivity/radial-wind assimilation planning profile.",
        observation_types=["radar_reflectivity", "radial_wind"],
        model_context=["storm_case", "wrf", "convective_scale"],
        assimilation_methods=["3dvar", "enkf", "nudging"],
        diagnostics=["precipitation_skill", "case_map", "innovation"],
        default_tools=["sanity_check", "paper_writer"],
        verifier_checks=["artifact", "cost"],
    ),
    "gnss_ro": DomainProfile(
        name="gnss_ro",
        description="GNSS radio-occultation bending-angle and refractivity assimilation planning profile.",
        observation_types=["gnss_ro", "bending_angle", "refractivity"],
        model_context=["nwp_background", "vertical_level", "humidity_temperature_profile"],
        assimilation_methods=["3dvar", "4dvar", "enkf"],
        diagnostics=["o_minus_b", "o_minus_a", "vertical_profile"],
        default_tools=["sanity_check", "paper_writer"],
        verifier_checks=["artifact", "cost"],
    ),
    "conventional_obs": DomainProfile(
        name="conventional_obs",
        description="Surface, radiosonde, aircraft, and ship/buoy observation workflow profile.",
        observation_types=["surface_station", "radiosonde", "aircraft", "ship_buoy"],
        model_context=["reanalysis", "forecast_background", "quality_control"],
        assimilation_methods=["oi", "3dvar", "enkf"],
        diagnostics=["bias", "rmse", "time_series"],
        default_tools=["sanity_check", "paper_writer"],
        verifier_checks=["artifact", "cost"],
    ),
    "diagnostics_paper": DomainProfile(
        name="diagnostics_paper",
        description="Cross-method DA diagnostics, figure, table, and paper-writing profile.",
        observation_types=["mixed_obs"],
        model_context=["experiment_archive", "metric_artifact"],
        assimilation_methods=["method_comparison", "ablation", "ose_osse"],
        diagnostics=["rmse", "mae", "acc", "innovation", "figure_caption", "latex_table"],
        default_tools=["sanity_check", "evaluator", "plotter", "paper_writer"],
        verifier_checks=["metric", "artifact", "cost"],
    ),
}


def get_profile(name: str) -> DomainProfile:
    return _PROFILES.get(name, _PROFILES["pasnet_satellite"])


def list_profiles() -> List[DomainProfile]:
    return list(_PROFILES.values())
