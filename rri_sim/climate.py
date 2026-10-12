"""Climate-change scenarios for flood simulations.

Future-climate rainfall is represented by *scaling the observed event* (the
approach of MLIT's design-rainfall change factors), optionally with a
spatial change-factor field derived from climate-model ensembles:

  rain_factor      multiplier of the event rainfall total (per cell)
  intensity_gamma  1.0 = uniform scaling.  > 1 concentrates the increase in
                   the heaviest hours (hourly extremes are projected to grow
                   faster than event totals): r' = c * r**gamma, with c chosen
                   per cell so that the event total equals rain_factor x the
                   original total.
  factor_field     optional (ny, nx) multiplier replacing rain_factor, e.g. the
                   ratio future/past of event-maximum rainfall from d4PDF /
                   気候予測データセット2022 (DIAS; requires registration).
  sea_level_rise_m added to the tidal stage boundary.
  apply_to_spinup  scale the antecedent (spin-up) rain as well (default False:
                   the antecedent wetness is that of the observed event).

Official presets (see docs/07_climate.md):
  降雨量変化倍率  国土交通省「気候変動を踏まえた治水計画のあり方」提言（改訂版, 2021）
      2℃上昇: 北海道・九州北西部 1.15, その他 1.1
      4℃上昇: 北海道・九州北西部 1.4,  その他 1.2   （全国平均 1.3）
  平均海面水位上昇（21世紀末, 20世紀末比） 気象庁・文科省「日本の気候変動2020」
      領域I 北海道・東北 0.38 / 0.70 m, 領域IV 北陸〜九州の東シナ海側 0.39 / 0.73 m (2℃ / 4℃)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .forcing import Rainfall

RAIN_FACTORS = {"2C": {"北海道": 1.15, "九州北西部": 1.15, "その他": 1.1},
                "4C": {"北海道": 1.4, "九州北西部": 1.4, "その他": 1.2}}
SEA_LEVEL_RISE = {"2C": {"I": 0.38, "II": 0.38, "III": 0.39, "IV": 0.39},
                  "4C": {"I": 0.70, "II": 0.70, "III": 0.74, "IV": 0.73}}
LABELS = {"current": "現在気候", "2C": "2℃上昇（RCP2.6相当, 21世紀末）",
          "4C": "4℃上昇（RCP8.5相当, 21世紀末）", "custom": "任意設定"}


@dataclass
class ClimateScenario:
    name: str = "current"
    rain_factor: float = 1.0
    intensity_gamma: float = 1.0
    sea_level_rise_m: float = 0.0
    apply_to_spinup: bool = False
    factor_field: np.ndarray | None = None
    source: str = ""

    @property
    def is_current(self) -> bool:
        return (self.rain_factor == 1.0 and self.intensity_gamma == 1.0 and self.sea_level_rise_m == 0.0
                and self.factor_field is None)

    def key(self) -> str:
        ff = "field" if self.factor_field is not None else ""
        return f"f{self.rain_factor:.3f}_g{self.intensity_gamma:.2f}_s{self.sea_level_rise_m:.2f}{ff}"

    def to_dict(self):
        d = asdict(self)
        d.pop("factor_field")
        d["label"] = LABELS.get(self.name, self.name)
        return d


def preset(level: str, rain_region: str = "その他", sea_region: str = "I") -> ClimateScenario:
    """MLIT / JMA preset for '2C' or '4C' (or 'current')."""
    if level == "current":
        return ClimateScenario()
    f = RAIN_FACTORS[level][rain_region]
    slr = SEA_LEVEL_RISE[level][sea_region]
    return ClimateScenario(name=level, rain_factor=f, sea_level_rise_m=slr,
                           source=f"降雨量変化倍率 {f}（{rain_region}, 国交省提言改訂版）; "
                                  f"海面上昇 {slr} m（領域{sea_region}, 日本の気候変動2020）")


def from_params(p: dict | None, rain_region="その他", sea_region="I") -> ClimateScenario:
    """GUI / CLI parameters -> scenario.  p = {"preset": "current|2C|4C|custom",
    "rain_factor", "intensity_gamma", "sea_level_rise_m"}; explicit values override the preset."""
    p = p or {}
    sc = preset(p.get("preset", "current") if p.get("preset", "current") != "custom" else "current",
                rain_region, sea_region)
    if p.get("preset") == "custom":
        sc.name = "custom"
        sc.source = "任意設定"
    for k in ("rain_factor", "intensity_gamma", "sea_level_rise_m"):
        if p.get(k) is not None:
            setattr(sc, k, float(p[k]))
    if p.get("apply_to_spinup"):
        sc.apply_to_spinup = True
    return sc


def scale_rainfall(rain: Rainfall, sc: ClimateScenario, t_start: float = -np.inf,
                   t_end: float = np.inf) -> Rainfall:
    """New Rainfall with the scenario applied to intervals inside [t_start, t_end]."""
    if sc.is_current:
        return rain
    rates = rain.rates.copy()
    mid = 0.5 * (rain.edges[:-1] + rain.edges[1:])
    w = (mid >= t_start) & (mid <= t_end)
    if not w.any():
        return rain
    dts = np.diff(rain.edges)[w][:, None, None]
    r = rates[w]
    target = sc.rain_factor if sc.factor_field is None else np.asarray(sc.factor_field)[None]
    if sc.intensity_gamma != 1.0:
        # r' = c r^gamma with sum(r' dt) = target * sum(r dt) per cell
        rg = np.power(np.maximum(r, 0.0) * 3.6e6, sc.intensity_gamma) / 3.6e6   # in mm/h space
        tot0 = (r * dts).sum(0)
        totg = (rg * dts).sum(0)
        c = np.where(totg > 0, target * tot0 / np.where(totg > 0, totg, 1.0), 0.0)
        r_new = rg * c
    else:
        r_new = r * target
    rates[w] = r_new
    edges = rain.edges.copy()
    return Rainfall(edges, rates * 1000.0 * 3600.0)   # Rainfall expects mm/h


def shift_stage(stage_bc: dict, sc: ClimateScenario) -> dict:
    if not sc.sea_level_rise_m:
        return stage_bc
    return {k: (t, h + sc.sea_level_rise_m) for k, (t, h) in stage_bc.items()}


def load_factor_raster(path, domain, band: int = 1) -> np.ndarray:
    """Spatial rainfall change factor (future / present) from a GeoTIFF on any
    CRS, resampled (bilinear) to the model grid.  Intended for factors derived
    from d4PDF / 気候予測データセット2022 (e.g. ratio of ensemble-mean annual
    maximum 24-h or 72-h rainfall, 4K experiment / historical experiment).
    Cells without data get the domain mean factor."""
    import rasterio
    import rasterio.warp as rw
    from rasterio.enums import Resampling
    with rasterio.open(path) as src:
        dst = np.full(domain.shape, np.nan)
        rw.reproject(rasterio.band(src, band), dst, dst_transform=domain.transform, dst_crs=domain.crs,
                     dst_nodata=np.nan, resampling=Resampling.bilinear)
    fill = np.nanmean(dst[domain.mask]) if np.isfinite(dst[domain.mask]).any() else 1.0
    return np.where(np.isfinite(dst), dst, fill)
