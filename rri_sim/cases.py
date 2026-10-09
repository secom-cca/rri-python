"""Registry of study basins and flood events."""
from __future__ import annotations

from dataclasses import dataclass

from .basin import BasinConfig

BASINS = {
    "kusu_ogase": dict(
        cfg=BasinConfig(name="kusu_ogase", outlet_lonlat=(130.96444, 33.29806),
                        bbox=(130.90, 33.05, 131.40, 33.40), crs="EPSG:6690", res=250.0,
                        river_threshold_km2=5.0,
                        gauges={"ogase": (130.96444, 33.29806),
                                "takenaka": (131.11806, 33.27667)}),
        mlit_suikei="89094000", jma_prec_nos=(82, 83, 86),
        discharge_gauges={"ogase": "309061289901330"},
        stage_gauges={"takenaka": "309061289920090"},
    ),
    # 阿武隈川 upstream of 阿久津 (郡山), 2019 台風19号.  JGD2011 / UTM 54N.
    "abukuma_akutsu": dict(
        cfg=BasinConfig(name="abukuma_akutsu", outlet_lonlat=(140.40917, 37.40778),
                        bbox=(139.95, 36.95, 140.75, 37.55), crs="EPSG:6691", res=250.0,
                        river_threshold_km2=5.0,
                        gauges={"akutsu": (140.40917, 37.40778),
                                "sukagawa": (140.38889, 37.30028)}),
        mlit_suikei="82020000", jma_prec_nos=(36,),
        discharge_gauges={"akutsu": "302011282218270", "sukagawa": "302011282218300"},
        stage_gauges={},
    ),
}


@dataclass
class Event:
    name: str
    basin: str
    start: str          # simulation start (spin-up begins here)
    end: str
    eval_start: str     # evaluation window
    eval_end: str
    role: str           # "calibration" | "validation"


EVENTS = {
    "kusu_2018": Event("kusu_2018", "kusu_ogase", "2018-06-10 00:00", "2018-07-12 00:00",
                       "2018-07-04 00:00", "2018-07-12 00:00", "calibration"),
    "kusu_2012": Event("kusu_2012", "kusu_ogase", "2012-06-15 00:00", "2012-07-20 00:00",
                       "2012-07-11 00:00", "2012-07-20 00:00", "validation"),
    "kusu_2021": Event("kusu_2021", "kusu_ogase", "2021-07-15 00:00", "2021-08-20 00:00",
                       "2021-08-10 00:00", "2021-08-20 00:00", "validation"),
    "abukuma_akutsu_2019": Event("abukuma_akutsu_2019", "abukuma_akutsu", "2019-09-20 00:00",
                                 "2019-10-20 00:00", "2019-10-11 00:00", "2019-10-18 00:00",
                                 "validation"),
    "kusu_2020": Event("kusu_2020", "kusu_ogase", "2020-06-10 00:00", "2020-07-10 00:00",
                       "2020-07-06 00:00", "2020-07-10 00:00", "validation"),
}
