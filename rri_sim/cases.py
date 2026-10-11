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
        discharge_gauges={"ogase": "309061289901330"}, prefs=(44, 40, 43), prefecture="大分",
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
        prefs=(7,), prefecture="福島",
        stage_gauges={},
    ),
    # ------------------------------------------------------------ whole basins
    # 筑後川全域: outlet at 若津 (tidal reach, observed stage as downstream boundary)
    "chikugo_whole": dict(
        cfg=BasinConfig(name="chikugo_whole", outlet_lonlat=(130.36222, 33.21472),
                        bbox=(130.20, 32.95, 131.35, 33.60), crs="EPSG:6690", res=250.0,
                        river_threshold_km2=5.0, gauge_snap_m=600.0,
                        burn_prefs=(40, 41, 43, 44), burn_system="筑後川",
                        gauges={"小渕": (130.94278, 33.30639), "小ヶ瀬": (130.96444, 33.29806),
                                "花月": (130.93444, 33.32972), "荒瀬": (130.83222, 33.34611),
                                "片ノ瀬": (130.63222, 33.35278), "端間": (130.55139, 33.37667),
                                "瀬ノ下": (130.49444, 33.31917)}),
        mlit_suikei="89094000", jma_prec_nos=(82, 83, 85, 86),
        discharge_gauges={"小渕": "309061289901020", "小ヶ瀬": "309061289901330",
                          "花月": "309061289901360", "荒瀬": "309061289901060",
                          "片ノ瀬": "309061289901130", "端間": "309061289901510",
                          "瀬ノ下": "309061289901190"},
        stage_gauges={},
        prefs=(40, 41, 43, 44), prefecture="福岡",
        tide_gauge="309061289901250",            # 若津 (water level, zero T.P. -1.94 m)
        # flood-control dams: (lon, lat), flood storage [m3], cut-off discharge [m3/s]
        dams={"下筌ダム": dict(lonlat=(130.98528, 33.15944), capacity=51.3e6, q_cut=350.0,
                             source="九州地整: 計画高水1,700→350 m3/s, 洪水貯留容量5,130万m3"),
              "松原ダム": dict(lonlat=(130.99333, 33.19444), capacity=45.8e6, q_cut=600.0,
                             source="九州地整: 治水容量4,580万m3; 放流量は2012実績(最大流入約2,110, 約7割低減)からの推定")},
    ),
    # 阿武隈川全域: outlet at 荒浜 (river mouth, observed stage as downstream boundary)
    "abukuma_whole": dict(
        cfg=BasinConfig(name="abukuma_whole", outlet_lonlat=(140.91139, 38.04944),
                        bbox=(139.90, 36.95, 141.00, 38.25), crs="EPSG:6691", res=250.0,
                        river_threshold_km2=5.0, gauge_snap_m=600.0,
                        burn_prefs=(7, 4), burn_system="阿武隈川",
                        gauges={"須賀川": (140.38889, 37.30028), "阿久津": (140.40917, 37.40778),
                                "本宮": (140.40472, 37.51583), "二本松": (140.46167, 37.59444),
                                "福島": (140.46889, 37.74972), "伏黒": (140.51861, 37.83),
                                "丸森": (140.76167, 37.92167), "舘矢間": (140.75111, 37.92611),
                                "江尻": (140.80222, 38.01861)}),
        mlit_suikei="82020000", jma_prec_nos=(36, 34),
        discharge_gauges={"須賀川": "302011282218300", "阿久津": "302011282218270",
                          "本宮": "302011282218250", "二本松": "302011282218210",
                          "福島": "302011282218100", "伏黒": "302011282218040",
                          "丸森": "302011282206080", "舘矢間": "302011282206090",
                          "江尻": "302011282206060"},
        stage_gauges={},
        prefs=(7, 4), prefecture="福島",
        tide_gauge="302011282206020",            # 荒浜 (zero T.P. -0.152 m)
        dams={"三春ダム": dict(lonlat=(140.46806, 37.40722), capacity=20.0e6, q_cut=100.0,
                             source="計画最大放流量100 m3/s (国交省事例集); 洪水調節容量は推定"),
              "七ヶ宿ダム": dict(lonlat=(140.51278, 37.96167), capacity=20.0e6, q_cut=250.0,
                              source="計画高水1,750→250 m3/s (二次資料); 洪水調節容量は推定"),
              "摺上川ダム": dict(lonlat=(140.41639, 37.93), capacity=20.0e6, q_cut=200.0,
                              source="放流量・容量とも推定（要確認）")},
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
    "chikugo_2020": Event("chikugo_2020", "chikugo_whole", "2020-06-10 00:00", "2020-07-12 00:00",
                          "2020-07-05 00:00", "2020-07-12 00:00", "validation"),
    "abukuma_2019": Event("abukuma_2019", "abukuma_whole", "2019-09-20 00:00", "2019-10-20 00:00",
                          "2019-10-11 00:00", "2019-10-18 00:00", "validation"),
    "kusu_2020": Event("kusu_2020", "kusu_ogase", "2020-06-10 00:00", "2020-07-10 00:00",
                       "2020-07-06 00:00", "2020-07-10 00:00", "validation"),
}
