"""Registry of study basins and flood events."""
from __future__ import annotations

from dataclasses import dataclass

from .basin import BasinConfig

BASINS = {
    "kusu_ogase": dict(
        cfg=BasinConfig(name="kusu_ogase", outlet_lonlat=(130.96444, 33.29806),
                        bbox=(130.90, 33.05, 131.40, 33.40), crs="EPSG:6690", res=250.0,   # NB: UTM53; zone 52 (EPSG:6689) is optimal for Kyushu (scale error ~0.14%)
                        river_threshold_km2=5.0,
                        gauges={"ogase": (130.96444, 33.29806),
                                "takenaka": (131.11806, 33.27667)},
                        expected_area_km2=528.0, gauge_area_km2={"ogase": 528.0, "takenaka": 389.0}),
        mlit_suikei="89094000", jma_prec_nos=(82, 83, 86),
        discharge_gauges={"ogase": "309061289901330"}, prefs=(44, 40, 43), prefecture="大分",
        climate_region=("九州北西部", "IV"),
        stage_gauges={"takenaka": "309061289920090"},
    ),
    # 阿武隈川 upstream of 阿久津 (郡山), 2019 台風19号.  JGD2011 / UTM 54N.
    "abukuma_akutsu": dict(
        cfg=BasinConfig(name="abukuma_akutsu", outlet_lonlat=(140.40917, 37.40778),
                        bbox=(139.95, 36.95, 140.75, 37.55), crs="EPSG:6691", res=250.0,
                        river_threshold_km2=5.0,
                        gauges={"akutsu": (140.40917, 37.40778),
                                "sukagawa": (140.38889, 37.30028)},
                        expected_area_km2=1854.3, gauge_area_km2={"akutsu": 1854.3, "sukagawa": 1190.4}),
        mlit_suikei="82020000", jma_prec_nos=(36,),
        discharge_gauges={"akutsu": "302011282218270", "sukagawa": "302011282218300"},
        prefs=(7,), prefecture="福島", climate_region=("その他", "I"),
        stage_gauges={},
    ),
    # ------------------------------------------------------------ whole basins
    # 筑後川全域: outlet at 若津 (tidal reach, observed stage as downstream boundary)
    "chikugo_whole": dict(
        cfg=BasinConfig(name="chikugo_whole", outlet_lonlat=(130.36222, 33.21472),
                        bbox=(130.20, 32.95, 131.35, 33.60), crs="EPSG:6690", res=250.0,   # NB: UTM53; zone 52 (EPSG:6689) is optimal for Kyushu (scale error ~0.14%)
                        river_threshold_km2=5.0, gauge_snap_m=600.0,
                        burn_prefs=(40, 41, 43, 44), burn_system="筑後川",
                        gauges={"小渕": (130.94278, 33.30639), "小ヶ瀬": (130.96444, 33.29806),
                                "花月": (130.93444, 33.32972), "荒瀬": (130.83222, 33.34611),
                                "片ノ瀬": (130.63222, 33.35278), "端間": (130.55139, 33.37667),
                                "瀬ノ下": (130.49444, 33.31917)},
                        expected_area_km2=2969.0,
                        gauge_area_km2={"小渕": 1167.1, "小ヶ瀬": 532.6, "花月": 67.4, "荒瀬": 1454.0,
                                        "片ノ瀬": 1805.3, "端間": 236.7, "瀬ノ下": 2279.9}),
        mlit_suikei="89094000", jma_prec_nos=(82, 83, 85, 86),
        discharge_gauges={"小渕": "309061289901020", "小ヶ瀬": "309061289901330",
                          "花月": "309061289901360", "荒瀬": "309061289901060",
                          "片ノ瀬": "309061289901130", "端間": "309061289901510",
                          "瀬ノ下": "309061289901190"},
        stage_gauges={},
        prefs=(40, 41, 43, 44), prefecture="福岡",
        # 降雨量変化倍率の地域区分 / 海面上昇の領域（日本の気候変動2020: IV=北陸〜九州の東シナ海側）
        climate_region=("九州北西部", "IV"),
        existing_levees={"source": "DEM5A", "min_area_km2": 30.0},
        # 低平地の排水（排水機場・下水道・クリーク）: assumption, see docs/04 4b
        lowland_drainage={"urban_mmh": 10.0, "agri_mmh": 2.0, "hand_max_m": 5.0},
        lowland_river_n={1e-3: 0.030, 2e-4: 0.025},   # bed slope < key -> Manning n
        gui_note=("注意: 本川の流量・水位は観測と整合（荒瀬・片ノ瀬・瀬ノ下 NSE 0.86–0.96）するが、"
                  "低平地の浸水（約410 km²）と被害額は過大。感潮区間の堤防天端（DEM推定）と排水機場・クリークの"
                  "表現が不十分なため。施策間の相対比較に用いること（docs/04 4b）。"),
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
                                "江尻": (140.80222, 38.01861)},
                        expected_area_km2=5489.6,
                        # 須賀川 895 km2 differs from the upstream model (1190 km2): mapping to be checked
                        gauge_area_km2={"須賀川": 895.1, "阿久津": 1865.7, "本宮": 2261.2, "二本松": 2382.6,
                                        "福島": 3165.6, "伏黒": 3664.4, "丸森": 4135.5, "舘矢間": 4132.4,
                                        "江尻": 4463.5}),
        mlit_suikei="82020000", jma_prec_nos=(36, 34),
        discharge_gauges={"須賀川": "302011282218300", "阿久津": "302011282218270",
                          "本宮": "302011282218250", "二本松": "302011282218210",
                          "福島": "302011282218100", "伏黒": "302011282218040",
                          "丸森": "302011282206080", "舘矢間": "302011282206090",
                          "江尻": "302011282206060"},
        stage_gauges={},
        prefs=(7, 4), prefecture="福島",
        climate_region=("その他", "I"),          # 東北: その他12地域 / 領域I 北海道・東北
        existing_levees={"source": "DEM5A", "min_area_km2": 30.0},
        lowland_drainage={"urban_mmh": 10.0, "agri_mmh": 2.0, "hand_max_m": 5.0},
        lowland_river_n={1e-3: 0.030, 2e-4: 0.025},
        gui_note=("注意: パラメータ未校正のため流量が系統的に過小（総量 −18〜−33%）。既設堤防はDEMからの推定。"
                  "施策間の相対比較に用いること（docs/04 4b）。"),
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
