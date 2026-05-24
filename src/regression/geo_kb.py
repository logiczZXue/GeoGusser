"""Geographic Knowledge Base: maps geographic elements to spatial distributions in China.

Each element (climate zone, terrain type, vegetation, architecture, etc.) maps to
one or more bounding boxes representing where that element occurs in China.

For the constraint fusion engine: each element casts a "vote" for where the image
could be. Elements are multiplied (intersection) to narrow the candidate region.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BBox:
    """Bounding box in decimal degrees."""
    lat_min: float
    lat_max: float
    lng_min: float
    lng_max: float
    label: str = ""  # region name

    @property
    def center_lat(self) -> float:
        return (self.lat_min + self.lat_max) / 2

    @property
    def center_lng(self) -> float:
        return (self.lng_min + self.lng_max) / 2

    @property
    def area_km2(self) -> float:
        """Approximate area in km^2 (rough, assumes mid-latitude)."""
        dlat = (self.lat_max - self.lat_min) * 111.0
        dlng = (self.lng_max - self.lng_min) * 102.0  # ~cos(30°) * 111
        return dlat * dlng


# ═══════════════════════════════════════════════════════════════════════════
# CLIMATE ZONES
# ═══════════════════════════════════════════════════════════════════════════

CLIMATE_ZONES = {
    "tropical": [
        BBox(18.0, 22.0, 108.0, 111.5, "Hainan"),
        BBox(21.0, 22.5, 100.0, 102.0, "Xishuangbanna"),
        BBox(20.0, 23.0, 109.0, 117.5, "S Guangdong / S Guangxi coast"),
    ],
    "subtropical": [
        BBox(22.0, 33.0, 104.0, 122.5, "S of Yangtze: Guangxi to Zhejiang"),
        BBox(22.0, 31.0, 97.0, 104.0, "Yunnan"),
        BBox(29.0, 32.5, 103.0, 110.5, "Sichuan Basin / Chongqing"),
    ],
    "temperate": [
        BBox(33.0, 43.0, 105.0, 126.5, "N China Plain to NE China"),
        BBox(31.5, 39.5, 105.5, 111.5, "Shaanxi / Shanxi"),
        BBox(32.0, 43.0, 92.0, 105.0, "Gansu / Ningxia"),
    ],
    "arid": [
        BBox(34.0, 49.5, 73.0, 96.5, "Xinjiang"),
        BBox(37.5, 43.0, 96.0, 109.0, "W Inner Mongolia / Gansu corridor"),
        BBox(34.0, 42.5, 88.0, 96.5, "Qaidam Basin / Qinghai"),
    ],
    "alpine": [
        BBox(26.5, 36.5, 78.0, 99.5, "Tibet Plateau"),
        BBox(31.0, 39.5, 89.0, 103.5, "Qinghai"),
        BBox(26.0, 32.0, 97.0, 104.0, "W Sichuan / NW Yunnan highlands"),
    ],
    "boreal": [
        BBox(48.0, 54.0, 121.0, 135.5, "N Heilongjiang / Greater Khingan"),
        BBox(46.0, 50.0, 80.0, 90.0, "Altai Mountains / N Xinjiang"),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# TERRAIN TYPES
# ═══════════════════════════════════════════════════════════════════════════

TERRAIN_TYPES = {
    "urban_flat": [
        BBox(18.0, 54.0, 73.0, 135.5, "All China cities"),  # fallback
    ],
    "farmland_plain": [
        BBox(30.0, 41.0, 113.5, 122.5, "North China Plain"),
        BBox(43.0, 48.5, 123.0, 135.5, "NE China Plain"),
        BBox(29.0, 33.0, 116.0, 122.0, "Yangtze Delta"),
        BBox(29.0, 32.5, 103.0, 108.0, "Sichuan Basin"),
        BBox(21.0, 24.0, 109.0, 117.5, "Pearl River Delta"),
    ],
    "rolling_hills": [
        BBox(24.0, 30.5, 113.5, 122.5, "SE China hills: Fujian/Jiangxi/Hunan/Zhejiang"),
        BBox(20.5, 26.5, 104.0, 112.5, "Guangxi/Guizhou hills"),
        BBox(30.0, 33.5, 117.0, 120.0, "S Anhui / S Jiangsu hills"),
    ],
    "sharp_mountains": [
        BBox(26.0, 34.5, 97.0, 104.0, "Hengduan Mountains: W Sichuan / NW Yunnan"),
        BBox(31.5, 39.5, 105.5, 111.5, "Qinling Mountains"),
        BBox(41.0, 45.0, 80.0, 96.5, "Tianshan / Altai"),
        BBox(26.5, 36.5, 78.0, 95.0, "Himalaya / Transhimalaya"),
        BBox(27.0, 31.0, 115.0, 121.0, "Wuyi / Yandang / coastal SE mountains"),
    ],
    "karst_peaks": [
        BBox(24.0, 26.5, 102.5, 112.5, "Yunnan-Guizhou-Guangxi karst (Shilin → Guilin)"),
        BBox(24.5, 30.5, 108.5, 114.5, "Hunan karst (Zhangjiajie area)"),
        BBox(20.5, 24.0, 103.5, 108.0, "N Vietnam border karst"),
    ],
    "sandstone_pillars": [
        BBox(28.5, 30.5, 109.0, 111.5, "Zhangjiajie / Wulingyuan"),
    ],
    "desert_dunes": [
        BBox(37.0, 46.0, 75.0, 90.0, "Taklamakan Desert"),
        BBox(39.0, 46.0, 80.0, 96.5, "Gurbantunggut Desert"),
        BBox(37.5, 42.5, 100.0, 107.0, "Tengger / Badain Jaran"),
        BBox(38.0, 43.0, 88.0, 94.0, "Kumtag Desert"),
    ],
    "grassland_steppe": [
        BBox(41.0, 51.0, 110.0, 126.5, "Inner Mongolia / Hulunbuir"),
        BBox(32.0, 39.5, 89.0, 103.5, "Qinghai / Gansu grassland"),
        BBox(27.0, 33.0, 97.0, 104.0, "W Sichuan / NW Yunnan alpine grassland (Zoige)"),
        BBox(41.0, 49.0, 80.0, 92.0, "N Xinjiang grassland (Ili / Altai)"),
    ],
    "plateau": [
        BBox(26.5, 36.5, 78.0, 99.5, "Tibet Plateau"),
        BBox(31.0, 39.5, 89.0, 103.5, "Qinghai Plateau"),
        BBox(34.0, 40.0, 73.0, 78.0, "Pamir Plateau"),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# VEGETATION ZONES
# ═══════════════════════════════════════════════════════════════════════════

VEGETATION_ZONES = {
    "tropical_rainforest": [
        BBox(18.0, 20.5, 108.0, 111.5, "Hainan"),
        BBox(21.0, 22.5, 100.0, 102.0, "Xishuangbanna"),
        BBox(20.0, 22.0, 109.0, 112.0, "SW Guangdong / S Guangxi coast"),
    ],
    "broadleaf_evergreen": [
        BBox(22.0, 33.0, 104.0, 122.5, "S China subtropical zone"),
        BBox(21.0, 29.5, 97.5, 104.0, "Yunnan"),
        BBox(29.0, 32.5, 103.0, 110.0, "Sichuan Basin"),
    ],
    "mixed_forest": [
        BBox(31.0, 37.0, 105.5, 122.5, "Qinling-Huaihe transitional zone"),
        BBox(29.0, 33.0, 108.0, 117.0, "Mid-Yangtze mixed forest"),
        BBox(36.0, 41.0, 113.5, 120.0, "N China mixed forest"),
        BBox(43.0, 48.5, 121.5, 132.0, "NE China mixed forest (Heilongjiang/Jilin lowlands)"),
    ],
    "conifer_forest": [
        BBox(48.0, 54.0, 121.0, 135.5, "N Heilongjiang boreal conifer"),
        BBox(41.0, 48.5, 120.0, 131.5, "NE China conifer (Changbai / Khingan)"),
        BBox(26.0, 34.5, 97.0, 104.0, "Hengduan conifer (W Sichuan / NW Yunnan)"),
        BBox(41.0, 45.0, 80.0, 96.5, "Tianshan / Altai spruce-fir"),
        BBox(26.5, 36.5, 88.0, 97.0, "SE Tibet conifer"),
        BBox(27.0, 33.0, 114.0, 121.0, "E China montane conifer (Dabie / Huangshan / Wuyi)"),
    ],
    "alpine_meadow": [
        BBox(26.5, 36.5, 78.0, 99.5, "Tibet alpine meadow"),
        BBox(31.0, 39.5, 89.0, 103.5, "Qinghai alpine meadow"),
        BBox(26.0, 32.0, 97.0, 104.0, "W Sichuan / NW Yunnan alpine (Zoige / Daocheng)"),
        BBox(41.0, 45.0, 80.0, 96.5, "Tianshan alpine meadow"),
    ],
    "bamboo_forest": [
        BBox(27.0, 32.0, 103.0, 108.0, "Sichuan bamboo (panda habitat)"),
        BBox(27.0, 31.0, 117.0, 122.0, "Zhejiang / Fujian bamboo"),
        BBox(24.0, 29.0, 110.0, 117.0, "Hunan / Jiangxi bamboo"),
    ],
    "desert_scrub": [
        BBox(37.0, 46.0, 75.0, 90.0, "Taklamakan"),
        BBox(37.5, 42.5, 100.0, 107.0, "Tengger desert edges"),
        BBox(34.0, 42.5, 88.0, 96.5, "Qaidam Basin"),
    ],
    "grassland": [
        BBox(41.0, 51.0, 110.0, 126.5, "Inner Mongolia grassland"),
        BBox(41.0, 46.0, 80.0, 90.0, "N Xinjiang grassland (Ili valley)"),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# TREE SPECIES (more specific than vegetation zone)
# ═══════════════════════════════════════════════════════════════════════════

TREE_SPECIES = {
    "banyan": [
        BBox(21.0, 25.5, 108.0, 121.0, "S China: Guangzhou/Fuzhou/Nanning/haikou"),
        BBox(21.0, 22.5, 100.0, 102.0, "Xishuangbanna"),
    ],
    "camphor": [
        BBox(27.0, 32.0, 110.0, 122.5, "Mid-lower Yangtze: Hangzhou/Changsha/Nanchang"),
        BBox(29.0, 32.5, 103.0, 108.0, "Sichuan Basin: Chengdu/Chongqing"),
    ],
    "plane_tree": [
        BBox(30.0, 32.5, 118.0, 122.0, "Shanghai/Nanjing/Hangzhou"),
        BBox(30.0, 31.5, 112.0, 115.0, "Wuhan area"),
    ],
    "gingko": [
        BBox(30.0, 41.0, 105.0, 122.5, "N China cities (planted, common)"),
    ],
    "poplar": [
        BBox(36.0, 48.5, 113.5, 135.5, "N China Plain + NE China"),
    ],
    "coconut_palm": [
        BBox(18.0, 20.5, 108.0, 111.5, "Hainan"),
        BBox(20.0, 23.0, 109.0, 117.5, "S Guangdong / S Guangxi coast"),
        BBox(21.0, 22.5, 100.0, 102.0, "Xishuangbanna"),
    ],
    "chinese_red_pine": [
        BBox(22.0, 33.0, 104.0, 122.5, "Widespread S China (masson pine)"),
    ],
    "bamboo": [
        BBox(27.0, 32.0, 103.0, 108.0, "Sichuan bamboo"),
        BBox(27.0, 31.0, 117.0, 122.0, "Zhejiang / Fujian bamboo"),
        BBox(24.0, 29.0, 110.0, 117.0, "Hunan / Jiangxi bamboo"),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECTURE STYLES
# ═══════════════════════════════════════════════════════════════════════════

ARCHITECTURE_STYLES = {
    "hui_style": [
        BBox(29.0, 31.5, 116.5, 119.5, "S Anhui / N Jiangxi / W Zhejiang (Huizhou)"),
    ],
    "tibetan_stone": [
        BBox(26.5, 34.0, 88.0, 99.5, "Tibet"),
        BBox(26.0, 33.0, 97.0, 104.0, "W Sichuan (Ganzi/ABA)"),
        BBox(27.0, 29.5, 98.5, 101.0, "NW Yunnan (Diqing/Shangri-La)"),
        BBox(31.5, 39.5, 89.0, 103.5, "Qinghai / S Gansu"),
    ],
    "stilt_house": [
        BBox(27.0, 31.0, 107.0, 112.0, "W Hunan / Guizhou / Chongqing (Tujia/Miao)"),
        BBox(23.5, 26.5, 104.0, 108.0, "S Guizhou"),
    ],
    "tulou": [
        BBox(23.5, 26.0, 116.0, 118.0, "SW Fujian (Longyan/Zhangzhou)"),
    ],
    "arcade": [
        BBox(22.5, 24.5, 113.0, 115.0, "Guangzhou / Pearl River Delta"),
        BBox(23.5, 26.0, 116.0, 121.0, "S Fujian (Xiamen/Quanzhou/Zhangzhou)"),
        BBox(21.0, 23.5, 108.0, 110.5, "Guangxi coast (Beihai)"),
        BBox(19.0, 20.5, 109.5, 111.5, "Hainan"),
    ],
    "courtyard": [
        BBox(39.0, 41.0, 115.5, 117.5, "Beijing"),
        BBox(36.0, 42.5, 113.5, 120.0, "Hebei / N China"),
    ],
    "shikumen": [
        BBox(30.5, 31.5, 121.0, 122.0, "Shanghai"),
    ],
    "traditional_official": [
        BBox(18.0, 54.0, 73.0, 135.5, "Nationwide (official/temple style)"),
    ],
    "modern_glass": [
        BBox(18.0, 54.0, 73.0, 135.5, "All major Chinese cities"),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# LANGUAGE SCRIPTS
# ═══════════════════════════════════════════════════════════════════════════

LANGUAGE_SCRIPTS = {
    "simplified_chinese": [
        BBox(18.0, 54.0, 73.0, 135.5, "Mainland China"),
    ],
    "traditional_chinese": [
        BBox(21.5, 25.5, 120.0, 122.5, "Taiwan"),
        BBox(22.0, 23.0, 113.5, 114.5, "Hong Kong"),
        BBox(22.0, 22.5, 113.0, 113.5, "Macau"),
    ],
    "tibetan": [
        BBox(26.5, 34.0, 88.0, 99.5, "Tibet"),
        BBox(31.5, 39.5, 89.0, 103.5, "Qinghai / S Gansu"),
        BBox(26.0, 33.0, 97.0, 104.0, "W Sichuan (Ganzi/ABA)"),
        BBox(27.0, 29.5, 98.5, 101.0, "NW Yunnan"),
    ],
    "uyghur_arabic": [
        BBox(34.0, 49.5, 73.0, 96.5, "Xinjiang"),
    ],
    "mongolian": [
        BBox(37.5, 51.0, 97.0, 126.5, "Inner Mongolia"),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# MOUNTAIN ROCK TYPES
# ═══════════════════════════════════════════════════════════════════════════

MOUNTAIN_ROCK_TYPES = {
    "granite_spheroidal": [
        BBox(29.5, 31.0, 117.5, 119.0, "Huangshan / Sanqingshan"),
        BBox(34.0, 35.0, 109.5, 111.0, "Huashan"),
        BBox(27.5, 29.0, 117.5, 118.5, "Wuyishan"),
        BBox(28.5, 30.0, 115.5, 117.0, "Lushan"),
        BBox(32.0, 33.0, 110.5, 111.5, "Wudangshan"),
        # High-altitude granite massifs
        BBox(33.5, 34.5, 107.0, 108.0, "Taibai Shan / Qinling granite (3767m)"),
        BBox(26.0, 32.0, 97.0, 104.0, "Hengduan granite (Gongga / Siguniang massif)"),
        BBox(27.0, 30.0, 88.0, 93.0, "E Himalaya granite (Namcha Barwa / Kangri)"),
        BBox(41.0, 43.0, 86.0, 89.0, "Tianshan granite (Bogda / Tomur)"),
    ],
    "quartz_sandstone_pillars": [
        BBox(28.5, 30.0, 109.5, 111.0, "Zhangjiajie / Wulingyuan"),
    ],
    "limestone_karst": [
        BBox(24.0, 26.0, 109.5, 111.5, "Guilin / Yangshuo"),
        BBox(24.5, 26.5, 104.0, 106.0, "Shilin / S Guizhou karst"),
    ],
    "red_sandstone_danxia": [
        BBox(27.5, 28.5, 117.5, 118.5, "Wuyishan (Danxia)"),
        BBox(38.5, 39.5, 99.5, 101.0, "Zhangye Danxia"),
        BBox(28.0, 29.0, 116.5, 117.5, "Longhushan"),
        BBox(24.5, 26.0, 113.0, 114.5, "Danxiashan (Guangdong)"),
    ],
    "snow_peaks_glaciers": [
        BBox(29.0, 30.5, 101.0, 103.0, "Gongga / Siguniang / Minya Konka"),
        BBox(27.0, 29.0, 98.0, 100.5, "Meili Snow Mountain / Yulong / Haba"),
        BBox(27.5, 29.0, 85.5, 87.5, "Everest / Cho Oyu region"),
        BBox(30.0, 32.0, 90.0, 95.0, "Nyainqentanglha / Namcha Barwa"),
        BBox(41.0, 43.5, 80.0, 89.0, "Tianshan / Tomur / Bogda"),
    ],
    "alpine_lakes": [
        BBox(32.5, 34.0, 103.0, 104.5, "Jiuzhaigou / Huanglong"),
        BBox(28.0, 29.5, 99.5, 101.0, "Daocheng Yading"),
        BBox(27.5, 28.5, 100.5, 101.5, "Lugu Lake"),
        BBox(30.5, 32.0, 90.0, 91.5, "Namtso / Yamdrok"),
        BBox(48.5, 49.0, 87.0, 87.5, "Kanas Lake"),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# SOIL COLORS
# ═══════════════════════════════════════════════════════════════════════════

SOIL_COLORS = {
    "red": [
        BBox(20.0, 30.0, 104.0, 122.5, "S China red soil zone"),
        BBox(21.0, 29.5, 97.5, 104.0, "Yunnan red soil"),
    ],
    "yellow": [
        BBox(24.0, 33.0, 108.5, 122.5, "Mid-China yellow-brown soil"),
        BBox(24.5, 30.5, 103.5, 108.5, "Guizhou / Chongqing yellow soil"),
    ],
    "yellow_brown": [
        BBox(24.0, 33.0, 108.5, 122.5, "Mid-China yellow-brown soil"),
        BBox(24.5, 30.5, 103.5, 108.5, "Guizhou / Chongqing yellow soil"),
    ],
    "brown": [
        BBox(36.0, 48.5, 113.5, 135.5, "NE China dark brown soil"),
        BBox(32.0, 43.0, 92.0, 113.5, "NW China / Gansu brown soil"),
        BBox(26.0, 34.5, 97.0, 104.0, "Hengduan mountain brown soil"),
    ],
    "black": [
        BBox(43.0, 48.5, 121.5, 131.5, "NE China black soil (Heilongjiang/Jilin)"),
    ],
    "grey": [
        BBox(37.0, 49.5, 75.0, 96.5, "Xinjiang grey desert soil"),
        BBox(37.5, 42.5, 96.0, 107.0, "W Inner Mongolia grey soil"),
    ],
    "loess_yellow": [
        BBox(34.5, 41.0, 105.5, 114.5, "Loess Plateau (Shaanxi/Shanxi/Gansu)"),
    ],
    "white": [
        BBox(34.0, 42.5, 88.0, 96.5, "Qaidam Basin saline soil"),
        BBox(39.0, 42.0, 75.0, 79.0, "W Tarim saline soil"),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# SKY / AIR QUALITY
# ═══════════════════════════════════════════════════════════════════════════

SKY_QUALITY = {
    "clear_blue": [
        BBox(26.5, 36.5, 78.0, 99.5, "Tibet"),
        BBox(26.0, 32.0, 97.0, 104.0, "W Sichuan / NW Yunnan highlands"),
        BBox(31.0, 39.5, 89.0, 103.5, "Qinghai"),
        BBox(41.0, 49.0, 80.0, 96.5, "Xinjiang highlands"),
    ],
    "grey_hazy": [
        BBox(30.0, 41.0, 113.5, 122.5, "N China Plain (Beijing/Hebei/Henan)"),
        BBox(22.0, 33.0, 104.0, 122.5, "S China humid cities"),
        BBox(29.0, 32.5, 103.0, 108.0, "Sichuan Basin (Chengdu/Chongqing fog)"),
    ],
    "thick_fog": [
        BBox(29.0, 33.0, 108.0, 122.5, "Yangtze River fog zone"),
        BBox(29.0, 32.5, 103.0, 108.0, "Sichuan Basin (Chongqing/Chengdu fog)"),
        BBox(29.5, 31.0, 117.5, 119.0, "Huangshan cloud sea"),
    ],
    "dusty_yellow": [
        BBox(34.0, 43.0, 92.0, 109.0, "Gansu / Inner Mongolia (spring sandstorms)"),
        BBox(34.0, 42.5, 75.0, 92.0, "Xinjiang dust"),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# COMPOUND SCENES — pre-computed intersections of terrain+vegetation+soil+climate
# These represent specific landscape types that only co-occur in narrow regions.
# Each entry has "conditions" (category→value pairs that must ALL match) and
# tight bboxes. Used in fusion engine as additional hard constraints.
# ═══════════════════════════════════════════════════════════════════════════

COMPOUND_SCENES = {
    # ── Karst landscapes ───────────────────────────────────────────────
    "guilin_yangshuo_karst": {
        "conditions": {
            "terrain_type": "karst_peaks",
            "vegetation_zone": "broadleaf_evergreen",
            "soil_color": "red",
            "climate_zone": "subtropical",
        },
        "bboxes": [
            BBox(24.0, 26.0, 109.5, 111.5, "Guilin-Yangshuo tower karst (Li River)"),
        ],
    },
    "shilin_stone_forest": {
        "conditions": {
            "terrain_type": "karst_peaks",
            "climate_zone": "subtropical",
            "vegetation_zone": "broadleaf_evergreen",
            "soil_color": "red",
            "scene_type": "mountain_trail_karst",
        },
        "bboxes": [
            BBox(24.5, 26.5, 103.5, 105.0, "Shilin stone forest (Yunnan)"),
        ],
    },
    "yunnan_karst_highland": {
        "conditions": {
            "terrain_type": "karst_peaks",
            "climate_zone": "subtropical",
            "soil_color": "red",
            "scene_type": "mountain_trail_karst",
        },
        "bboxes": [
            BBox(24.0, 26.0, 102.5, 105.0, "Yunnan high karst (Shilin / Puzhehei)"),
        ],
    },

    # ── Sandstone landscapes ───────────────────────────────────────────
    "zhangjiajie_sandstone_pillars": {
        "conditions": {
            "terrain_type": "sandstone_pillars",
            "climate_zone": "subtropical",
        },
        "bboxes": [
            BBox(28.5, 30.0, 109.0, 111.5, "Zhangjiajie / Wulingyuan quartz sandstone"),
        ],
    },

    # ── Granite landscapes (subtropical) ───────────────────────────────
    "huangshan_granite_pines": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "climate_zone": "subtropical",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(29.5, 31.0, 117.5, 119.0, "Huangshan rounded granite + pines"),
        ],
    },
    "huangshan_granite_conifer": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "climate_zone": "subtropical",
            "vegetation_zone": "conifer_forest",
            "soil_color": "yellow_brown",
        },
        "bboxes": [
            BBox(29.5, 31.0, 117.5, 119.0, "Huangshan granite + conifer + yellow soil"),
        ],
    },
    "sanqingshan_granite": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(28.5, 29.5, 117.5, 118.5, "Sanqingshan sharp granite peaks"),
        ],
    },
    "lushan_wudang_granite": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "climate_zone": "subtropical",
            "vegetation_zone": "mixed_forest",
        },
        "bboxes": [
            BBox(28.5, 33.0, 110.5, 117.0, "Lushan / Wudangshan granite"),
        ],
    },

    # ── Granite landscapes (alpine/temperate) ──────────────────────────
    "qinling_taibai_alpine_granite": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(32.0, 35.0, 106.5, 109.0, "Qinling Taibai-Shan alpine granite + conifer"),
        ],
    },
    "hengduan_granite_massif": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "alpine_meadow",
        },
        "bboxes": [
            BBox(28.0, 32.0, 97.0, 104.0, "Hengduan granite massif (Gongga / Siguniang / Minya Konka)"),
        ],
    },

    # ── Granite landscapes (temperate) ──────────────────────────────────
    "huashan_qinling_granite": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "climate_zone": "temperate",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "mixed_forest",
        },
        "bboxes": [
            BBox(33.5, 35.5, 109.0, 111.0, "Huashan / Qinling granite (Shaanxi)"),
        ],
    },

    # ── Snow mountain landscapes ───────────────────────────────────────
    "gongga_siguniang_snow": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(29.0, 30.5, 101.0, 103.0, "Gongga / Siguniang / Minya Konka snow peaks"),
        ],
    },
    "meili_yulong_snow": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
            "language_script": "tibetan",
        },
        "bboxes": [
            BBox(27.0, 29.5, 98.0, 101.0, "Meili Snow Mtn / Yulong / Haba"),
        ],
    },
    "everest_chooyu_snow": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
            "vegetation_zone": "alpine_meadow",
            "language_script": "tibetan",
        },
        "bboxes": [
            BBox(27.5, 29.0, 85.5, 87.5, "Everest / Cho Oyu region (Himalaya)"),
        ],
    },
    "nyainqentanglha_snow": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
            "vegetation_zone": "alpine_meadow",
        },
        "bboxes": [
            BBox(30.0, 32.0, 90.0, 95.0, "Nyainqentanglha / Namcha Barwa"),
        ],
    },
    "tianshan_snow_alpine": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "arid",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(41.0, 44.0, 80.0, 89.0, "Tianshan / Tomur / Bogda snow + glacier"),
        ],
    },

    # ── Alpine lake landscapes ─────────────────────────────────────────
    "daocheng_yading_lakes": {
        "conditions": {
            "mountain_rock_type": "alpine_lakes",
            "climate_zone": "alpine",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(28.0, 29.5, 99.5, 101.0, "Daocheng Yading alpine lakes + conifer"),
        ],
    },
    "jiuzhaigou_huanglong_lakes": {
        "conditions": {
            "mountain_rock_type": "alpine_lakes",
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(32.5, 34.0, 103.0, 105.0, "Jiuzhaigou / Huanglong travertine lakes"),
        ],
    },
    "kanas_lake": {
        "conditions": {
            "mountain_rock_type": "alpine_lakes",
            "climate_zone": "boreal",
        },
        "bboxes": [
            BBox(48.0, 49.0, 87.0, 88.0, "Kanas Lake (Altai)"),
        ],
    },

    # ── Desert / arid landscapes ───────────────────────────────────────
    "qaidam_basin_desert": {
        "conditions": {
            "vegetation_zone": "desert_scrub",
            "climate_zone": "arid",
            "terrain_type": "plateau",
        },
        "bboxes": [
            BBox(34.0, 39.0, 88.0, 97.0, "Qaidam Basin desert plateau"),
        ],
    },
    "taklamakan_desert": {
        "conditions": {
            "vegetation_zone": "desert_scrub",
            "climate_zone": "arid",
            "terrain_type": "desert_dunes",
        },
        "bboxes": [
            BBox(37.0, 43.0, 75.0, 90.0, "Taklamakan Desert core"),
        ],
    },

    # ── Grassland / wetland landscapes ─────────────────────────────────
    "zoige_alpine_wetland": {
        "conditions": {
            "vegetation_zone": "alpine_meadow",
            "terrain_type": "grassland_steppe",
            "climate_zone": "alpine",
        },
        "bboxes": [
            BBox(32.0, 34.5, 101.0, 104.0, "Zoige alpine wetland (NW Sichuan)"),
        ],
    },
    "inner_mongolia_grassland": {
        "conditions": {
            "vegetation_zone": "grassland",
            "terrain_type": "grassland_steppe",
            "climate_zone": "temperate",
        },
        "bboxes": [
            BBox(41.0, 51.0, 110.0, 126.5, "Inner Mongolia / Hulunbuir grassland"),
        ],
    },
    "east_inner_mongolia_arid_steppe": {
        "conditions": {
            "vegetation_zone": "grassland",
            "terrain_type": "grassland_steppe",
            "climate_zone": "arid",
            "language_script": "mongolian",
        },
        "bboxes": [
            BBox(41.0, 46.0, 110.0, 120.0, "E Inner Mongolia arid steppe (Xilingol)"),
        ],
    },

    # ── Tropical landscapes ────────────────────────────────────────────
    "xishuangbanna_tropical": {
        "conditions": {
            "climate_zone": "tropical",
            "vegetation_zone": "tropical_rainforest",
            "terrain_type": "karst_peaks",
        },
        "bboxes": [
            BBox(21.0, 22.5, 100.0, 102.0, "Xishuangbanna tropical rainforest + karst"),
        ],
    },
    "hainan_tropical": {
        "conditions": {
            "climate_zone": "tropical",
            "vegetation_zone": "tropical_rainforest",
        },
        "bboxes": [
            BBox(18.0, 20.5, 108.5, 111.5, "Hainan tropical"),
        ],
    },

    # ── Danxia landscapes ──────────────────────────────────────────────
    "danxiashan_red_sandstone": {
        "conditions": {
            "mountain_rock_type": "red_sandstone_danxia",
            "climate_zone": "subtropical",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(24.5, 26.0, 113.0, 114.5, "Danxiashan (Guangdong)"),
        ],
    },
    "zhangye_danxia": {
        "conditions": {
            "mountain_rock_type": "red_sandstone_danxia",
            "climate_zone": "arid",
        },
        "bboxes": [
            BBox(38.5, 39.5, 99.5, 101.0, "Zhangye Danxia (Gansu corridor)"),
        ],
    },
    "wuyishan_danxia": {
        "conditions": {
            "mountain_rock_type": "red_sandstone_danxia",
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(27.5, 28.5, 117.5, 118.5, "Wuyishan Danxia + tea terraces"),
        ],
    },

    # ── Volcanic / special landscapes ──────────────────────────────────
    "changbaishan_volcanic": {
        "conditions": {
            "climate_zone": "temperate",
            "vegetation_zone": "conifer_forest",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(41.5, 42.5, 127.0, 129.0, "Changbaishan volcanic + crater lake"),
        ],
    },

    # ── Bamboo landscapes ──────────────────────────────────────────────
    "sichuan_bamboo_forest": {
        "conditions": {
            "vegetation_zone": "bamboo_forest",
            "climate_zone": "subtropical",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(27.0, 32.0, 103.0, 108.0, "Sichuan bamboo (panda habitat)"),
        ],
    },
    "zhejiang_fujian_bamboo": {
        "conditions": {
            "vegetation_zone": "bamboo_forest",
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(27.0, 31.0, 117.0, 122.0, "Zhejiang / Fujian bamboo mountains"),
        ],
    },

    # ── Tropical forest variants ──────────────────────────────────────
    "xishuangbanna_tropical_hills": {
        "conditions": {
            "climate_zone": "tropical",
            "vegetation_zone": "tropical_rainforest",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(21.0, 23.0, 100.0, 102.0, "Xishuangbanna tropical hills"),
        ],
    },

    # ── Tibet specific ────────────────────────────────────────────────
    "lhasa_valley": {
        "conditions": {
            "climate_zone": "alpine",
            "terrain_type": "plateau",
            "vegetation_zone": "alpine_meadow",
            "sky_quality": "clear_blue",
        },
        "bboxes": [
            BBox(29.0, 30.5, 90.0, 92.0, "Lhasa / Yarlung Tsangpo valley"),
        ],
    },
    "everest_high_desert": {
        "conditions": {
            "climate_zone": "alpine",
            "vegetation_zone": "desert_scrub",
            "terrain_type": "plateau",
        },
        "bboxes": [
            BBox(28.0, 33.0, 80.0, 88.0, "W Tibet high-altitude cold desert"),
        ],
    },

    # ── Desert variants ───────────────────────────────────────────────
    "qaidam_desert_dunes": {
        "conditions": {
            "vegetation_zone": "desert_scrub",
            "climate_zone": "arid",
            "terrain_type": "desert_dunes",
            "scene_type": "desert",
        },
        "bboxes": [
            BBox(35.0, 39.5, 88.0, 97.0, "Qaidam Basin dunes"),
        ],
    },
    "qaidam_tibetan_desert": {
        "conditions": {
            "vegetation_zone": "desert_scrub",
            "climate_zone": "arid",
            "terrain_type": "desert_dunes",
            "language_script": "tibetan",
        },
        "bboxes": [
            BBox(35.0, 38.0, 91.0, 96.0, "Qaidam core desert + Tibetan script"),
        ],
    },
    "gobi_desert": {
        "conditions": {
            "vegetation_zone": "desert_scrub",
            "climate_zone": "arid",
            "soil_color": "grey",
        },
        "bboxes": [
            BBox(40.0, 47.0, 95.0, 107.0, "Gobi Desert (S Mongolia / N Gansu)"),
        ],
    },

    # ── NE China forest ───────────────────────────────────────────────
    "ne_china_conifer_forest": {
        "conditions": {
            "vegetation_zone": "conifer_forest",
            "soil_color": "black",
            "climate_zone": "temperate",
        },
        "bboxes": [
            BBox(41.0, 48.0, 121.0, 132.0, "NE China mixed-conifer forest belt"),
        ],
    },
    "changbaishan_broadleaf": {
        "conditions": {
            "climate_zone": "temperate",
            "vegetation_zone": "mixed_forest",
            "terrain_type": "sharp_mountains",
            "soil_color": "black",
        },
        "bboxes": [
            BBox(41.0, 43.0, 126.0, 129.0, "Changbaishan mixed forest"),
        ],
    },

    # ── Canyon landscapes ─────────────────────────────────────────────
    "tiger_leaping_gorge": {
        "conditions": {
            "landform_detail": "river_canyon",
            "climate_zone": "alpine",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(26.5, 28.0, 99.5, 101.0, "Tiger Leaping Gorge / Nu River canyon"),
        ],
    },
    "three_gorges": {
        "conditions": {
            "landform_detail": "river_canyon",
            "climate_zone": "subtropical",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(30.0, 31.5, 108.0, 111.5, "Three Gorges (Yangtze) canyon"),
        ],
    },

    # ── Urban compound scenes ─────────────────────────────────────────
    "guangzhou_arcade_banyan": {
        "conditions": {
            "architecture_style": "arcade",
            "tree_species": "banyan",
            "climate_zone": "subtropical",
        },
        "bboxes": [
            BBox(23.0, 24.5, 112.5, 114.5, "Guangzhou / Foshan arcade + banyan"),
        ],
    },
    "shanghai_shikumen_planetree": {
        "conditions": {
            "architecture_style": "shikumen",
            "tree_species": "plane_tree",
            "urbanization": "metropolis",
        },
        "bboxes": [
            BBox(31.0, 31.5, 121.2, 121.7, "Shanghai shikumen + plane tree lanes"),
        ],
    },
    "beijing_courtyard_traditional": {
        "conditions": {
            "architecture_style": "traditional_official",
            "climate_zone": "temperate",
            "urbanization": "metropolis",
        },
        "bboxes": [
            BBox(39.7, 40.2, 116.1, 116.7, "Beijing courtyard / traditional"),
        ],
    },
    "shenzhen_modern_metro": {
        "conditions": {
            "urbanization": "metropolis",
            "architecture_style": "modern_glass",
            "climate_zone": "subtropical",
            "sky_quality": "grey_hazy",
        },
        "bboxes": [
            BBox(22.3, 23.5, 113.0, 114.5, "Shenzhen / Guangzhou / Dongguan"),
        ],
    },
    "harbin_ne_urban": {
        "conditions": {
            "soil_color": "black",
            "urbanization": "medium_city",
            "climate_zone": "temperate",
            "vegetation_zone": "mixed_forest",
        },
        "bboxes": [
            BBox(45.0, 47.0, 125.5, 128.0, "Harbin / Daqing NE China black soil urban"),
        ],
    },
    "tibetan_town_highland": {
        "conditions": {
            "urbanization": "medium_city",
            "language_script": "tibetan",
            "sky_quality": "clear_blue",
        },
        "bboxes": [
            BBox(28.5, 30.5, 90.5, 92.0, "Lhasa / Shigatse area"),
            BBox(31.0, 33.0, 96.0, 97.5, "Yushu / Nangqen (S Qinghai)"),
            BBox(27.0, 29.0, 98.5, 100.0, "Shangri-La / Diqing (NW Yunnan)"),
        ],
    },

    # ── Coastal scenes ────────────────────────────────────────────────
    "south_china_coastal": {
        "conditions": {
            "scene_type": "coastal",
            "climate_zone": "subtropical",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(21.0, 27.0, 108.0, 118.0, "S China subtropical coast"),
        ],
    },
    "tropical_coastal": {
        "conditions": {
            "scene_type": "coastal",
            "climate_zone": "tropical",
            "tree_species": "coconut_palm",
        },
        "bboxes": [
            BBox(18.0, 20.5, 108.5, 111.5, "Hainan tropical coast"),
        ],
    },

    # ── Loess Plateau ─────────────────────────────────────────────────
    "loess_plateau_farmland": {
        "conditions": {
            "soil_color": "yellow",
            "climate_zone": "temperate",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(34.0, 39.0, 104.0, 112.0, "Loess Plateau (Shaanxi / Gansu / Shanxi)"),
        ],
    },

    # ── Sichuan Basin ─────────────────────────────────────────────────
    "chengdu_basin_urban": {
        "conditions": {
            "urbanization": "metropolis",
            "climate_zone": "subtropical",
            "terrain_type": "urban_flat",
            "tree_species": "camphor",
        },
        "bboxes": [
            BBox(30.3, 30.9, 103.8, 104.3, "Chengdu basin metro"),
        ],
    },
}


# ── Fuzzy matching rules for VLM common confusions ──────────────────────
# Maps (category, expected_value) → set of acceptable VLM output values

_FUZZY_RULES = {
    # soil_color: "yellow_brown" never matches because prompt only has
    # "yellow" and "brown" separately — accept either
    ("soil_color", "yellow_brown"): {"yellow", "brown"},

    # vegetation_zone: broadleaf family confusion
    ("vegetation_zone", "broadleaf_evergreen"): {"broadleaf_deciduous"},

    # vegetation_zone: conifer/mixed overlap
    ("vegetation_zone", "conifer_forest"): {"mixed_forest"},

    # terrain_type: karst visually similar to generic sharp mountains
    ("terrain_type", "karst_peaks"): {"sharp_mountains"},

    # terrain_type: sandstone pillars confused with karst
    ("terrain_type", "sandstone_pillars"): {"karst_peaks", "sharp_mountains"},

    # mountain_rock_type: granite prefix match
    ("mountain_rock_type", "granite_spheroidal"): {"granite_dome", "granite_tors"},

    # mountain_rock_type: red sandstone confused with quartz sandstone
    ("mountain_rock_type", "red_sandstone_danxia"): {"quartz_sandstone_pillars"},
    ("mountain_rock_type", "quartz_sandstone_pillars"): {"red_sandstone_danxia"},

    # scene_type: mountain trail sub-type confusion
    ("scene_type", "mountain_trail_karst"): {"mountain_trail_granite", "mountain_trail"},

    # climate_zone: alpine / boreal confusion (high elevation vs high latitude)
    ("climate_zone", "alpine"): {"boreal"},
    ("climate_zone", "boreal"): {"alpine"},

    # tree_species: confusable pairs
    ("tree_species", "camphor"): {"banyan"},           # both subtropical broadleaf
    ("tree_species", "chinese_red_pine"): {"conifer"},  # generic conifer
}


def _fuzzy_match(expected_val: str, actual_val, cat: str) -> bool:
    """Check if actual_val matches expected_val, with fuzzy rules for common VLM confusions."""
    if actual_val is None:
        return False
    # Handle list values (e.g. likely_provinces, elevation_estimate_m)
    if isinstance(actual_val, (list, dict)):
        return False
    if expected_val == actual_val:
        return True
    # Check fuzzy rules for this (category, expected) pair
    acceptable = _FUZZY_RULES.get((cat, expected_val), set())
    return actual_val in acceptable


def match_compound_scenes(elements: dict) -> list[tuple[str, list[BBox]]]:
    """Detect compound scenes matching the given geographic elements.

    Each compound scene requires ALL its conditions to match. If matched,
    its tight bboxes are returned as additional constraints.

    Returns:
        List of (scene_name, bboxes) tuples.
    """
    matches = []
    for name, scene in COMPOUND_SCENES.items():
        conditions = scene["conditions"]
        if all(
            elements.get(cat) == val
            for cat, val in conditions.items()
        ):
            matches.append((name, scene["bboxes"]))
    return matches


def match_compound_scenes_soft(
    elements: dict, min_ratio: float = 0.6, min_matches: int = 1,
) -> list[tuple[str, list[BBox], float]]:
    """Fuzzy compound scene matching with partial credit.

    Unlike match_compound_scenes() which requires exact string match on ALL
    conditions, this version:
      1. Uses fuzzy rules (_FUZZY_RULES) to accept VLM-common near-misses
      2. Returns a match_ratio = matched / total for partial matches
      3. Only returns scenes with match_ratio >= min_ratio AND matched >= min_matches

    min_matches=2 prevents 1-condition matches on 2-3 condition scenes
    from triggering false positives (e.g. Shanghai matching harbin_ne_urban
    with only temperate+medium_city=2/4).

    Returns:
        List of (scene_name, bboxes, match_ratio) tuples, sorted by ratio desc.
    """
    matches = []
    for name, scene in COMPOUND_SCENES.items():
        conditions = scene["conditions"]
        if not conditions:
            continue
        matched = 0
        for cat, expected_val in conditions.items():
            actual_val = elements.get(cat)
            if _fuzzy_match(expected_val, actual_val, cat):
                matched += 1
        ratio = matched / len(conditions)
        if ratio >= min_ratio and matched >= min_matches:
            matches.append((name, scene["bboxes"], ratio))
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches


# ═══════════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE TAGS — visible infrastructure clues from GeoCoT regional stage
# ═══════════════════════════════════════════════════════════════════════════

INFRASTRUCTURE_TAGS = {
    "electric_scooters": [
        BBox(22.0, 33.0, 104.0, 122.5, "S China e-scooter cities"),
    ],
    "tibetan_signs": [
        BBox(26.5, 34.0, 88.0, 99.5, "Tibet proper"),
        BBox(31.5, 39.5, 89.0, 103.5, "Qinghai / S Gansu"),
        BBox(26.0, 33.0, 97.0, 104.0, "W Sichuan (Ganzi/ABA)"),
        BBox(27.0, 29.5, 98.5, 101.0, "NW Yunnan (Diqing)"),
    ],
    "cable_cars": [
        BBox(18.0, 45.0, 97.0, 122.5, "Major Chinese mountain scenic areas"),
    ],
    "green_taxis": [
        BBox(22.5, 23.5, 113.0, 115.0, "Guangzhou / Pearl River Delta"),
        BBox(23.5, 26.0, 116.0, 121.0, "S Fujian (Xiamen area)"),
        BBox(30.5, 31.5, 121.0, 122.0, "Shanghai"),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# LANDFORM DETAILS — micro-geomorphology from GeoCoT local stage
# ═══════════════════════════════════════════════════════════════════════════

LANDFORM_DETAILS = {
    "limestone_tower_karst_scattered": [
        BBox(24.0, 26.0, 109.5, 111.5, "Guilin-Yangshuo scattered tower karst"),
    ],
    "limestone_tower_karst_dense": [
        BBox(24.5, 26.5, 104.5, 109.0, "S Guizhou / N Guangxi dense karst pinnacles"),
    ],
    "granite_spheroidal_rounded": [
        BBox(29.5, 31.0, 117.5, 119.0, "Huangshan / Sanqingshan rounded granite"),
        BBox(28.5, 30.0, 115.5, 117.0, "Lushan / Wudangshan rounded granite"),
    ],
    "granite_spheroidal_sharp": [
        BBox(28.5, 29.5, 117.5, 118.5, "Sanqingshan sharp granite"),
        BBox(34.0, 35.0, 109.5, 111.0, "Huashan sharp granite"),
        BBox(26.0, 32.0, 97.0, 104.0, "Hengduan sharp granite massif"),
    ],
    "quartz_sandstone_pillars": [
        BBox(28.5, 30.0, 109.0, 111.5, "Zhangjiajie / Wulingyuan quartz pillars"),
    ],
    "red_sandstone_danxia": [
        BBox(27.5, 28.5, 117.5, 118.5, "Wuyishan Danxia"),
        BBox(38.5, 39.5, 99.5, 101.0, "Zhangye Danxia"),
        BBox(24.5, 26.0, 113.0, 114.5, "Danxiashan (Guangdong)"),
    ],
    "snow_peaks_glaciers": [
        BBox(29.0, 30.5, 101.0, 103.0, "Gongga / Siguniang"),
        BBox(27.0, 29.0, 98.0, 100.5, "Meili / Yulong / Haba"),
        BBox(27.5, 29.0, 85.5, 87.5, "Everest / Cho Oyu"),
        BBox(30.0, 32.0, 90.0, 95.0, "Nyainqentanglha / Namcha Barwa"),
        BBox(41.0, 43.5, 80.0, 89.0, "Tianshan glaciers"),
    ],
    "alpine_meadow": [
        BBox(26.5, 36.5, 78.0, 99.5, "Tibet alpine meadow"),
        BBox(31.0, 39.5, 89.0, 103.5, "Qinghai alpine meadow"),
        BBox(26.0, 32.0, 97.0, 104.0, "W Sichuan / NW Yunnan alpine"),
        BBox(41.0, 45.0, 80.0, 96.5, "Tianshan alpine meadow"),
    ],
    "plateau_plain": [
        BBox(26.5, 36.5, 78.0, 99.5, "Tibet Plateau"),
        BBox(31.0, 39.5, 89.0, 103.5, "Qinghai Plateau"),
    ],
    "river_canyon": [
        BBox(26.0, 30.0, 98.5, 101.0, "Tiger Leaping Gorge / Nu River"),
        BBox(30.5, 31.5, 109.5, 111.5, "Three Gorges (Yangtze)"),
        BBox(29.0, 30.5, 94.0, 96.5, "Yarlung Tsangpo Grand Canyon"),
    ],
    "coastal_shore": [
        BBox(18.0, 42.0, 108.0, 123.0, "China coastline"),
    ],
    "urban_flat": [
        BBox(18.0, 48.5, 73.0, 135.5, "All China urban lowlands"),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# WATER TYPES — visible water bodies from GeoCoT local stage
# ═══════════════════════════════════════════════════════════════════════════

WATER_TYPES = {
    "ocean": [
        BBox(18.0, 42.0, 108.0, 123.0, "China east / south coastline"),
    ],
    "lake": [
        BBox(27.5, 33.0, 87.0, 91.5, "Tibet lakes (Namtso / Yamdrok)"),
        BBox(41.0, 49.0, 80.0, 92.0, "Xinjiang lakes (Bosten / Kanas)"),
        BBox(28.0, 34.0, 103.0, 111.0, "SW China lakes (Jiuzhaigou / Lugu)"),
        BBox(28.0, 33.0, 116.0, 121.0, "Yangtze Delta lakes (Taihu / Poyang)"),
        BBox(37.0, 39.0, 96.0, 101.0, "Qinghai Lake area"),
    ],
    "glacier": [
        BBox(27.5, 36.5, 78.0, 104.0, "Tibet / W Sichuan glaciers"),
        BBox(41.0, 44.0, 80.0, 89.0, "Tianshan glaciers"),
    ],
    "waterfall": [
        BBox(23.5, 30.5, 104.0, 112.0, "Guizhou / Guangxi waterfall region"),
        BBox(28.5, 30.0, 109.5, 111.5, "Zhangjiajie waterfalls"),
        BBox(24.5, 28.0, 100.5, 101.5, "Yunnan waterfalls"),
    ],
    "river": [
        BBox(18.0, 54.0, 73.0, 135.5, "All China (rivers are ubiquitous)"),
    ],
    "stream": [
        BBox(18.0, 54.0, 73.0, 135.5, "All China (streams are ubiquitous)"),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# ROCK COLORS — visible rock surface color from GeoCoT local stage
# ═══════════════════════════════════════════════════════════════════════════

ROCK_COLORS = {
    "grey_white": [
        BBox(26.0, 35.0, 97.0, 119.0, "Granite / limestone mountains (wide)"),
    ],
    "red_brown": [
        BBox(27.5, 28.5, 117.5, 118.5, "Wuyishan red sandstone"),
        BBox(38.5, 39.5, 99.5, 101.0, "Zhangye red beds"),
        BBox(24.5, 26.0, 113.0, 114.5, "Danxiashan red sandstone"),
        BBox(28.0, 29.0, 116.5, 117.5, "Longhushan red sandstone"),
        BBox(34.0, 41.0, 105.5, 114.5, "Loess Plateau red-brown earth"),
    ],
    "dark_grey": [
        BBox(26.5, 36.5, 78.0, 104.0, "Tibet / W Sichuan dark metamorphic rock"),
        BBox(26.0, 32.0, 97.0, 104.0, "Hengduan dark granite"),
        BBox(41.0, 44.0, 80.0, 89.0, "Tianshan dark metamorphic"),
    ],
    "yellow_brown": [
        BBox(34.5, 41.0, 105.5, 114.5, "Loess Plateau (Shaanxi / Shanxi / Gansu)"),
        BBox(24.0, 33.0, 108.5, 122.5, "Mid-China yellow-brown earth"),
    ],
    "black": [
        BBox(41.5, 42.5, 127.0, 129.0, "Changbaishan black basalt"),
        BBox(19.0, 20.5, 109.5, 111.5, "Hainan basalt"),
    ],
    "white": [
        BBox(34.0, 42.5, 88.0, 96.5, "Qaidam Basin white saline soil"),
        BBox(27.5, 36.5, 78.0, 88.0, "W Tibet white evaporite"),
    ],
    "mixed": [
        BBox(18.0, 54.0, 73.0, 135.5, "All China (non-specific)"),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# SCENE TYPES — high-level scene category from GeoCoT local stage
# ═══════════════════════════════════════════════════════════════════════════

SCENE_TYPE_REGIONS = {
    "desert": [
        BBox(34.0, 49.5, 73.0, 107.0, "NW China deserts (Taklamakan / Gobi / Tengger)"),
    ],
    "grassland": [
        BBox(41.0, 51.0, 110.0, 126.5, "Inner Mongolia grassland"),
        BBox(32.0, 39.5, 89.0, 103.5, "Qinghai / Gansu alpine grassland"),
        BBox(27.0, 33.0, 97.0, 104.0, "W Sichuan / NW Yunnan alpine grassland"),
        BBox(41.0, 49.0, 80.0, 92.0, "N Xinjiang grassland (Ili / Altai)"),
    ],
    "coastal": [
        BBox(18.0, 42.0, 108.0, 125.0, "China east / south coastline"),
    ],
    "river_valley": [
        BBox(24.0, 35.0, 96.0, 118.0, "Hengduan river canyons + Three Gorges"),
    ],
    "city_street": [
        BBox(18.0, 48.5, 73.0, 135.5, "All China urban"),
    ],
    "mountain_trail_karst": [
        BBox(23.5, 30.5, 103.5, 115.0, "S China karst belt"),
    ],
    "mountain_trail_granite": [
        BBox(26.0, 35.0, 97.0, 119.0, "China granite mountain belt"),
        BBox(29.0, 31.0, 117.0, 119.0, "Huangshan / Sanqingshan granite"),
        BBox(33.5, 35.0, 107.0, 111.0, "Huashan / Qinling granite"),
    ],
    "mountain_trail_snow": [
        BBox(26.5, 36.5, 78.0, 104.0, "Tibet / W Sichuan snow mountains"),
        BBox(41.0, 45.0, 80.0, 96.5, "Tianshan snow mountains"),
    ],
    "rural_village": [
        BBox(22.0, 45.0, 104.0, 126.5, "Agricultural China (dense rural)"),
    ],
    "tourist_scenic": [
        BBox(18.0, 45.0, 73.0, 135.5, "All China scenic areas"),
    ],
    "wilderness_forest": [
        BBox(18.0, 54.0, 108.0, 135.5, "E China forest belt"),
        BBox(26.0, 34.5, 97.0, 104.0, "Hengduan forest"),
        BBox(48.0, 54.0, 121.0, 135.5, "NE China boreal forest"),
        BBox(41.0, 45.0, 80.0, 96.5, "Tianshan spruce forest"),
    ],
}


PROVINCE_BBOXES = {
    # Province-level bounding boxes for provincial anchoring
    "Beijing":       [BBox(39.4, 41.0, 115.4, 117.5, "Beijing")],
    "Tianjin":       [BBox(38.5, 40.2, 116.6, 118.0, "Tianjin")],
    "Shanghai":      [BBox(30.6, 31.9, 120.8, 122.0, "Shanghai")],
    "Chongqing":     [BBox(28.1, 32.2, 105.3, 110.2, "Chongqing")],
    "Hebei":         [BBox(36.0, 42.6, 113.5, 119.9, "Hebei")],
    "Shanxi":        [BBox(34.5, 40.8, 110.2, 114.5, "Shanxi")],
    "Liaoning":      [BBox(38.7, 43.5, 118.8, 125.8, "Liaoning")],
    "Jilin":         [BBox(40.8, 46.3, 121.6, 131.3, "Jilin")],
    "Heilongjiang":  [BBox(43.4, 53.6, 121.2, 135.1, "Heilongjiang")],
    "Jiangsu":       [BBox(30.7, 35.1, 116.3, 122.0, "Jiangsu")],
    "Zhejiang":      [BBox(27.0, 31.2, 118.0, 123.0, "Zhejiang")],
    "Anhui":         [BBox(29.4, 34.7, 114.9, 119.6, "Anhui")],
    "Fujian":        [BBox(23.5, 28.7, 115.8, 120.7, "Fujian")],
    "Jiangxi":       [BBox(24.5, 30.1, 113.6, 118.5, "Jiangxi")],
    "Shandong":      [BBox(34.3, 38.3, 114.8, 122.7, "Shandong")],
    "Henan":         [BBox(31.4, 36.4, 110.3, 116.7, "Henan")],
    "Hubei":         [BBox(29.0, 33.3, 108.3, 116.2, "Hubei")],
    "Hunan":         [BBox(24.6, 30.1, 108.8, 114.2, "Hunan")],
    "Guangdong":     [BBox(20.2, 25.5, 109.6, 117.3, "Guangdong")],
    "Guangxi":       [BBox(21.4, 26.3, 104.5, 112.0, "Guangxi")],
    "Hainan":        [BBox(18.1, 20.3, 108.6, 111.0, "Hainan")],
    "Sichuan":       [BBox(26.0, 34.3, 97.3, 108.5, "Sichuan")],
    "Guizhou":       [BBox(24.6, 29.2, 103.6, 109.6, "Guizhou")],
    "Yunnan":        [BBox(21.1, 29.3, 97.5, 106.2, "Yunnan")],
    "Tibet":         [BBox(26.8, 36.5, 78.4, 99.1, "Tibet (Xizang)")],
    "Shaanxi":       [BBox(31.7, 39.6, 105.5, 111.3, "Shaanxi")],
    "Gansu":         [BBox(32.5, 42.8, 92.7, 108.8, "Gansu")],
    "Qinghai":       [BBox(31.5, 39.3, 89.4, 103.1, "Qinghai")],
    "Xinjiang":      [BBox(34.3, 49.2, 73.4, 96.4, "Xinjiang")],
    "Inner Mongolia":[BBox(37.4, 53.4, 97.2, 126.1, "Inner Mongolia")],
    "Ningxia":       [BBox(35.2, 39.3, 104.3, 107.6, "Ningxia")],
    "Hong Kong":     [BBox(22.1, 22.6, 113.8, 114.4, "Hong Kong")],
    "Macau":         [BBox(22.0, 22.3, 113.4, 113.7, "Macau")],
    "Taiwan":        [BBox(21.9, 25.3, 120.0, 122.0, "Taiwan")],
}

URBANIZATION_LEVELS = {
    "metropolis": [
        # Tier-1 megacities (>10M urban pop) — tight ~0.5° bboxes
        BBox(39.6, 40.2, 116.0, 116.8, "Beijing"),
        BBox(30.9, 31.6, 121.1, 121.8, "Shanghai"),
        BBox(22.4, 23.3, 113.0, 114.4, "Guangzhou / Shenzhen / Dongguan"),
        BBox(30.3, 30.9, 103.8, 104.3, "Chengdu"),
        BBox(29.2, 29.8, 106.2, 106.8, "Chongqing"),
        BBox(30.2, 30.9, 113.9, 114.7, "Wuhan"),
        BBox(31.7, 32.5, 118.5, 119.0, "Nanjing"),
        BBox(30.0, 30.5, 119.9, 120.5, "Hangzhou"),
        BBox(38.8, 39.4, 117.0, 117.7, "Tianjin"),
        BBox(34.0, 34.5, 108.6, 109.3, "Xi'an"),
        BBox(28.0, 28.4, 112.7, 113.3, "Changsha"),
        BBox(24.2, 24.8, 117.8, 118.5, "Xiamen"),
        BBox(22.5, 23.3, 112.8, 113.7, "Foshan / Zhongshan / Zhuhai"),
    ],
    "medium_city": [
        # Provincial capitals and prefecture cities (~50-100km bboxes)
        BBox(45.3, 46.1, 126.3, 127.2, "Harbin"),
        BBox(43.5, 44.3, 125.0, 126.0, "Changchun / Jilin"),
        BBox(41.5, 42.1, 123.0, 124.0, "Shenyang"),
        BBox(36.3, 36.9, 116.8, 117.4, "Jinan"),
        BBox(34.3, 35.0, 113.2, 114.0, "Zhengzhou"),
        BBox(37.5, 38.2, 112.2, 113.0, "Taiyuan"),
        BBox(31.5, 32.4, 116.3, 117.5, "Hefei"),
        BBox(28.3, 29.0, 115.5, 116.2, "Nanchang"),
        BBox(25.8, 26.4, 119.0, 119.7, "Fuzhou"),
        BBox(22.5, 23.2, 107.8, 108.8, "Nanning"),
        BBox(19.7, 20.3, 109.8, 110.7, "Haikou"),
        BBox(24.8, 25.6, 102.3, 103.1, "Kunming"),
        BBox(26.2, 26.9, 106.3, 107.1, "Guiyang"),
        BBox(36.3, 37.0, 103.3, 104.3, "Lanzhou"),
        BBox(36.2, 37.0, 101.3, 102.3, "Xining"),
        BBox(43.3, 44.2, 87.2, 88.3, "Urumqi"),
        BBox(38.3, 39.0, 106.0, 106.7, "Yinchuan"),
        BBox(29.3, 29.9, 90.7, 91.5, "Lhasa"),
        BBox(48.0, 48.5, 88.5, 89.5, "Altay"),
        BBox(49.0, 49.5, 119.4, 120.0, "Hailar / Hulunbuir"),
        BBox(31.0, 31.8, 108.0, 108.8, "Dazhou / Wanzhou"),
        BBox(25.8, 26.5, 113.0, 113.6, "Chenzhou / Hengyang"),
        BBox(32.0, 32.8, 119.0, 120.0, "Yangzhou / Zhenjiang"),
        BBox(27.3, 28.0, 119.3, 120.5, "Wenzhou / Ningde"),
    ],
    "small_town": [
        # County towns — most populated areas of China
        BBox(18.0, 48.0, 97.0, 135.5, "E/SE China populated belt"),
    ],
    "village": [
        # Rural villages — similar to small_town but more widespread
        BBox(18.0, 48.0, 97.0, 135.5, "Rural settled China"),
    ],
    "rural": [
        # Agricultural countryside — China's populated plains and valleys
        BBox(18.0, 48.0, 97.0, 135.5, "Populated China"),
        # But exclude true wilderness — rural doesn't mean uninhabited desert
    ],
    "wilderness": [
        # True uninhabited / very sparsely populated areas
        BBox(26.5, 36.5, 78.0, 95.5, "NW Tibet / W Qinghai uninhabited"),
        BBox(36.5, 42.0, 78.0, 88.0, "Taklamakan Desert"),
        BBox(40.0, 47.0, 88.0, 99.0, "Gobi / E Xinjiang"),
        BBox(27.0, 32.0, 97.0, 100.5, "Hengduan canyons wilderness"),
        BBox(44.0, 49.0, 85.0, 90.0, "Altai wilderness"),
        BBox(33.0, 37.0, 88.0, 95.0, "Qaidam Basin / Hoh Xil"),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# Lookup helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_bboxes_for_element(category: str, value) -> list[BBox]:
    """Get bounding boxes for a geographic element value.

    Args:
        category: one of 'climate_zone', 'terrain_type', 'vegetation_zone',
                  'tree_species', 'architecture_style', 'language_script',
                  'mountain_rock_type', 'soil_color', 'sky_quality'
        value: the element value (str or list of str)

    Returns:
        List of BBox objects, empty if not found.
    """
    kb_map = {
        "climate_zone": CLIMATE_ZONES,
        "terrain_type": TERRAIN_TYPES,
        "vegetation_zone": VEGETATION_ZONES,
        "tree_species": TREE_SPECIES,
        "architecture_style": ARCHITECTURE_STYLES,
        "language_script": LANGUAGE_SCRIPTS,
        "mountain_rock_type": MOUNTAIN_ROCK_TYPES,
        "soil_color": SOIL_COLORS,
        "sky_quality": SKY_QUALITY,
        "scene_type": SCENE_TYPE_REGIONS,
        "compound_scene": {},  # dynamic, handled via match_compound_scenes()
        "landform_detail": LANDFORM_DETAILS,
        "water_type": WATER_TYPES,
        "infrastructure_tags": INFRASTRUCTURE_TAGS,
        "rock_color": ROCK_COLORS,
        "urbanization": URBANIZATION_LEVELS,
        "likely_provinces": PROVINCE_BBOXES,
    }

    kb = kb_map.get(category, {})
    if kb is None or value is None:
        return []

    # Handle list values
    if isinstance(value, list):
        result = []
        for v in value:
            result.extend(kb.get(v, []))
        return result

    # Handle dict values (e.g. water_type with nested fields)
    if isinstance(value, dict):
        return []

    return kb.get(value, [])


def get_all_categories() -> list[str]:
    """Return all supported geographic element categories."""
    return [
        "climate_zone", "terrain_type", "vegetation_zone",
        "tree_species", "architecture_style", "language_script",
        "mountain_rock_type", "soil_color", "sky_quality",
    ]
