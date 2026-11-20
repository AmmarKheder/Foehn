"""Regions of the hidden-exposure analyses (written 2026-11-19 for the Nature Communications version).

Two definitions are used side by side:
  BOXES   the latitude-longitude boxes of the paper's figures (figdata_exposure.py:REGIONS, the same boxes as the regional
          scores of the forecasts). They OVERLAP: 'Africa' (37N-35S, 20W-52E) contains the Arabian Peninsula, the Levant, Iraq
          and western Iran, which are also in 'Middle East' (42N-12N, 35E-63E); the eight boxes sum to 106 % of the world
          count of hidden people and leave Central Asia out.
  GROUPS  a partition of the world by country (country raster of Kummu et al. 2025 at 5 arc-minutes, the one of
          figdata_exposure_income.py), in the geographic regions of the United Nations M49 standard with two changes stated
          here: Iran is counted with the Middle East (M49: Southern Asia), and Turkiye, Cyprus and the Caucasus are counted
          with Europe (M49: Western Asia). Every country is in exactly one group, so the groups sum to the world.
"""
import numpy as np

BOXES = {"World": None, "South Asia": (37, 5, 60, 97), "East Asia": (54, 18, 97, 146), "Africa": (37, -35, -20, 52),
         "Middle East": (42, 12, 35, 63), "Southeast Asia": (18, -11, 92, 142), "Europe": (72, 35, -25, 45),
         "North America": (72, 15, -170, -50), "South America": (15, -56, -95, -30)}

_G = {
    "South Asia": "AFG BGD BTN IND MDV NPL PAK LKA",
    "East Asia": "CHN HKG TWN JPN KOR PRK MNG",
    "Southeast Asia": "BRN KHM IDN LAO MYS MMR PHL SGP THA TLS VNM",
    "Central Asia": "KAZ KGZ TJK TKM UZB",
    "Middle East": "BHR IRN IRQ ISR JOR KWT LBN OMN QAT SAU SYR ARE PSE YEM",
    "North Africa": "DZA EGY LBY MAR SDN TUN ESH",
    "Sub-Saharan Africa": ("AGO BEN BWA BFA BDI CPV CMR CAF TCD COM COD COG CIV DJI GNQ ERI SWZ ETH GAB GMB GHA GIN GNB KEN LSO "
                           "LBR MDG MWI MLI MRT MUS MOZ NAM NER NGA RWA STP SEN SYC SLE SOM ZAF SSD TZA TGO UGA ZMB ZWE REU MYT SHN"),
    "Europe": ("ALB AND ARM AUT AZE BLR BEL BIH BGR HRV CYP ZNC CZE DNK EST FRO FIN FRA GEO DEU GIB GRC HUN ISL IRL IMN ITA "
               "KSV LVA LIE LTU LUX MLT MDA MCO MNE NLD MKD NOR POL PRT ROU RUS SMR SRB SVK SVN ESP SWE CHE TUR UKR GBR GGY JEY SJM"),
    "North America": "USA CAN BMU GRL SPM",
    "Latin America and the Caribbean": ("ATG ARG ABW BHS BRB BLZ BOL BRA VGB CYM CHL COL CRI CUB CUW DMA DOM ECU SLV GRD GTM GUY "
                                        "HTI HND JAM MEX NIC PAN PRY PER PRI SXM KNA LCA MAF VCT SUR TTO TCA URY VEN VIR GLP MTQ "
                                        "GUF AIA BLM MSR BES FLK"),
    "Oceania": "AUS NZL FJI KIR MHL FSM NRU NCL PLW PNG WSM SLB TON TUV VUT ASM PYF GUM MNP TKL WLF",
}
GROUPS = {k: v.split() for k, v in _G.items()}
GROUP_NAMES = list(GROUPS) + ["Unassigned"]                        # 'Unassigned': pixels outside the country raster (coasts)
AFRICA = ("North Africa", "Sub-Saharan Africa")                    # the continent = these two groups

INC = "/scratch/project_462001804/ammar/eccv/data/world/income_derived/income_5arcmin.npz"


def country_to_group(iso3):
    """index of the group of every country code of the raster (code c is iso3[c - 1]; code 0 and unnamed codes -> Unassigned)"""
    lut = np.full(len(iso3) + 1, len(GROUPS), np.int64)
    seen = {}
    for g, (name, lst) in enumerate(GROUPS.items()):
        for s in lst:
            assert s not in seen, (s, name, seen[s])
            seen[s] = name
    for c, s in enumerate(iso3, start=1):
        s = str(s).strip()
        if s:
            assert s in seen, f"country {s} has no group"
            lut[c] = list(GROUPS).index(seen[s])
    missing = set(seen) - {str(s).strip() for s in iso3}
    assert not missing, f"codes absent from the raster: {missing}"
    return lut


def pixel_groups(rows, cols):
    """group index of 1 km pixels (rows, cols of the 14,400 x 36,000 grid from 84N, 180W) from the 5 arc-minute country raster"""
    z = np.load(INC)
    lut = country_to_group(z["iso3"])
    r5 = np.floor((6.0 + 0.01 * (rows + 0.5)) * 12).astype(np.int64)
    c5 = np.floor(0.01 * (cols + 0.5) * 12).astype(np.int64)
    return lut[z["country"][r5, c5].astype(np.int64)]


if __name__ == "__main__":
    z = np.load(INC)
    lut = country_to_group(z["iso3"])
    print("groups:", {n: int((lut[1:] == g).sum()) for g, n in enumerate(GROUP_NAMES)})
