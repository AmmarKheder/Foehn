# Figures added or redrawn in version 1.1.0

| Folder | Figure of the paper |
|---|---|
| `fig1_v2/` (and `fig1/`, its first version) | Fig. 1, the schematic with the real-time system (d) and the downscaling (e) |
| `fig2_v3/fig_hidden_v4.py` | Fig. 2, hidden exposure, with the warnings panel f |
| `fig_rt_day/` | Fig. 8, the two real-time cases chosen by rule (Lahore and Delhi, 11 November 2026; Karachi, 10 November 2026) |
| `fig_rt_skill/figS_rt_proof.py` | Supplementary figure of the real-time verification at the monitors (RMSE by lead day); `rt_skill_data.py` reads the scorer's tables |
| `source_data/build_source_data.py` | the Source Data workbook (one sheet per graph panel) |

The other figure scripts of the paper are one level up, in `figures/`.

The scripts are the working files of the study, as they ran on the LUMI supercomputer: they carry the paths of our project space and the Slurm settings of that machine, like the rest of this repository. Data, trained weights, logs and outputs are not in the repository (see *Data availability* and *Code availability* in the paper).

## Files
| File | What it does (first line of its own header) |
|---|---|
| `fig1/check_overlay.py` | Overlay check of fig_gc1_schematic_rt: are panels a, b, c the Nature figure, pixel for pixel? A. Render of the original against the new figure (the proof) |
| `fig1/fig_gc1_schematic_rt.py` | Fig |
| `fig1_v2/check_overlay.py` | Overlay check of fig_gc1_schematic_rt2 (version 2 of the lower band): are panels a, b, c the Nature figure, pixel for pixel? A. Render of the original against the new figure (the proof) |
| `fig1_v2/fig_gc1_schematic_rt2.py` | Fig |
| `fig1_v2/prep_downscaler_fields.py` | Fields of panel e of the new Fig |
| `fig2_v3/fig_hidden_v4.py` | Fig |
| `fig_rt_day/fig_rt_day.py` | Main figure: a real-time forecast at 1 km (house style of fig_overview.py and fig_hidden.py; data from rt_day_data.py) |
| `fig_rt_day/legend_count.py` | Word count of legend.tex with the rules of the paper's counting tool (build_test/wordcount.py, imported read-only) |
| `fig_rt_day/render.sh` | renders the figure on a login node (light: the world map is drawn from every 5th 1 km pixel); usage: ./render.sh [variant] |
| `fig_rt_day/rt_day_data.py` | Data of the figure 'a real-time forecast at 1 km': the two case rules, the fields of every panel, the numbers a legend may quote |
| `fig_rt_day/rt_loaders.py` | Read-only loaders of the real-time forecasts for the figure 'a real-time forecast at 1 km' (after Appendix A of facts/figures.md; the arithmetic of the monitor comparison is that of the live verification script live_baselines.py) |
| `fig_rt_day/rt_safe_io.py` | Safe writes for the files of this figure folder (the file system dropped writes on 2026-11-19): every file is written under a temporary name in the same directory, flushed, fsynced, read back and compared, then renamed with os.rep … |
| `fig_rt_skill/check_ensemble_cams.py` | Does any quantity drawn or quoted for the ensemble figure (figS_rt_ensemble.py, numbers_cal.json, key f_ensemble) depend on the CAMS arm? Method: (1) the station-day tables of score_cal (CAMS = three instants) and of score … |
| `fig_rt_skill/check_plain_numpy.py` | Independent check of the rebuilt figure data with plain numpy (no scorer code): from the station-day table of score_cams3h, recompute the per-cell RMSE and the share of cells of panel a, the per-issue-day changes of panel e, the R … |
| `fig_rt_skill/figS_rt_ensemble.py` | Figure not in the submitted manuscript: the eight members of the real-time 1 km forecast at the monitors, as issued and with the calibrated noise (the panel f of fig_rt_skill.py, drawn alone; same numbers: numbers_cal.json, key f_ensemble, made by  … |
| `fig_rt_skill/figS_rt_proof.py` | The real-time forecasts verified at the monitors, ONE small panel: RMSE by lead day of the real-time 1 km forecast, of its 0.4 degree input, of the CAMS forecast of t … |
| `fig_rt_skill/fig_rt_skill.py` | Main figure: the real-time 1 km forecasts verified at the monitors (house style of the paper; companion of fig_rt_day) |
| `fig_rt_skill/fig_rt_skill_main.py` | Main figure: the real-time 1 km forecasts verified at the monitors (house style of the paper; companion of fig_rt_day) |
| `fig_rt_skill/legend_check.py` | Numbers of the legends of fig_rt_skill against numbers.json, in both directions (reads only; writes nothing) |
| `fig_rt_skill/legend_count.py` | Word count of legend.tex with the rules of the paper's counting tool (build_test/wordcount.py, imported read-only) |
| `fig_rt_skill/render.sh` | fig_rt_skill on a login node (CPU only, about 1 min in all); usage: ./render.sh [--data [options of rt_skill_data.py]] [tag] ./render.sh draws fig_rt_skill.pdf / .png from numbers.json and data/rt_skill.npz (15 s) ./render.sh --da … |
| `fig_rt_skill/rt_skill_data.py` | Data of fig_rt_skill: the real-time 1 km forecasts verified at the monitors (48 issue days, lead days 1-5) |
| `source_data/build_source_data.py` | Source Data of "Global probabilistic forecasting of fine particulate air pollution at the kilometre scale". Builds, from the data files that the figure scripts read: Source_Data.xlsx one sheet per figure panel with graph data (fir … |
