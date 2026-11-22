# Figures and Source Data of the manuscript as submitted

| Folder | Figure or item of the paper |
|---|---|
| `fig1/` | Fig. 1, *Foehn computes where the air comes from and learns what it carries* |
| `fig2/` | Fig. 2, *Foehn's warnings reach more of the people whom a 40 km grid hides* (panel f with eight members) |
| `fig3/` | Fig. 3, *Foehn forecasts all inhabited land at 1 km and lowers the error at the stations* (9 February 2022) |
| `fig4/` | Fig. 4, *In Aurora's own protocol, Foehn's design has a lower error than Aurora 3 and 5 days ahead* |
| `fig5/` | Fig. 5, the North China Plain haze of 24 January 2022 |
| `fig6/` | Fig. 6, *The computed transport traces a dust episode to its origin* |
| `fig7/` | Fig. 7, *Foehn gives a bird reserve a one-in-five probability of harmful smoke 5 days ahead* |
| `fig8/` | Fig. 8, *Issued in real time, the 1 km forecast resolves what the 40 km forecasts average* |
| `render/` | render wrappers (a figure script of `figures/` drawn as in the manuscript, under a noise design, without overwriting a stored figure) and Supplementary Fig. 11, *On 25 January 2022, each of Foehn's members keeps the sharp structure of the haze that the operational forecasts smooth away* (`fig_gc2_realism_letters.py`) |
| `showcase/` | how the days of Figs. 3 and 5 were chosen (candidates, the tally over all days, the re-run forecast of 9 February 2022) and independent recomputations of the plotted values |
| `source_data/` | the Source Data workbook of the manuscript (`build_final.py` runs `build_source_data.py` under `FOEHN_DESIGN=ar`; `check_final.py` checks it against the manuscript) |
| `numbers/` | values printed in the paper that the analysis chain does not write directly (`blocked.py`), and the headline quantities under the two noise designs (`old_vs_new.py`, `leafdiff.py`) |
| `tables/` | the portrait versions of the long Supplementary Tables (`si_tables_portrait.py`) |

Supplementary Fig. 8, *Foehn forecasts all inhabited land at 1 km and resolves basins that the operational grid blurs: the
forecast for 21 January 2022*, is `figures/fig_overview.py` rendered by `analyses/noise_design/rebuild_chain/step9.sh` under
`FOEHN_DESIGN=ar`; Supplementary Fig. 11 is `render/fig_gc2_realism_letters.py` rendered by `render/render_asold.py` under
`FOEHN_DESIGN=ar`. The scripts of `figures/` and `figures/version_1_1/` that these import or that the analysis chain renders
were updated in this version for the noise design switch (`figures/final_model.py`: `FOEHN_DESIGN=ar` adds the suffix `_ar`
to the evaluation folders of the final model and to the label of every derived file). Four scripts of `figures/` are new in
the repository: `fig_gc1_schematic.py` (panels a-c of Fig. 1, imported by `version_1_1/fig1/fig_gc1_schematic_rt.py` and
`version_1_1/fig1_v2/fig_gc1_schematic_rt2.py`; `fig1/fig_gc1_schematic_polish.py` is its copy for the final render) with
the builders of its data, `figdata_arch2.py` and `figdata_arch_fields.py`, and `extremes_stations_own.py` (local extremes
at the stations with thresholds of each system's own).

Working folders (under `$FOEHN_WORK`): `fig*/`, `showcase/numbers_show.py`, `showcase/show_numbers.py` and `showcase/run/` are
in `eccv/cran-pm/paper/natcomms/rebuild/figs_show` or its sub-folders; `render/` is
`eccv/cran-pm/paper/natcomms/rebuild/print/figs_letters`; the other files of `showcase/`, `source_data/build_final.py`,
`source_data/check_final.py` and `numbers/` are in `eccv/cran-pm/paper/natcomms/rebuild/print` or its sub-folders
(`showcase/recompute_fig3_fig5.py` has another name there); `source_data/build_source_data.py` is in
`eccv/cran-pm/paper/natcomms/source_data` and `tables/` in `eccv/cran-pm/paper/natcomms/tools`.

**Noise design switch.** `FOEHN_DESIGN=ar` selects the design of the paper (the noise field of each member drawn anew
at every start day and evolving between lead days as a first-order autoregressive process, rho = 0.59); every run folder
and derived file of that design carries the suffix `_ar`. With `FOEHN_DESIGN` unset, the scripts read and write the files of
the first design (fields drawn once), as in version 1.1.0.

**Paths and environment.** The scripts are the working files of the study as they ran on the LUMI supercomputer,
with their absolute paths replaced by two environment variables (except the scorers fixed before their runs, released as they
ran): `FOEHN_WORK`, the folder that holds the working trees, and `FOEHN_ENVS`, the folder of the Python environments (`world`, a numpy/xarray virtual environment, and `torchenv`, used inside
the PyTorch container of LUMI). Both must be set: a Python script stops with `KeyError` and a shell script with a message when
one is missing. The scripts expect the layout of the working project under `$FOEHN_WORK`, not that of this repository:
`eccv/cran-pm/experiments/global1km` (here `foehn/`), `eccv/cran-pm/experiments/figures` (`figures/`), the manuscript
workspace `eccv/cran-pm/paper/natcomms` (its `analyses/` is `analyses/` here, its `rebuild/` folders are mapped in the tables
of the READMEs) and the downscaling project `global_model_forecasting_air_pollution` (its downscaling folders are
`downscaling/` here; its experiment folders are mapped in the tables). The folder of the real-time 1 km forecast is named
`realtime_1km`, as in version 1.1.0, and must be reachable under that name. Slurm scripts carry the account placeholder
`<project>` (submit with `sbatch --account=...`) and the GPU jobs run in the PyTorch container image of LUMI
(`/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif`). Data, trained weights, logs
and outputs are not in the repository (see *Data availability* and *Code availability* in the paper).

**Jobs.** Each Slurm script is submitted from the working folder below (under `$FOEHN_WORK`): the folder that holds
its `jobs/` folder, or the script's own folder, as its usage lines show. Its `#SBATCH --output` path is relative to
that folder, and scripts that read their own log back find it only when submitted from there.

| Submitted from (under `$FOEHN_WORK`) | Scripts |
|---|---|
| `eccv/cran-pm/paper/natcomms/rebuild/figs_show/run` | `showcase/run/show_d38.sh` |

**Not in this release.**

- `safe_edit.py`, the exact-string editing helper of the manuscript workspace, imported by `tables/si_tables_portrait.py`.

## Files
| File | What it does |
|---|---|
| `fig1/fig_gc1_schematic_polish.py` | panels a-c of Fig. 1 (computed transport, learned tracers, generator), imported by the next script |
| `fig1/fig_gc1_schematic_rt2_polish.py` | Fig. 1 as submitted: panels a-c and the real-time panels d and e |
| `fig2/fig_hidden_v4_polish.py` | Fig. 2 as submitted: a-e hidden exposure; f warnings 1 day ahead, Foehn with eight members |
| `fig3/fig_overview_final3.py` | Fig. 3 as submitted: the 1 km forecast of 9 February 2022, the four basins and the RMSE at the six networks |
| `fig4/fig_gc5_aurora_setup_polish.py` | Fig. 4 as submitted: Foehn trained in Aurora's protocol (Foehn-CAMS) against Aurora and CAMS |
| `fig5/fig_gc2_realism_final2.py` | Fig. 5 as submitted: the North China Plain haze, 24 January 2022 |
| `fig5/render_fig5_final2.py` | renders Fig. 5 as the figures of the manuscript were rendered, without overwriting a file |
| `fig6/fig_transport_polish.py` | Fig. 6 as submitted: the Kano dust episode (origins of the air, estimates, probabilities, transport gain) |
| `fig6/render_polish.py` | renders Fig. 6 as the figures of the manuscript were rendered, without overwriting a file |
| `fig7/fig_smoke_polish.py` | Fig. 7 as submitted: the Gironde fires of July 2022 |
| `fig7/render_polish.py` | renders Fig. 7 as the figures of the manuscript were rendered, without overwriting a file |
| `fig8/fig_rt_day_polish.py` | Fig. 8 as submitted: one real-time 1 km forecast |
| `numbers/blocked.py` | values for three places of the manuscript that the analysis chain does not write directly |
| `numbers/leafdiff.py` | leaf-by-leaf comparison of the stored outputs of the analysis chain with those of a noise design |
| `numbers/old_vs_new.py` | the headline quantities of the paper under the two noise designs |
| `render/fig_gc2_realism_letters.py` | Supplementary Fig. 11 as submitted (25 January 2022): `figures/fig_gc2_realism.py` with the letter placement of the manuscript, rendered by `render_asold.py` |
| `render/render_asold.py` | renders a figure script of `figures/` as the figures of the manuscript were rendered, under a noise design, without overwriting a stored figure |
| `showcase/numbers_show.py` | numbers of Fig. 5 and Supplementary Fig. 11 (24 and 25 January 2022: box means, ranks of members 1-3, exceedance probabilities) |
| `showcase/recheck_final2.py` | independent recomputation of the values of Figs. 3 and 5 (band values; RMSE at the stations of the North China Plain on 24 January 2022) |
| `showcase/recompute_fig3_fig5.py` | integrity check of Figs. 3 and 5: the plotted values recomputed from the inputs of `figures/compare_fair.py` |
| `showcase/rerun_9feb_station_mean.py` | the re-run 9 February 2022 forecast against the evaluated one: difference of the ensemble means on the 0.25 degree cells and at the station pixels |
| `showcase/run/eval_show.py` | the evaluator of the noise design with the 1 km map of one forecast saved (the re-run forecast of 9 February 2022) |
| `showcase/run/show_d38.sh` | the GPU job of that re-run: start day 8 February 2022, lead day 1, as the evaluation run of 2022 |
| `showcase/show_numbers.py` | numbers behind Fig. 3: identity of the re-run 9 February 2022 forecast with the evaluated one, the four basins on every day, panel c |
| `showcase/stations_plain_22days.py` | the North China Plain case at the official stations of the plain (8-29 January 2022, 16 members): RMSE of Foehn, the CAMS forecast and Aurora 1, 3 and 5 days ahead, the tallies of the days and the values of 24 and 25 January |
| `showcase/verify_A.py` | independent check of the North China Plain case (8-29 January 2022, 16 members): the tallies of the days against the 1 km estimates |
| `showcase/verify_BC.py` | independent check of the station scores of the six official networks in 2022 (final model, design ar) |
| `source_data/build_final.py` | the Source Data workbook of the manuscript as submitted: runs `build_source_data.py` under `FOEHN_DESIGN=ar` |
| `source_data/build_source_data.py` | the builder of the Source Data, one sheet per panel with data |
| `source_data/check_final.py` | checks of the Source Data against the manuscript text and the printed figure labels |
| `tables/si_tables_portrait.py` | portrait, shortened versions of the long Supplementary Tables |
