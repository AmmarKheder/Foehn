#!/bin/bash
#SBATCH --job-name=ghap_plumb
#SBATCH --account=project_462001804
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/logs/plumb_%j.out
# Code test (no truth read, no 2022 truth, no station file): the 2022 prediction script and the RT 1 km GHAP switch with
# the smoke checkpoint of pmG_base (step 2,000) -> runs_ghap/plumb/ (deleted after the check). Not a result.
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
T=$W/runs_ghap/plumb
S=$W/logs/GLOBAL_PREP_STATUS.md
source $W/env.sh
cd $W
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $T/sys/ckpt
cp $W/runs_ghap/pmG_base/ckpt/step002000.ckpt $T/sys/ckpt/best.ckpt
cp $W/runs_ghap/pmG_base/args.json $T/sys/args.json
PMAX=$(python3 -c "import json;print(json.load(open('$W/data/processed/ghap/manifest.json'))['P_max_ug'])")
echo "{\"z_bound\": 10.0, \"pmax_ug\": $PMAX, \"z0_trunc\": 3.0}" > $T/sys/SAMPLER.json
/projappl/project_462001804/envs/world/bin/python -I -c "
import numpy as np, json
n = json.load(open('$W/data/processed/ghap/manifest.json'))['mask']['Nm']
np.savez('$T/st_fake.npz', pos=np.sort(np.random.default_rng(0).choice(n, 5000, replace=False)).astype(np.int64))"
srun8() { srun --ntasks=8 --cpus-per-task=7 --cpu-bind=$CDFFM_CPU_BIND_8 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash -c "source $CDFFM_ENVSH; cd $CDFFM_W; $*"; }
T0=$(date +%s)
srun8 "python -u test2022_ghap/predict_2022.py --system $T/sys --days 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --full-every 16 --out $T/pred --st-pos $T/st_fake.npz"
rc1=$?
T1=$(date +%s)
srun8 "python -u realtime_1km/infer.py 2026-10-28 --ckpt $T/sys/ckpt/best.ckpt --bbox 55,72,15,45 --leads 1 --tag ghap_smoke --out-root $T/rt"
rc2=$?
T2=$(date +%s)
rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"
printf '%s\n' "- $(date '+%F %H:%M') [ghap] job $SLURM_JOB_ID plumbing test (smoke ckpt, no truth): predict_2022 16 days (1 full) rc=$rc1 ($(( (T1 - T0) / 60 )) min), realtime_1km/infer GHAP switch bbox 55-72N rc=$rc2 ($(( (T2 - T1) / 60 )) min) -> runs_ghap/plumb" >> $S
exit $(( rc1 + rc2 ))
