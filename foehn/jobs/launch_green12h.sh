#!/bin/bash
# FLEXGREEN in Aurora's protocol for 'Foehn trained as Aurora' (foehn_cams.py): 12-hour leads 12..120 h released at the
# valid time, every 0.5 deg cell, 32 particles, emission term, fires of day d-1 -> store green12h. Initial days from
# 2 Oct 2017 (t0 - 12 h must be in the CAMS series) to 30 Nov 2022. One job per year; days already done are skipped.
#     ./launch_green12h.sh <nodes per job> <time limit>      e.g. ./launch_green12h.sh 2 12:00:00
set -euo pipefail
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km
N=$1; T=$2
X="--particles 32 --store green12h --emis --instant --all-cells"
for spec in "2017 274 364" "2018 0 364" "2019 0 364" "2020 0 365" "2021 0 364" "2022 0 333"; do
  set -- $spec
  sbatch --parsable --nodes $N --time $T --job-name fg12h_$1 flexgreen_job.sh hres $1 $2 $3 "$X"
done
