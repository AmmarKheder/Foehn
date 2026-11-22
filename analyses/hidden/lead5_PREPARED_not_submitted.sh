#!/bin/bash
# PREPARED, NOT SUBMITTED (a GPU job outside the scope of analyses/hidden).
# What is missing: Foehn's members at the residents' pixels 5 days ahead (stored for leads 1 and 3 only), so the share of the
# hidden people warned 5 days ahead by the 1 km forecast cannot be given. Cost of the run that would give it: the run of
# leads 1 and 3 (job 22867011, 15 Oct 2026) took 8 nodes of standard-g for 42 min 35 s for 358 forecasts (about 469 s per
# forecast and GPU); lead 5 alone is 179 forecasts: about 25 min on 8 nodes, 3 to 4 node-hours. Output: about 16 GB.
#
# 1. the forecasts (same checkpoint, sampler and members as eval_pop_crps_e01; arguments copied from the log of job 22867011):
#    cd /scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km
#    sbatch --nodes 8 --time 01:00:00 --job-name pop5_crps_e01 eval_green_job.sh \
#        runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4 \
#        --ckpt runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4/ckpts/epoch=01.ckpt \
#        --psub 16 --steps 4 --refine-tail 1 --members 4 --pop --every 2 --leads 5 --out eval_pop5_crps_e01
#    (check first with a short run: add --doys 0 64 and --out eval_pop5test_crps_e01)
#
# 2. the tables of people (CPU, partition small, about 25 min on 32 cores): in job_sys.sh add EVP_EXTRA=eval_pop5_crps_e01 before
#    the python call ->  data/hidden_systems_pop_leads.npz  (leads 1, 3 and 5; the file of leads 1 and 3 is not overwritten;
#    figdata_hidden_systems.py with EVP_EXTRA unset was checked to give the file of job 23864988 exactly: logs/hid_test2_*.out)
#
# 3. the numbers:  HID_FILE=hidden_systems_pop_leads.npz python tables.py systems   (tables.py reads the leads from the file)
#    then add the bars of lead 5 in fig_hidden_warned.py (loops 'for j, h in enumerate((1, 3))') and in make_si.py ('for h in (1, 3)').
echo "prepared only: read the comments of this file"
