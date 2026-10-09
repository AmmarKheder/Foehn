# Run a command inside the LUMI PyTorch container with the project venv:  torchrun_env <cmd...>
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
torch_exec() {
  singularity exec -B /projappl,/scratch,/pfs,/appl "$SIF" bash -c "unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch; \
source /projappl/project_462001140/envs/torchenv/bin/activate; export PYTHONPATH=/scratch/project_462001140/ammar/eccv/cran-pm/src; $*"
}
