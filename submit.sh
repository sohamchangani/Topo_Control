#!/bin/bash
#SBATCH --job-name=ks_topo
#SBATCH --partition=cpu          # set to a CPU partition of your cluster (the code uses no GPU)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10               # 5 objectives in parallel x 2 threads each
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --requeue                        # preempted jobs are requeued; finished runs are reused
#SBATCH --open-mode=append
##SBATCH --mail-type=END,FAIL            # to get emails, remove one '#' on these two lines
##SBATCH --mail-user=you@example.com
#
# Kuramoto-Sivashinsky experiments of the paper (Table 2): every objective, hyperparameter search on
# seeds 0-2, best validation checkpoint, 10 training seeds, retest of the controllers that pass screening.
#
#   sbatch submit.sh                     # from the repository root
#
# If the job reaches its time limit, submit it again: finished (configuration, seed) runs are skipped.
# Results: runs_ks/summary.md and runs_ks/hpsearch/<objective>_bestval/hpsearch.md.
# Per-objective logs: logs/ks_<objective>.log.  Override the output folder with: OUTDIR=... sbatch submit.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

# Python environment: edit for your cluster.
# module load python/3.11
# source ~/venvs/topo_attractor/bin/activate

OUTDIR=${OUTDIR:-runs_ks}
EXTRA_ARGS=${EXTRA_ARGS:-}               # e.g. EXTRA_ARGS=--smoke for a quick test of this script
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p logs

losses=(topology energy recurrence topology_energy topology_recurrence)

# Reference values and all initial-condition pools, computed once so the parallel runs only read them.
python main_ks.py --mode setup --outdir "$OUTDIR" $EXTRA_ARGS

pids=()
for loss in "${losses[@]}"; do
  echo "starting $loss (log: logs/ks_${loss}.log)"
  python main_ks.py \
    --mode pipeline \
    --loss "$loss" \
    --hpsearch \
    --hp_seeds 0,1,2 \
    --use_best_val \
    --threads 2 \
    --outdir "$OUTDIR" \
    $EXTRA_ARGS \
    >> "logs/ks_${loss}.log" 2>&1 &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

python main_ks.py --mode summarize --outdir "$OUTDIR" $EXTRA_ARGS
exit $status
