#!/bin/bash
################################################################################
# F03: Downscaling Model Inference and Evaluation Pipeline Wrapper
#
# This wrapper script submits separate SLURM jobs for inference and evaluation.
# The evaluation job is submitted with a dependency on the inference job,
# so it only runs after inference completes successfully.
#
# This allows:
#   - Different resource allocations for inference vs evaluation
#   - Independent control over each step
#   - Better queue management and resource utilization
#
# Usage:
#   bash Sh/F03_inference_evaluate.sh           # Run both inference and evaluation
#   bash Sh/F03_inference_evaluate.sh --inference-only    # Run inference only
#   bash Sh/F03_inference_evaluate.sh --evaluate-only     # Run evaluation only
#
# The individual SLURM scripts can also be submitted directly:
#   sbatch Sh/F03a_inference_slurm.sh
#   sbatch Sh/F03b_evaluate_slurm.sh
################################################################################

# Set and export root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# NOTE: We do NOT source env_setting.sh here because it runs 'module purge'
# which can break SLURM. The individual SLURM job scripts will source it.

################################################################################
# PARSE COMMAND LINE ARGUMENTS
################################################################################

RUN_INFERENCE="true"
RUN_EVALUATION="true"

while [[ $# -gt 0 ]]; do
    case $1 in
        --inference-only)
            RUN_INFERENCE="true"
            RUN_EVALUATION="false"
            shift
            ;;
        --evaluate-only)
            RUN_INFERENCE="false"
            RUN_EVALUATION="true"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --inference-only   Run only the inference step"
            echo "  --evaluate-only    Run only the evaluation step"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Without options, runs both inference and evaluation with dependency."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

################################################################################
# HELPER FUNCTIONS
################################################################################

# Function to submit a job and get its job ID
submit_job() {
    local script=$1
    local dependency=$2
    
    if [ -z "$dependency" ]; then
        # Submit job without dependency
        job_id=$(sbatch --parsable --chdir="${ROOT_DIR}/Sh" "$script")
    else
        # Submit job with dependency (only run after previous job succeeds)
        job_id=$(sbatch --parsable --dependency=afterok:$dependency --chdir="${ROOT_DIR}/Sh" "$script")
    fi
    
    echo $job_id
}

################################################################################
# MAIN EXECUTION
################################################################################

echo "================================================================================"
echo "F03: Downscaling Model Inference and Evaluation Pipeline"
echo "================================================================================"
echo "Root directory:      ${ROOT_DIR}"
echo "Run inference:       ${RUN_INFERENCE}"
echo "Run evaluation:      ${RUN_EVALUATION}"
echo "================================================================================"
echo ""

# Track submitted job IDs
INFERENCE_JOB_ID=""
EVALUATE_JOB_ID=""

# Submit inference job (if enabled)
if [ "${RUN_INFERENCE}" = "true" ]; then
    echo "Submitting inference job..."
    INFERENCE_JOB_ID=$(submit_job "${ROOT_DIR}/Sh/inference_slurm.sh" "")
    
    if [ -z "$INFERENCE_JOB_ID" ]; then
        echo "ERROR: Failed to submit inference job"
        exit 1
    fi
    
    echo "  ✓ Inference job submitted with ID: ${INFERENCE_JOB_ID}"
    echo ""
fi

# Submit evaluation job (if enabled)
if [ "${RUN_EVALUATION}" = "true" ]; then
    echo "Submitting evaluation job..."
    
    if [ "${RUN_INFERENCE}" = "true" ]; then
        # Evaluation depends on inference completing successfully
        EVALUATE_JOB_ID=$(submit_job "${ROOT_DIR}/Sh/evaluate_slurm.sh" "${INFERENCE_JOB_ID}")
        echo "  ✓ Evaluation job submitted with ID: ${EVALUATE_JOB_ID}"
        echo "    (will run after inference job ${INFERENCE_JOB_ID} completes)"
    else
        # No dependency - run immediately
        EVALUATE_JOB_ID=$(submit_job "${ROOT_DIR}/Sh/evaluate_slurm.sh" "")
        echo "  ✓ Evaluation job submitted with ID: ${EVALUATE_JOB_ID}"
    fi
    echo ""
fi

################################################################################
# SUMMARY
################################################################################

echo "================================================================================"
echo "Pipeline Submitted Successfully!"
echo "================================================================================"
echo ""

if [ -n "$INFERENCE_JOB_ID" ]; then
    echo "Inference Job:"
    echo "  Job ID:        ${INFERENCE_JOB_ID}"
    echo "  Script:        Sh/inference_slurm.sh"
    echo "  Log output:    Log/inference.out"
    echo "  Log errors:    Log/inference.err"
    echo ""
fi

if [ -n "$EVALUATE_JOB_ID" ]; then
    echo "Evaluation Job:"
    echo "  Job ID:        ${EVALUATE_JOB_ID}"
    echo "  Script:        Sh/evaluate_slurm.sh"
    echo "  Log output:    Log/evaluate.out"
    echo "  Log errors:    Log/evaluate.err"
    if [ -n "$INFERENCE_JOB_ID" ]; then
        echo "  Dependency:    afterok:${INFERENCE_JOB_ID}"
    fi
    echo ""
fi

echo "Job Dependencies:"
if [ "${RUN_INFERENCE}" = "true" ] && [ "${RUN_EVALUATION}" = "true" ]; then
    echo "  Inference (${INFERENCE_JOB_ID}) → Evaluation (${EVALUATE_JOB_ID})"
elif [ "${RUN_INFERENCE}" = "true" ]; then
    echo "  Inference only (${INFERENCE_JOB_ID})"
elif [ "${RUN_EVALUATION}" = "true" ]; then
    echo "  Evaluation only (${EVALUATE_JOB_ID})"
fi
echo ""

echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs with:"
if [ -n "$INFERENCE_JOB_ID" ]; then
    echo "  tail -f Log/inference.out"
fi
if [ -n "$EVALUATE_JOB_ID" ]; then
    echo "  tail -f Log/evaluate.out"
fi
echo ""
echo "================================================================================"

exit 0

