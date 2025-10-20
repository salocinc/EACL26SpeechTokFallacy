#!/bin/bash

# Generic hyperparameter sweep script for any Python script that accepts --seed parameter
# This script runs multiple combinations of seeds for any specified Python script

# ====== CONFIGURATION SECTION ======
# Change this to the script you want to run, use .py extension
SCRIPT_NAME="baseline_roberta_wavlm.py"

# Define seeds to test and learning rates
SEEDS=(42 100 6 10 420)
LEARNING_RATES=(1e-4 5e-4 1e-3 2e-3)

# ====== END CONFIGURATION SECTION ======

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_BASENAME=$(basename "$SCRIPT_NAME" .py)
RESULTS_DIR="${SCRIPT_BASENAME}_seed_sweep_results/${TIMESTAMP}"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Calculate total combinations
TOTAL_SEEDS=${#SEEDS[@]}
TOTAL_LR=${#LEARNING_RATES[@]}

if [ ${TOTAL_LR} -gt 0 ]; then
    TOTAL_COMBINATIONS=$((TOTAL_LR * TOTAL_SEEDS))
    USE_LEARNING_RATE=true
else
    TOTAL_COMBINATIONS=$TOTAL_SEEDS
    USE_LEARNING_RATE=false
fi

echo "======================================="
echo "GENERIC SEED SWEEP CONFIGURATION"
echo "======================================="
echo "Script: $SCRIPT_NAME"
echo "Seeds: ${SEEDS[*]}"
if [ "$USE_LEARNING_RATE" = true ]; then
    echo "Learning rates: ${LEARNING_RATES[*]}"
fi
if [ -n "$EXTRA_PARAMS" ]; then
    echo "Extra parameters: $EXTRA_PARAMS"
fi
echo "Total combinations: $TOTAL_COMBINATIONS"
echo "Results directory: $RESULTS_DIR"
echo "======================================="

# Ask for confirmation
read -p "Do you want to run all $TOTAL_COMBINATIONS experiments? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Experiment cancelled."
    exit 1
fi

# Initialize counters
CURRENT_EXP=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0

# Initialize log files
MAIN_LOG="$RESULTS_DIR/sweep_log.txt"
ERROR_LOG="$RESULTS_DIR/sweep_errors.txt"

if [ "$USE_LEARNING_RATE" = true ]; then
    SUMMARY_CSV="$RESULTS_DIR/sweep_summary.csv"
    echo "learning_rate,seed,success,runtime,output_dir,error_message" > "$SUMMARY_CSV"
else
    SUMMARY_CSV="$RESULTS_DIR/sweep_summary.csv"
    echo "seed,success,runtime,output_dir,error_message" > "$SUMMARY_CSV"
fi

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

log_message "Starting seed sweep for $SCRIPT_NAME with $TOTAL_COMBINATIONS experiments"

# Function to run a single experiment
run_experiment() {
    local seed=$1
    local lr=$2
    
    CURRENT_EXP=$((CURRENT_EXP + 1))
    
    echo ""
    echo "******************** Experiment $CURRENT_EXP/$TOTAL_COMBINATIONS ********************"
    log_message "Running experiment $CURRENT_EXP/$TOTAL_COMBINATIONS"
    log_message "  Seed: $seed"
    if [ -n "$lr" ]; then
        log_message "  Learning Rate: $lr"
    fi
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Build command
    CMD="python3 \"$SCRIPT_NAME\" --seed $seed"
    if [ -n "$lr" ]; then
        CMD="$CMD --learning_rate $lr"
        OUTPUT_DIR="${SCRIPT_BASENAME}_lr_${lr}_seed_${seed}"
    else
        OUTPUT_DIR="${SCRIPT_BASENAME}_seed_${seed}"
    fi
    
    if [ -n "$EXTRA_PARAMS" ]; then
        CMD="$CMD $EXTRA_PARAMS"
    fi
    
    # Create temporary file for capturing errors
    TEMP_ERROR_FILE="$RESULTS_DIR/temp_error_${CURRENT_EXP}.txt"
    
    # Run the experiment and capture both stdout and stderr
    if eval "$CMD" > "$RESULTS_DIR/experiment_${CURRENT_EXP}_output.txt" 2> "$TEMP_ERROR_FILE"; then
        # Success
        END_TIME=$(date +%s)
        RUNTIME=$((END_TIME - START_TIME))
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        
        log_message "[SUCCESS] Experiment $CURRENT_EXP completed successfully in ${RUNTIME}s"
        
        if [ "$USE_LEARNING_RATE" = true ]; then
            echo "$lr,$seed,True,$RUNTIME,$OUTPUT_DIR," >> "$SUMMARY_CSV"
        else
            echo "$seed,True,$RUNTIME,$OUTPUT_DIR," >> "$SUMMARY_CSV"
        fi
        
        # Clean up temporary error file if successful
        rm -f "$TEMP_ERROR_FILE"
    else
        # Failure
        END_TIME=$(date +%s)
        RUNTIME=$((END_TIME - START_TIME))
        FAILED_RUNS=$((FAILED_RUNS + 1))
        
        # Capture error details
        ERROR_MSG=""
        if [ -s "$TEMP_ERROR_FILE" ]; then
            # Get last few lines of error for summary (escape quotes for CSV)
            ERROR_MSG=$(tail -3 "$TEMP_ERROR_FILE" | tr '\n' ' ' | sed 's/"/\""/g')
            
            # Log full error to error log
            echo "==================== Experiment $CURRENT_EXP Error ====================" >> "$ERROR_LOG"
            echo "Script: $SCRIPT_NAME" >> "$ERROR_LOG"
            echo "Seed: $seed" >> "$ERROR_LOG"
            if [ -n "$lr" ]; then
                echo "Learning Rate: $lr" >> "$ERROR_LOG"
            fi
            echo "Command: $CMD" >> "$ERROR_LOG"
            echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')" >> "$ERROR_LOG"
            echo "Full Error Output:" >> "$ERROR_LOG"
            cat "$TEMP_ERROR_FILE" >> "$ERROR_LOG"
            echo "" >> "$ERROR_LOG"
            
            # Clean up temporary error file
            rm -f "$TEMP_ERROR_FILE"
        else
            ERROR_MSG="Unknown error (no error output captured)"
        fi
        
        log_message "[FAILED] Experiment $CURRENT_EXP failed after ${RUNTIME}s"
        log_message "   Error: $ERROR_MSG"
        
        if [ "$USE_LEARNING_RATE" = true ]; then
            echo "$lr,$seed,False,$RUNTIME,\"$ERROR_MSG\"" >> "$SUMMARY_CSV"
        else
            echo "$seed,False,$RUNTIME,\"$ERROR_MSG\"" >> "$SUMMARY_CSV"
        fi
    fi
    
    # Progress update
    REMAINING=$((TOTAL_COMBINATIONS - CURRENT_EXP))
    log_message "Progress: $CURRENT_EXP/$TOTAL_COMBINATIONS completed, $REMAINING remaining"
}

# Run experiments
if [ "$USE_LEARNING_RATE" = true ]; then
    # Run with both learning rates and seeds
    for lr in "${LEARNING_RATES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_experiment "$seed" "$lr"
        done
    done
else
    # Run with seeds only
    for seed in "${SEEDS[@]}"; do
        run_experiment "$seed"
    done
fi

# Final summary
echo ""
echo "======================================="
echo "SEED SWEEP COMPLETED"
echo "======================================="
echo "Script: $SCRIPT_NAME"
echo "Total experiments: $TOTAL_COMBINATIONS"
echo "Successful: $SUCCESSFUL_RUNS"
echo "Failed: $FAILED_RUNS"

if [ $TOTAL_COMBINATIONS -gt 0 ]; then
    SUCCESS_RATE=$(( (SUCCESSFUL_RUNS * 100) / TOTAL_COMBINATIONS ))
    echo "Success rate: ${SUCCESS_RATE}%"
fi

echo "======================================="
echo "Results saved to: $RESULTS_DIR"
echo "Main log: $MAIN_LOG"
echo "Error log: $ERROR_LOG"
echo "Summary CSV: $SUMMARY_CSV"

# Show successful configurations
if [ $SUCCESSFUL_RUNS -gt 0 ]; then
    echo ""
    echo "Successful configurations:"
    if [ "$USE_LEARNING_RATE" = true ]; then
        while IFS=',' read -r lr seed success runtime output_dir error_msg; do
            if [ "$success" = "True" ]; then
                echo "  LR: $lr, Seed: $seed -> $output_dir"
            fi
        done < <(tail -n +2 "$SUMMARY_CSV")  # Skip header line
    else
        while IFS=',' read -r seed success runtime output_dir error_msg; do
            if [ "$success" = "True" ]; then
                echo "  Seed: $seed -> $output_dir"
            fi
        done < <(tail -n +2 "$SUMMARY_CSV")  # Skip header line
    fi
fi

# Show failed configurations with error summaries
if [ $FAILED_RUNS -gt 0 ]; then
    echo ""
    echo "Failed configurations:"
    if [ "$USE_LEARNING_RATE" = true ]; then
        while IFS=',' read -r lr seed success runtime output_dir error_msg; do
            if [ "$success" = "False" ]; then
                echo "  LR: $lr, Seed: $seed -> Error: $error_msg"
            fi
        done < <(tail -n +2 "$SUMMARY_CSV")  # Skip header line
    else
        while IFS=',' read -r seed success runtime output_dir error_msg; do
            if [ "$success" = "False" ]; then
                echo "  Seed: $seed -> Error: $error_msg"
            fi
        done < <(tail -n +2 "$SUMMARY_CSV")  # Skip header line
    fi
    echo ""
    echo "Check $ERROR_LOG for detailed error information."
fi

log_message "Seed sweep completed for $SCRIPT_NAME. Total: $TOTAL_COMBINATIONS, Successful: $SUCCESSFUL_RUNS, Failed: $FAILED_RUNS"