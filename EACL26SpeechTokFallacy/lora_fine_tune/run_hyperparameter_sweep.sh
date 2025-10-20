#!/bin/bash

# Hyperparameter sweep script for LoRA fine-tuning
# This script runs multiple combinations of seeds, learning rates, and LoRA rank values

# Configuration
SCRIPT_NAME="lora_fine_tune.py"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="hyperparameter_sweep_results/${TIMESTAMP}"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Define hyperparameter arrays
LEARNING_RATES=(1e-4 5e-4 1e-3 2e-3)
R_LORA_VALUES=(32 64)
SEEDS=(42 100 6 10 420)

# Calculate total combinations
TOTAL_LR=${#LEARNING_RATES[@]}
TOTAL_R=${#R_LORA_VALUES[@]}
TOTAL_SEEDS=${#SEEDS[@]}
TOTAL_COMBINATIONS=$((TOTAL_LR * TOTAL_R * TOTAL_SEEDS))

echo "======================================="
echo "HYPERPARAMETER SWEEP CONFIGURATION"
echo "======================================="
echo "Learning rates: ${LEARNING_RATES[*]}"
echo "LoRA ranks: ${R_LORA_VALUES[*]}"
echo "Seeds: ${SEEDS[*]}"
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
SUMMARY_CSV="$RESULTS_DIR/sweep_summary.csv"
echo "learning_rate,r_lora,seed,success,runtime,output_dir,error_message" > "$SUMMARY_CSV"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

log_message "Starting hyperparameter sweep with $TOTAL_COMBINATIONS experiments"

# Run experiments
for lr in "${LEARNING_RATES[@]}"; do
    for r_val in "${R_LORA_VALUES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            CURRENT_EXP=$((CURRENT_EXP + 1))
            
            echo ""
            echo "******************** Experiment $CURRENT_EXP/$TOTAL_COMBINATIONS ********************"
            log_message "Running experiment $CURRENT_EXP/$TOTAL_COMBINATIONS"
            log_message "  Learning Rate: $lr"
            log_message "  LoRA Rank (r): $r_val"
            log_message "  Seed: $seed"
            
            # Record start time
            START_TIME=$(date +%s)
            
            # Define output directory
            OUTPUT_DIR="llama_lora_text_final_${r_val}_${lr}_${seed}"
            
            # Create temporary file for capturing errors
            TEMP_ERROR_FILE="$RESULTS_DIR/temp_error_${CURRENT_EXP}.txt"
            
            # Run the experiment and capture both stdout and stderr
            if CUDA_VISIBLE_DEVICES=0 python3 "$SCRIPT_NAME" --learning_rate "$lr" --r_lora "$r_val" --seed "$seed" > "$RESULTS_DIR/experiment_${CURRENT_EXP}_output.txt" 2> "$TEMP_ERROR_FILE"; then
                # Success
                END_TIME=$(date +%s)
                RUNTIME=$((END_TIME - START_TIME))
                SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
                
                log_message "[SUCCESS] Experiment $CURRENT_EXP completed successfully in ${RUNTIME}s"
                echo "$lr,$r_val,$seed,True,$RUNTIME,$OUTPUT_DIR," >> "$SUMMARY_CSV"
                
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
                    echo "Learning Rate: $lr, LoRA Rank: $r_val, Seed: $seed" >> "$ERROR_LOG"
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
                echo "$lr,$r_val,$seed,False,$RUNTIME,\"$ERROR_MSG\"" >> "$SUMMARY_CSV"
            fi
            
            # Progress update
            REMAINING=$((TOTAL_COMBINATIONS - CURRENT_EXP))
            log_message "Progress: $CURRENT_EXP/$TOTAL_COMBINATIONS completed, $REMAINING remaining"
        done
    done
done

# Final summary
echo ""
echo "======================================="
echo "HYPERPARAMETER SWEEP COMPLETED"
echo "======================================="
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
    while IFS=',' read -r lr r_val seed success runtime output_dir error_msg; do
        if [ "$success" = "True" ]; then
            echo "  LR: $lr, R: $r_val, Seed: $seed -> $output_dir"
        fi
    done < <(tail -n +2 "$SUMMARY_CSV")  # Skip header line
fi

# Show failed configurations with error summaries
if [ $FAILED_RUNS -gt 0 ]; then
    echo ""
    echo "Failed configurations:"
    while IFS=',' read -r lr r_val seed success runtime output_dir error_msg; do
        if [ "$success" = "False" ]; then
            echo "  LR: $lr, R: $r_val, Seed: $seed -> Error: $error_msg"
        fi
    done < <(tail -n +2 "$SUMMARY_CSV")  # Skip header line
    echo ""
    echo "Check $ERROR_LOG for detailed error information."
fi

log_message "Hyperparameter sweep completed. Total: $TOTAL_COMBINATIONS, Successful: $SUCCESSFUL_RUNS, Failed: $FAILED_RUNS"
