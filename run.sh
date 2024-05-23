#!/bin/bash

# Set variables
VENV_PATH="$HOME/lambda-stack-with-tensorflow-pytorch" # access the virtual env for gpu support
PROJECT_PATH="$HOME/my_programs/bakery_sales_forecasting"              # navigate to project directory
MODULE_PATH="experiment.run_training"                                  # The Python module to run

# Optional: specify args to pass to run_training wrapper
ARGS="--batch_size 32 --learning_rate 0.001 --num_epochs 50 --config_file experiment/dataset_analysis.yaml --data_directory data/sales_forecasting/sales_forecasting_1w --heartbeat_file /tmp/heartbeat --heartbeat_interval 120"
HEARTBEAT_FILE="/tmp/heartbeat"
HEARTBEAT_INTERVAL=120  # Interval in seconds for checking heartbeat
BUFFER_TIME=10  # Buffer time in seconds

# Function to handle cleanup on script exit
cleanup() {
    if [ $PID -ne 0 ]; then
        kill $PID 2>/dev/null
    fi
    echo -e "\033[1;31mExiting...\033[0m"
    exit 0
}

# Trap SIGINT and SIGTERM signals to allow manual stop
trap 'cleanup' SIGINT SIGTERM

# Function to run the training script
run_training() {
    source "$VENV_PATH/bin/activate"
    cd "$PROJECT_PATH"
    python -m "$MODULE_PATH" $ARGS &
    PID=$!
}

# Function to check heartbeat
check_heartbeat() {
    while true; do
        sleep "$HEARTBEAT_INTERVAL"
        if [ ! -f "$HEARTBEAT_FILE" ]; then
            echo -e "\033[1;31mHeartbeat file not found. Restarting training...\033[0m"
            kill $PID
            return 1
        fi
        CURRENT_TIME=$(date +%s)
        FILE_MOD_TIME=$(stat -c %Y "$HEARTBEAT_FILE")
        if grep -q "finished" "$HEARTBEAT_FILE"; then
            echo -e "\033[1;32mTraining completed successfully. Exiting...\033[0m"
            kill $PID
            return 1
        elif grep -q "stopped" "$HEARTBEAT_FILE"; then
            echo -e "\033[1;31mTraining manually stopped. Exiting...\033[0m"
            kill $PID
            return 1
        elif [ $(($CURRENT_TIME - $FILE_MOD_TIME)) -ge $(($HEARTBEAT_INTERVAL + $BUFFER_TIME)) ]; then
            echo -e "\033[1;31mNo heartbeat signal received. Restarting training...\033[0m"
            kill $PID
            return 0
        fi
    done
}

while true; do
    run_training
    if check_heartbeat; then
        break
    else
        echo "Restarting training after crash in 60 seconds..."
        sleep 60
    fi
done

cleanup