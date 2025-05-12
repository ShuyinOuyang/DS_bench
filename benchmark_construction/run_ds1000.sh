#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --job-name=sy_run_evaluation

MODEL_NAME="$1"
FILE_NAME="$2"

if [ -z "$MODEL_NAME" ] || [ -z "$FILE_NAME" ]; then
    echo "Usage: $0 <model_name> <file_name>"
    exit 1
fi

python run_ds1000_test.py -m "$MODEL_NAME" -f "$FILE_NAME"

#while true; do
#    echo "Running run_ds1000_test.py with model: $MODEL_NAME and file: $FILE_NAME..."
#    python run_test.py -m "$MODEL_NAME" -f "$FILE_NAME"
#
#    if [ $? -eq 0 ]; then
#        echo "run_ds1000_test.py finished successfully."
#        break
#    else
#        python segfault_handle.py -m "$MODEL_NAME" -f "$FILE_NAME"
#        echo "run_ds1000_test.py was killed or failed. Restarting..."
#    fi
#done

