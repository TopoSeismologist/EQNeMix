#!/bin/bash

# Overwrite countbase.txt to start at zero
echo "256" > countbase.txt

# Python script path
python_script="EQNeMix_3D_Multiple_V1.py"

# Record start time
start_time=$(date +%s)

# Loop to execute the Python script repeatedly
while true; do
    # Execute the Python script
    python3 "$python_script"

    # Check if the Python script has finished
    if grep -q "EVENTS COMPLETED" countbase.txt; then
        echo "Gracias por confiar en EQNeMix :)."
        break
    fi
done

# Calculate total elapsed time
end_time=$(date +%s)
total_time=$((end_time - start_time))

# Print total elapsed time
echo "Tiempo total transcurrido: $total_time segundos"

