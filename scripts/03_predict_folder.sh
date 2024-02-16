#!/bin/bash
export TORCH_HOME=.cache

#pip install -r requirements.txt
#pip install -e .
python --version

export nnUNet_preprocessed=$HOME/.totalsegmentator/nnunet/results
export nnUNet_results=$HOME/.totalsegmentator/nnunet/results
export nnUNet_raw=$HOME/.totalsegmentator/nnunet/results


# Define the directory to list files from
directory="/data/core-rad/data/tucker/raw/000-tdata/imagesTs"

# Specify the number of chunks
n_chunks=$1

# Read the file paths into an array
readarray -d '' files < <(find "$directory" -type f -print0)

# Calculate the total number of files
total_files=${#files[@]}

# Calculate the chunk size (integer division; some chunks may have one more file)
chunk_size=$(( (total_files + n_chunks - 1) / n_chunks ))

# Function to process chunks with the Python script asynchronously
process_chunk_with_python_async() {
    local start=$1
    local end=$2
    local args=()
    for ((i=start; i<end && i<total_files; i++)); do
        args+=("${files[i]}")
    done
    # Call the Python script with the chunk of file paths and run it in the background
    python3 scripts/03_predict_folder.py "${args[@]}" &
}

# Chunk the array and process each chunk with the Python script asynchronously
for ((chunk=0; chunk<n_chunks; chunk++)); do
    start=$((chunk * chunk_size))
    end=$(((chunk + 1) * chunk_size))
    echo "Launching Python process $((chunk + 1)) ..."
    process_chunk_with_python_async $start $end
done

# Optionally, wait for all background processes to complete
wait
echo "All Python scripts have completed."

