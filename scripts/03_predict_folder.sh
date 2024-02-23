#!/bin/bash
export TORCH_HOME=.cache

python --version

export nnUNet_preprocessed=$HOME/.totalsegmentator/nnunet/results
export nnUNet_results=$HOME/.totalsegmentator/nnunet/results
export nnUNet_raw=$HOME/.totalsegmentator/nnunet/results


# Initialize variables for options
configPath=""
directoryPath=""
numberOfWorkers=0

# Usage message
usage() {
    echo "Usage: $0 -c <configPath> -d <directoryPath> -n <numberOfWorkers>"
    exit 1
}

# Parse options
while getopts "c:d:n:" opt; do
    case ${opt} in
        c )
            configPath=$OPTARG
            ;;
        d )
            directoryPath=$OPTARG
            ;;
        n )
            numberOfWorkers=$OPTARG
            ;;
        \? )
            usage
            ;;
    esac
done

# Check if options are provided
if [ -z "$configPath" ] || [ -z "$directoryPath" ] || [ -z "$numberOfWorkers" ]; then
    echo "All options -c, -d, and -n are required."
    usage
fi

# Specify the number of chunks
n_chunks=$numberOfWorkers

# Read the file paths into an array
readarray -d '' files < <(find "$directoryPath" -type f -print0)

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
    # Call the Python script with the chunk of file paths and run it in the background, supress stdout and err with > /dev/null 2>&1 #>/dev/null 2>&1 &
    python3 scripts/03_predict_folder.py "$configPath" "${args[@]}" &
}

# Timer start
start_time=$(date +%s)

# Chunk the array and process each chunk with the Python script asynchronously
for ((chunk=0; chunk<n_chunks; chunk++)); do
    start=$((chunk * chunk_size))
    end=$(((chunk + 1) * chunk_size))
    echo "Launching Python process $((chunk + 1)) ..."
    process_chunk_with_python_async $start $end
done

# Optionally, wait for all background processes to complete
wait

# Timer end
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "All Python scripts have completed in $duration seconds."

