if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi
# Loop over each line in the input file
while IFS= read -r line
do
  # Remove 'datas/' from the start of the line
  line=${line#datas/}
  
  # Run your python script with the current line as the inputFile
  cmd="python run.py --inputFile \"/mounted_path/$line\" --gpuNumber 0"
  eval $cmd
done < "$1"
