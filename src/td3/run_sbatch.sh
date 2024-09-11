#!/bin/bash

# ANSI color codes
RED='\033[0;31m' # Red
G='\033[0;32m' # Green
NC='\033[0m' # Reset color

# Display script usage
function show_usage {
  echo "Runs sbatch on all scripts present in the directory"
  echo "Usage: $0 [-f] [-h]"
  echo "  -f: which file to sbatch. Can be either a directory or a file.
  If a directory, all files in the directory are sbatched. If a file, only that file is sbatched."
  echo "  -h: Display this help message"
  exit 1
}

DEFAULT_DIR="./configs"
# Parse command-line options
while getopts ":f:h" opt; do
  case $opt in
    f)
      DEFAULT_DIR="$OPTARG"
      echo -e "${G}Default directory changed to $DEFAULT_DIR${NC}"
      ;;
    h)
      show_usage
      ;;
    \?)
      echo -e "${RED}Error: Invalid option: -${OPTARG}${NC}"
      show_usage
      ;;
  esac
done

# Get args for the script
# Check if vars.txt exists
MORE_ARGS=0
VAR_FILE="./configs/vars.txt"
START_SEED=0
END_SEED=10


function process_vars_file() {
  local main_script_name="$1"
  echo -e "Processing vars file for $main_script_name"

  if [ -f "$VAR_FILE" ]; then
    while IFS= read -r line; do
      local script_name

      script_name=$(echo "$line" | cut -d' ' -f1)

      if [ "$script_name" = "$main_script_name" ]; then
        START_SEED=$(echo "$line" | cut -d' ' -f2)
        END_SEED=$(echo "$line" | cut -d' ' -f3)

        echo "First seed: $START_SEED"
        echo "Second seed: $END_SEED"
        MORE_ARGS=1
      fi
    done < "$VAR_FILE"
  else
    echo "Default run, no additional config"
    MORE_ARGS=0
  fi
}


# Check if the directory exists and is dir
if [ -d "$DEFAULT_DIR" ]; then
  echo -e "${G}Directory $DEFAULT_DIR exists${NC}. Running sbatch on all scripts."
  for FILE in "$DEFAULT_DIR"/*.sh; do
        # Check if the file is a regular file
        if [ -f "$FILE" ]; then
            BASENAME=$(basename "$FILE")
            process_vars_file "$BASENAME"
            if [ "$MORE_ARGS" -eq 1 ]; then
                echo -e "Running sbatch with $FILE and seeds $START_SEED and $END_SEED"
                sbatch "$FILE" -s "$START_SEED" -e "$END_SEED"
                MORE_ARGS=0
            else
                echo "Running sbatch with $FILE"
                sbatch "$FILE"
            fi
        fi
    done


# check if file exists
elif [ -f "$DEFAULT_DIR" ]; then
  echo -e "${G}File $DEFAULT_DIR exists${NC}. Running sbatch on the file."
  BASENAME=$(basename "$DEFAULT_DIR")
  process_vars_file "$BASENAME"

  if [ "$MORE_ARGS" -eq 1 ]; then
      echo -e "Running sbatch with $DEFAULT_DIR and seeds $START_SEED and $END_SEED"
      sbatch "$DEFAULT_DIR" -s "$START_SEED" -e "$END_SEED"
      MORE_ARGS=0
  else
      echo "Running sbatch with $DEFAULT_DIR"
      sbatch "$DEFAULT_DIR"
  fi

else
  echo -e "${RED}Error: $DEFAULT_DIR does not exist${NC}"
  show_usage
  exit 1
fi