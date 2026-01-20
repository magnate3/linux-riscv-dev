import re
import sys
from collections import defaultdict

# --- Configuration ---
LOG_FILE_PATH = sys.argv[1] # <--- CHANGE THIS
TARGET_OPERATIONS = {"AllReduce", "Broadcast", "AllGather"}
# These are interpreted as limits on the ELEMENT COUNT
BUCKET_LIMITS_COUNT = [32, 256, 2048, 16384, 65536, 262144, 8388608]

# --- End Configuration ---

def get_bucket_index(element_count, limits):
    """Finds the correct bucket index for a given element count."""
    for i, limit in enumerate(limits):
        if element_count < limit:
            return i
    # If count is greater than or equal to the last limit, it goes in the last bucket
    return len(limits)

def create_bucket_labels(limits):
    """Creates human-readable labels for the buckets based on count."""
    labels = []
    lower_bound = 0
    for limit in limits:
        labels.append(f"{lower_bound} <= count < {limit}")
        lower_bound = limit
    labels.append(f"count >= {lower_bound}")
    return labels

def parse_target_nccl_ops(file_path, target_ops, bucket_limits):
    """
    Parses the NCCL log file, focusing *only* on successfully extracting
    count from lines containing the target operations.
    Silently ignores all other lines or lines where extraction fails.
    """
    # Initialize results structure: results[op_type][bucket_index] = count
    results = defaultdict(lambda: [0] * (len(bucket_limits) + 1))
    lines_processed = 0
    lines_matched_and_counted = 0 # Count lines we successfully processed

    # Pre-compile a simpler regex just to find 'count <number>'
    # We'll check for the target op using string methods first.
    count_pattern = re.compile(r"\scount\s+(\d+)")

    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                lines_processed += 1
                line = line.strip()
                if not line or "NCCL INFO" not in line:
                    continue # Silently skip blank/irrelevant lines

                found_target_op = None
                # Check if the line contains one of the target operations we care about
                for op in target_ops:
                    # Look for "NCCL INFO OpType:" specifically
                    if f"[1] NCCL INFO {op}:" in line:
                        found_target_op = op
                        break

                if not found_target_op:
                    continue # Silently skip NCCL INFO lines for other ops

                # --- Line contains a target operation ---
                # Now, try to extract the count value using regex
                match = count_pattern.search(line)
                if match:
                    try:
                        count = int(match.group(1))

                        # Find the correct bucket based *directly* on the count
                        bucket_idx = get_bucket_index(count, bucket_limits)

                        # Increment the counter
                        results[found_target_op][bucket_idx] += 1
                        lines_matched_and_counted += 1

                    except ValueError:
                        # Silently skip if count value wasn't a valid integer
                        # (This shouldn't happen often with the regex, but good practice)
                        pass
                # else:
                    # Silently skip if 'count \d+' pattern wasn't found in the target line

    except FileNotFoundError:
        print(f"Error: Log file not found at '{file_path}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Processed {lines_processed} lines from '{file_path}'.")
    print(f"Successfully extracted and bucketed data from {lines_matched_and_counted} target operation lines.")

    return results

# --- Main Execution ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_nccl.py <input_file>")
        sys.exit(1)

    # Use the count-based limits
    bucketed_counts = parse_target_nccl_ops(
        LOG_FILE_PATH,
        TARGET_OPERATIONS,
        BUCKET_LIMITS_COUNT
    )

    # Create labels based on count limits
    bucket_labels = create_bucket_labels(BUCKET_LIMITS_COUNT)

    print("\n--- NCCL Operation Counts per Element Count Bucket ---")
    print(f"(Only includes counts from successfully parsed {', '.join(TARGET_OPERATIONS)} lines)")


    if not bucketed_counts:
        print("\nNo matching operations found or processed successfully.")
    else:
        # Sort operations for consistent output
        # Ensure all target ops are potentially in the output, even if count is 0
        all_ops_keys = sorted(list(TARGET_OPERATIONS))
        # Make sure defaultdict creates entries for ops found, even if no successful parse
        for op in bucketed_counts.keys():
            if op not in all_ops_keys: # Should not happen with defaultdict, but safe
                 all_ops_keys.append(op)
                 all_ops_keys.sort()


        # Print header - reflect that bucketing is by count
        header = "Element Count Range".ljust(25) + "".join([op.center(12) for op in all_ops_keys])
        print(header)
        print("-" * len(header))

        # Print counts for each bucket
        for i, label in enumerate(bucket_labels):
            row = label.ljust(25)
            for op in all_ops_keys:
                # Use .get() on the inner list in case an op had 0 entries
                count_val = bucketed_counts.get(op, [0] * (len(BUCKET_LIMITS_COUNT) + 1))[i]
                row += str(count_val).center(12)
            print(row)

        # Print totals per operation
        print("-" * len(header))
        total_row = "Total".ljust(25)
        for op in all_ops_keys:
             # Use .get() for safety
             total_count = sum(bucketed_counts.get(op, []))
             total_row += str(total_count).center(12)
        print(total_row)
