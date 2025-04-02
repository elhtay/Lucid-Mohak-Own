#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
import time 
import shutil 
from datetime import datetime

def get_files_by_size(directory, max_group_size, extension='.pcap'):
    """
    Group files such that each group's total size doesn't exceed max_group_size.
  
    Args:
        directory: Directory containing the files
        max_group_size: Maximum total size for each group in bytes
        extension: File extension to filter by
        
    Returns:
        List of groups, where each group is a list of file paths
    """
    # Get all matching files with their sizes
    all_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(extension):
                filepath = os.path.join(root, filename)
                filesize = os.path.getsize(filepath)
                all_files.append((filepath, filesize))
    
    # Sort files by size in descending order for better packing
    all_files.sort(key=lambda x: x[1], reverse=True)
    print(f"Script Runner: Found {len(all_files)} files in {directory} with extension {extension}")
    
    groups = []
    current_group = []
    current_size = 0
    
    for filepath, filesize in all_files:
        # If file itself is larger than max_group_size, put it in its own group
        if filesize > max_group_size:
            print(f"Script Runner: Warning: File {filepath} size ({filesize} bytes) exceeds max group size ({max_group_size} bytes)")
            groups.append([(filepath, filesize)])
            continue
            
        # If adding this file would exceed the limit, start a new group
        if current_size + filesize > max_group_size:
            if current_group:  # Only add non-empty groups
                groups.append(current_group)
            current_group = [(filepath, filesize)] # Start a new group
            current_size = filesize
        else:
            # Add to current group
            current_group.append((filepath, filesize))
 
            print(f"Script Runner: Current group contents: {current_group}")
            # print tne value of current_size
            current_size += filesize
    
    # Add the last group if it's not empty
    if current_group:
        groups.append(current_group)
    
    return groups

def main():
    # These constants would hardcode values, making the script less flexible
    dataset_folder = "./sample-dataset/"
    max_group_size = 1000000000/2  # 0.5 GB
    # max_group_size = 1000000000  # 1 GB
    
    # Using argparse instead provides several advantages:
    parser = argparse.ArgumentParser(description='Sort and process dataset files by size')
    parser.add_argument('--output_folder', help='Output folder for processed files')
    parser.add_argument('--dataset_type', default= "DOS2019", help='Type of the dataset (DOS2017, DOS2018, DOS2019, SYN2020)')
    parser.add_argument('--packets_per_flow', default="10", help='Number of packets per flow')
    parser.add_argument('--dataset_id', default= "DOS2019",  help='ID for the dataset')
    parser.add_argument('--traffic_type', default='all', help='Type of traffic to process (all, benign, ddos)')
    parser.add_argument('--dataset_folder', default=dataset_folder, help='Folder containing the dataset files')
    parser.add_argument('--max_group_size', default=max_group_size, type=int, help='Maximum total size (in bytes) for each group of files')
    parser.add_argument('--keep_temp_folders', action='store_true', help='Keep temporary group folders instead of cleaning them up')
    args = parser.parse_args()
    
    # Get files grouped by maximum size
    file_groups = get_files_by_size(args.dataset_folder, args.max_group_size)
    total_processed = 0
    
    print(f"Script Runner: Created {len(file_groups)} groups with maximum size of {args.max_group_size} bytes each")
    
    # Process each group
    for group_index, group in enumerate(file_groups, 1):
        
        # Extract just the filepaths from the (filepath, filesize) tuples
        files_to_process = [f[0] for f in group]
        total_group_size = sum(f[1] for f in group)
        
        if not files_to_process:
            continue
        
        print(f"Script Runner: \n{datetime.now()} - Processing group {group_index}/{len(file_groups)} with {len(files_to_process)} files (total {total_group_size} bytes)...")
        
        # Create a temporary directory for this group
        group_dir = os.path.join(args.dataset_folder, f"group_{group_index}")
        os.makedirs(group_dir, exist_ok=True)
        
        # Copy actual files to the temporary directory instead of creating symbolic links
        for file in files_to_process:
            dest_file = os.path.join(group_dir, os.path.basename(file))
            if not os.path.exists(dest_file):
                print(f"Script Runner: Copying file {file} to {dest_file}")
                shutil.copy2(file, dest_file)  # copy2 preserves file metadata
        
        # Build command for the parser
        cmd = [
            "python3", "lucid_dataset_parser.py",
            "--dataset_type", args.dataset_type,
            "--dataset_folder", group_dir, 
            "--packets_per_flow", str(args.packets_per_flow),
            "--dataset_id", f"{args.dataset_id}",
            "--traffic_type", args.traffic_type,
            "--time_window", "10" 
        ]
        
        # Add output folder if specified
        print(f"Command to be executed: {cmd}")
        if args.output_folder:
            cmd.extend(["--output_folder", args.output_folder])
        
        # Run the parser and wait for completion before proceeding
        try:
            print(f"Script Runner: Starting parser for group {group_index}...")
            print(f"Script Runner: Running command: {' '.join(cmd)}")
            
            # Use Popen to stream output in real-time without a timeout
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Print output in real-time
            complete_output = []
            while True:
                output_line = process.stdout.readline()
                if output_line == '' and process.poll() is not None:
                    break
                if output_line:
                    print(output_line.strip())
                    complete_output.append(output_line.strip())
            
            # Get return code and any stderr output
            return_code = process.poll()
            _, stderr_output = process.communicate()
            
            if return_code != 0:
                print(f"Script Runner: Parser command failed with return code: {return_code}")
                if stderr_output:
                    print(f"Script Runner: Error output: {stderr_output}")
                raise subprocess.CalledProcessError(return_code, cmd)
            
            print(f"Script Runner: Parser command for group {group_index} completed successfully")
            total_processed += len(files_to_process)
            
        except KeyboardInterrupt:
            print("\nScript Runner: Processing interrupted by user. Cleaning up...")
            process.terminate()
            process.wait()
            raise
        except Exception as e:
            print(f"Script Runner: Error processing group {group_index}: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"Script Runner: Error output: {e.stderr}")
        
        # Confirm completion of this group's processing
        print(f"Script Runner: Group {group_index}/{len(file_groups)} fully processed. Moving to next group...\n")
    
    
    if args.keep_temp_folders:
        print(f"Script Runner: Temporary group folders have been preserved. You can find them at: {args.dataset_folder}/group_*")
        print(f"Script Runner: To keep these folders, run this script with the --keep_temp_folders flag")
    
    print(f"Script Runner: \nAll {total_processed} files have been processed in {len(file_groups)} groups.")

if __name__ == "__main__":
    main()