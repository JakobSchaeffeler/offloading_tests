# -*- coding: utf-8 -*-
import subprocess
import os
import argparse
import pandas as pd
import re
import glob

global accumulate

def extract_kernel_names_nsys(filepath):
    kernel_names = []
    with open(filepath, 'r') as file:
        lines = file.readlines()

        # Locate the line containing "Time(%)"
        for i, line in enumerate(lines):
            if "Time(%)" in line or "Time (%)" in line:
                # Move two lines ahead from "Time(%)" line
                start_index = i + 2
                break
        else:
            # "Time(%)" not found
            return kernel_names

        # Collect kernel names until an empty line
        for line in lines[start_index:]:
            if line.strip() == "":
                break
            # Get the last element from the line after splitting
            kernel_name = line.strip().split()[-1]
            kernel_names.append(kernel_name)

    return kernel_names

def get_largest_substring(str1, str2):
    max_substring = ""  # Initialize the largest substring found
    
    # Iterate through all possible substrings of str1
    for i in range(len(str1)):
        for j in range(i + 1, len(str1) + 1):
            substring = str1[i:j]
            # Check if the substring exists in str2
            if substring in str2:
                # Update the max_substring if the found substring is larger
                if len(substring) > len(max_substring):
                    max_substring = substring

    return max_substring

def get_line_with_substring_linebreak(filename, substring):
    with open(filename, 'r') as file:
        # Read the first line
        line = next(file, None)  # Use next() with default to handle empty files

        while line is not None:
            # Check if the line contains the desired substring
            if substring in line:
                return line
            
            # Try to get the next line
            next_line = next(file, None)

            # Check for line breaks in the substring comparison
            largest_substring = get_largest_substring(substring, line)
            if len(largest_substring) > 0:
                rest = substring.replace(largest_substring, "")
                # Only check next_line if it exists
                if next_line is not None and rest in next_line:
                    return line
            
            # Move to the next line
            line = next_line

    # Return None if no matching line is found
    return None


def get_line_with_substring(filename, substring):
    with open(filename, 'r') as file:
        for line in file:
            # Check if the line contains the desired substring
            if substring in line:
                return line
    #print("Error in parsing profiling output: No line containing the substring " + substring + " was found.")
    return None

def get_every_line_with_substring(filename, substring):
    lines=[]
    with open(filename, 'r') as file:
        for line in file:
            # Check if the line contains the desired substring
            if substring in line:
                lines.append(line)
    #print("Error in parsing profiling output: No line containing the substring " + substring + " was found.")
    return lines

def contains_number(s):
    return bool(re.search(r'\d', s))


def profile_amd(executable, kernel_name, no_rerun):
    metrics = {}
    
    metrics["SALU Instructions per wave"] = "10.1.4" #should be summed up if multiple kernels exist
    metrics["VALU Instructions per wave"] = "10.1.0" #should be summed up if multiple kernels exist 
    metrics["IPC"] = "11.2.0"
    metrics["L1 Requests per wave"] = "16.3.0" #should be summed up if multiple kernels exist
    metrics["L1 Hit Rate"] = "16.3.5"
    metrics["L2 Requests per wave"] = "17.3.1" #should be summed up if multiple kernels exist
    metrics["L2 Hit Rate"] = "17.3.7"
    metrics["Wavefront Occupancy (% of peak)"] = "2.1.15" 
    metrics["Dependency Wait Cycles per wave"] = "7.2.4" #should be summed up if multiple kernels exist
    metrics["Issue Wait Cycles per wave"] = "7.2.5" #should be summed up if multiple kernels exist
    metrics["Grid Size"] = "7.1.0" 
    metrics["#Threads"] = "7.1.1"
    metrics["Shared Memory Allocated"] = "3.1.14" #should be summed up if multiple kernels exist
    metrics["Shared Memory Instructions per wave"] = "12.2.0" #should be summed up if multiple kernels exist
    metrics["Shared Memory Bank Conflicts (%)"] = "12.1.3"
    metrics["Register Spill Instructions per wave"] = "15.1.9" #should be summed up if multiple kernels exist
    metrics["Global Memory Instructions per wave"] = "10.3.0" #should be summed up if multiple kernels exist
    metrics["Global Memory Read Instructions per wave"] = "10.3.1" #should be summed up if multiple kernels exist
    metrics["Global Memory Write Instructions per wave"] = "10.3.2" #should be summed up if multiple kernels exist
    metrics["Coalesced Instructions (% of peak)"] = "16.1.3"
    metrics["Global Atomic Operations per wave"] = "15.1.8" #should be summed up if multiple kernels exist
    result_dict, overall_kernel_count = profile_omni(executable,kernel_name, metrics, no_rerun)
    
    #process obtained metrics
    
    # metrics with per wave and Shared Memory Allocated should be summed up accross all kernels => multiply average metric by number of kernel launches overall
    for key in result_dict.keys():
        if "per wave" in key or "Shared Memory Allocated" in key:
            result_dict[key]*= overall_kernel_count

    #normalize per wave matrics to be able to compare executions with different amount of waves
    waves = result_dict["Grid Size"]/64
    keys = list(result_dict.keys())
    for r in keys:
        if "per wave" in r:
            result_dict[r.replace(" per wave", "")] = result_dict[r] * waves
    return result_dict


def get_overall_kernel_count(file_path):
    # get the overall number of kernel launches from a omniperf profiling file. This is done by checking for the highest ID in the file and adding 1 (IDs start with 0)
    kernel_id = 0  # Initialize to 0 to handle files with no launches 
    in_dispatch_list = False

    with open(file_path, 'r') as file:
        for line in file:
            if 'Dispatch list' in line:
                in_dispatch_list = True
            elif in_dispatch_list:
                parts = line.split('|')
                if len(parts) > 1 and parts[1].strip().isdigit():
                    kernel_id = int(parts[1].strip())
    # Add 1 to convert the last Dispatch_ID to the total number of launches
    return kernel_id + 1

def profile_omni(executable, kernel_name, metrics, no_rerun):
    
    profile_command = ["omniperf profile -n gpu -- " + executable]
    # profile executable
    subprocess.run(profile_command, shell=True, check=True, stdout=subprocess.DEVNULL)


    # extract kernel number
    directory = "workloads/gpu/"
    folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder)) and folder.startswith('MI')]

    if len(folders) == 0:
        print("Found no profiling outputs in " + directory)

    if len(folders) > 1:
        print("Found multiple profiling outputs in " + directory + ". Please remove and try again")


    directory += folders[0]
    analyze_command = ["omniperf analyze -p " +  directory + " --list-stats > kernel_stats.txt"]
    subprocess.run(analyze_command, shell=True, check=True)
    

    file = None
    global accumulate
    if not accumulate:
        overall_kernel_count = 1
        kernel_line = get_line_with_substring_linebreak("kernel_stats.txt", kernel_name)  
        parts = kernel_line.split('│')
        kernel_number = parts[1].strip()

        # get stats for passed kernelname
        analyze_command = "omniperf analyze -p " + directory + " -k " + kernel_number + " > " + kernel_name + "_stats.txt"
        subprocess.run(analyze_command, shell=True, check=True)

        file = kernel_name + "_stats.txt"

        metric_dict = {}
        metric_dict["Runtime"] = float(get_line_with_substring_linebreak(file, kernel_name).split('│')[4].strip())
    else:
        # TODO remove, only used for debugging
        print("storing in:  results/" +kernel_name + "_stats.txt")
        analyze_command = "omniperf analyze -p " + directory +  " >  results/" +kernel_name + "_stats.txt"
        subprocess.run(analyze_command, shell=True, check=True)

        overall_kernel_count = get_overall_kernel_count("kernel_stats.txt")
        #kernel_line = get_line_with_substring_linebreak("kernel_stats.txt", kernel_name)  
        #parts = kernel_line.split('│')
        #kernel_number = parts[1].strip()
        kernel_name = "accumulate"
        # get stats for passed kernelname
        analyze_command = "omniperf analyze -p " + directory +  " > " + kernel_name + "_stats.txt"
        subprocess.run(analyze_command, shell=True, check=True)


        file = kernel_name + "_stats.txt"

        metric_dict = {}

        with open(kernel_name + "_stats.txt", 'r') as f:
            kernels = []
            record = False
            for line in f:
                # Start recording after "Top Kernels" header line
                if "Top Kernels" in line:
                    record = True
                elif "Dispatch List" in line:  # Stop recording before the "Dispatch List" section
                    break
                elif record:
                    line_parts = line.split('│')
                    if len(line_parts)>4 and contains_number(line_parts[1]):
                        kernels.append(int(float(line_parts[4].strip())))
            
            metric_dict["Runtime"] = sum(kernels)
    for name, prof_metric in metrics.items():
        line = get_line_with_substring(file, prof_metric)
        line_split = line.split('│')
        val = line_split[3].strip()
        if val == "":
            val = 0
        metric_value = float(val)
        metric_dict[name] = metric_value
    return metric_dict, overall_kernel_count


def profile_nsys(executable, kernel_name, no_rerun):
    if not (os.path.isabs(executable) or executable.startswith('./')):
        executable = "./" + executable
    if not no_rerun:
        #remove old profiling files so they are not accidentally used
        for filepath in glob.glob("*.qdrep"):
            os.remove(filepath)
        for filepath in glob.glob("*.nsys-rep"):
            os.remove(filepath)
        if os.path.isfile("nsys_reports.txt"):
            os.remove("nsys_reports.txt")
        if os.path.isfile("nsys_out.sqlite"):
            os.remove("nsys_out.sqlite")
        try:
            subprocess.run(["nsys profile --gpu-metrics-device=all -o nsys_out " + executable ], shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print("Profiling of " + executable + " failed, skipping benchmark")
            raise RuntimeError("Profiling of " + executable + " failed, skipping benchmark")       
    
    # output file ending may differ with nsys versions
    filename = ""
    if os.path.isfile("nsys_out.qdrep"):
        filename = "nsys_out.qdrep"
    elif os.path.isfile("nsys_out.nsys-rep"):
        filename = "nsys_out.nsys-rep"
 
    #TODO: check if gpu_kern_sum or gpukernsum is available
    
    subprocess.run(["nsys stats --report gpukernsum,gpumemtimesum " +  filename + " > nsys_reports.txt"], shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    kernel_names = extract_kernel_names_nsys("nsys_reports.txt")

    kernel_line = get_line_with_substring("nsys_reports.txt", kernel_name)
    print("number of kernels: " + str(len(kernel_names)))
    if kernel_line is None:
        if len(kernel_names) == 1:
            kernel_name = kernel_names[0]
            kernel_line = get_line_with_substring("nsys_reports.txt", kernel_name)
        else:
            print("Multiple kernels exist, thus accumulated metrics will be given in " + executable)
            
    dtoh_line = get_line_with_substring("nsys_reports.txt", "DtoH")
    htod_line = get_line_with_substring("nsys_reports.txt", "HtoD")
    global accumulate
    if not accumulate:
        kernel_line_list = kernel_line.split()

        kernel_time = int(kernel_line_list[1].replace(',', ''))
    else: 
        kernel_time = 0
        for k_name in kernel_names:
            kernel_line = get_line_with_substring("nsys_reports.txt", k_name)
            kernel_line_list = kernel_line.split()

            kernel_time += int(kernel_line_list[1].replace(',', ''))


    dtoh_list = dtoh_line.split() 
    dtoh_time = int(dtoh_list[1].replace(',', ''))  

    htod_list = htod_line.split()
    htod_time = int(htod_list[1].replace(',', '')) 

    metric_dict = {}

    metric_dict["Runtime"] = kernel_time
    metric_dict["HtoD copy time"] = htod_time
    metric_dict["DtoH copy time"] = dtoh_time

    if not accumulate or len(kernel_names) == 1:
        
        profile_command = "ncu --set full " + executable + " > ncu_full_out.txt"
        
        subprocess.run(profile_command, shell=True, check=True)
        
        
        #get thread/team config
        thread_line = get_line_with_substring("ncu_full_out.txt", "Block Size")
        thread = int(thread_line.split()[2].replace(',', ''))

        grid_line = get_line_with_substring("ncu_full_out.txt", "Grid Size")
        grid = int(grid_line.split()[2].replace(',', ''))
                
        metric_dict["#Threads"] = thread
        metric_dict["#Teams"] = grid


    ncu_metrics = {}
    ncu_sections = []
    
    ncu_metrics["Instructions Executed"] = "inst_executed"
    ncu_metrics["IPC"] = "sm__inst_executed.avg.per_cycle_active"
    # get #shared memory lds/stores (smsp__inst_executed_op_shared_st.sum, smsp__inst_executed_op_shared_ld.sum)
    ncu_metrics["Shared Memory Stores"] = "smsp__inst_executed_op_shared_st.sum"
    ncu_metrics["Shared Memory Loads"] = "smsp__inst_executed_op_shared_ld.sum"


    # get shared memory throughput: l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.per_second, l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second
    #ncu_metrics["Shared Memory Store Throughput"] = "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.per_second"
    #ncu_metrics["Shared Memory Load Troughput"] = "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second"

    # get shared memory bank conflicts: l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
    ncu_metrics["Shared Memory Store Bank Conflicts"] = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
    ncu_metrics["Shared Memory Load Bank Conflicts"] = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"


    # global store/load instructions
    ncu_metrics["Global Memory Load Instructions"] = "smsp__inst_executed_op_global_ld.sum"
    ncu_metrics["Global Memory Store Instructions"] = "smsp__inst_executed_op_global_st.sum"

    # global store/load requests
    ncu_metrics["Global Memory Load (MB)"] = "dram__bytes_read.sum"
    ncu_metrics["Global Memory Store (MB)"] = "dram__bytes_write.sum"

    # global store/load Throughput
    ncu_metrics["Global Memory Load Throughput (GB/s)"] = "dram__bytes_read.sum.per_second"
    ncu_metrics["Global Memory Store Throughput (GB/s)"] = "dram__bytes_write.sum.per_second"
    
    ncu_metrics["Global Memory Load Bank Conflicts"] = "l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_ld.sum"
    ncu_metrics["Global Memory Store Bank Conflicts"] = "l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_st.sum"


    ncu_metrics["L1 Cache Accesses (Bytes)"] = "l1tex__t_bytes.sum"
    ncu_metrics["L1 Cache Hits (%)"] = "l1tex__t_sector_hit_rate.pct"
    ncu_metrics["L1 Cache Throughput (GB/s)"] = "l1tex__t_bytes.sum.per_second"
    ncu_metrics["L2 Cache Accesses (Bytes)"] = "lts__t_bytes.sum"
    ncu_metrics["L2 Cache Hits (%)"] = "lts__t_sector_hit_rate.pct"
    ncu_metrics["L2 Cache Throughput (GB/s)"] = "lts__t_bytes.sum.per_second"
    
    ncu_metrics["#Divergent Branches"] = "smsp__sass_branch_targets_threads_divergent.sum"

    ncu_sections.append("Occupancy")
    
    ncu_sections.append("SourceCounters")
    #TODO add metric names from sections: Occupancy and Divergent Branches, Warp Stalls
    sections_string = ""
    for section in ncu_sections:
        sections_string += "--section " + section + " "
    metrics_string = ','.join(value for value in ncu_metrics.values() if isinstance(value, str))
    if accumulate:
        profile_command = "ncu --metrics " + metrics_string + " " + sections_string + executable + " > ncu_out.txt"
    else:
        profile_command = "ncu --metrics " + metrics_string + " " + sections_string + "-k regex:" + kernel_name + " "  + executable + " > ncu_out.txt"

    subprocess.run(profile_command, shell=True, check=True, stdout=subprocess.DEVNULL)
    
    # append metrics in sections

    ncu_metrics["Occupancy"] = "Achieved Occupancy"
    
    # get all metrics from profilig output and store them in dict
    for name, prof_metric in ncu_metrics.items():
        metric_lines = get_every_line_with_substring("ncu_out.txt", prof_metric)
        metric = 0
        for line in metric_lines:
            metric += float(line.split()[-1].replace(',', ''))
        
        # build average for metrics where this makes more sense: units like %, per second, ... 
        if ".per_second" in prof_metric or ".pct" in prof_metric or ".per_cycle_active" in prof_metric:
            metric = metric/len(metric_lines)
        metric_dict[name] = metric
    
    return metric_dict





def main():

    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Pass executables and GPU type to automatically extract profiling metrics that differ')

    # Add optional argument --gpu with choices of nvidia or amd
    parser.add_argument('--gpu', choices=['nvidia', 'amd'], default="nvidia", help='Optional argument specifying GPU type (nvidia or amd)')

    # Optional test_name to store results in 
    parser.add_argument('--test_name', default=None, help='Optional test name to store results in')

    # Add executables and names as pairs
    parser.add_argument('exec_name_pairs', nargs='+', help='Pairs of executables and their corresponding kernel names to profile. Example: ./exec1 name1 ./exec2 name2')

    parser.add_argument("--verbose", nargs="?", const=True, type=float, 
                    help="If verbose is set without a value, print all items that differ. If a number is passed, only print items that differ by at least that percentage.") 
    
    parser.add_argument("--no_rerun", action="store_true", help="Disable that profiler is run again.")

    parser.add_argument("--metrics", nargs="+", help="List of metrics that are not allowed to differ")

    parser.add_argument("--accumulate", action='store_true', help="Set this flag to accumulate profiling stats of all kernels within a execution")
    # Parse the arguments
    args = parser.parse_args()
    
    global accumulate
    accumulate = args.accumulate

    # Check that an even number of arguments (for executable-name pairs) was provided
    if len(args.exec_name_pairs) % 2 != 0:
        print(args.exec_name_pairs)
        raise ValueError("Please provide pairs of executables and their corresponding names.")

    # Separate executables and their names
    exec_name_pairs = [(args.exec_name_pairs[i], args.exec_name_pairs[i + 1]) for i in range(0, len(args.exec_name_pairs), 2)]
     
    results = {}

    print("Profiling for: " + args.gpu)
    for executable, name in exec_name_pairs:
        print("Profiling " + executable)
        res_dict = {}
        if args.gpu == "amd":
            try:
                res_dict = profile_amd(executable, name, args.no_rerun)
            except RuntimeError as e:
                return
        elif args.gpu == "nvidia":
            try:
                res_dict = profile_nsys(executable, name, args.no_rerun)
            except RuntimeError as e:
                return
        print("Storing in:")
        print(executable)
        results[executable] = res_dict
        print(results)
    
    # store all metrics in result csv file
    test_name = args.test_name

    stripped_dict = {key.rsplit('/', 1)[-1]: value for key, value in results.items()}
    pd.set_option('display.float_format', '{:.2f}'.format)
    df = pd.DataFrame(stripped_dict)
   
    if test_name is not None:
        if not os.path.exists("results"):
            os.makedirs("results")
        test_name = "results/" + test_name
        if not test_name.endswith('.csv'):
            test_name += '.csv'
        df.to_csv(test_name, index=True)
    # print errors for passed metrics that are not allowed to differ
    
    if args.metrics:
        for metric in args.metrics:
            if metric in results[list(results.keys())[0]]:
                vals = []
                for r in results:
                    vals.append(results[r][metric])
                if len(vals) > 0 and min(vals) != max(vals):
                    print("[Error] Metric " + metric + " differs in " + args.test_name)
                    stripped_dict = {key: {metric: results[key][metric]} for key, value in results.items()}
                    error_frame = pd.DataFrame(stripped_dict)
                    print(error_frame)

    # print warnings if values differ by more than the percentage passed in verbose
    pct = 0
    if args.verbose is not None:
        if args.verbose is True:
            pct = 0
        else: 
            pct = args.verbose
    keys = results[exec_name_pairs[0][0]].keys()
    del_keys = []
    for key in keys:
        # Get the value of the key from the first executable's dictionary
        vals = []
        for r in results:
            vals.append(results[r][key])
        min_value = min(vals)
        max_value = max(vals)
        percentage_diff = ((max_value - min_value) / min_value) * 100 if min_value != 0 else max_value - min_value
        if percentage_diff <= pct:
            del_keys.append(key)

    for key in del_keys:
        for executable, _ in exec_name_pairs:
            del results[executable][key]

    if len(results[exec_name_pairs[0][0]].keys()) > 0:    
        print("[WARNING] The following metrics differ by more than " + str(pct) + "% in " +test_name)
        stripped_dict = {key: value for key, value in results.items()}
        pd.set_option('display.float_format', '{:.2f}'.format)
        df = pd.DataFrame(stripped_dict)
        pd.set_option('display.max_columns', None)  # Show all columns
        print(df)
   
if  __name__ == "__main__":
    main()
