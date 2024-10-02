# -*- coding: utf-8 -*-
import subprocess
import os
import argparse
import pandas as pd



def get_line_with_substring(filename, substring):
    with open(filename, 'r') as file:
        for line in file:
            # Check if the line contains the desired substring
            if substring in line:
                return line

    print("Error in parsing profiling output: No line containing the substring " + substring + " was found.")
    return None


def profile_amd(executable, kernel_name):
    metrics = {}
    
    
    metrics["Grid Size"] = "7.1.0"
    metrics["#Threads"] = "7.1.1"
    metrics["Shared Memory Allocated"] = "3.1.14"
    metrics["Shared Memory Instructions per wave"] = "12.2.0"
    metrics["Shared Memory Bank Conflicts (%)"] = "12.1.3"
    metrics["Register Spill Instructions per wave"] = "15.1.9"
    metrics["Global Memory Instructions per wave"] = "10.3.0"
    metrics["Global Memory Read Instructions per wave"] = "10.3.1"
    metrics["Global Memory Write Instructions per wave"] = "10.3.2"
    metrics["Coalesced Instructions (% of peak)"] = "16.1.3"

    result_dict = profile_omni(executable,kernel_name, metrics)
    return result_dict


def profile_omni(executable, kernel_name, metrics):
    
    # profile executable
    profile_command = ["omniperf profile -n gpu -- " + executable]

    subprocess.run(profile_command, shell=True, check=True, stdout=subprocess.DEVNULL)


    # extract kernel number
    directory = "workloads/gpu/"
    folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder)) and folder.startswith('MI')]

    if len(folders) == 0:
        print("Found no profiling outputs in " + directory)

    if len(folders) > 1:
        print("Found no profiling outputs in " + directory + ". Please remove and try again")


    directory += folders[0]
    analyze_command = ["omniperf analyze -p " +  directory + " --list-stats > kernel_stats.txt"]
    subprocess.run(analyze_command, shell=True, check=True)

    kernel_line = get_line_with_substring("kernel_stats.txt", kernel_name)  
    
    parts = kernel_line.split('│')
    kernel_number = parts[1].strip()

    # get stats for passed kernelname
    analyze_command = "omniperf analyze -p " + directory + " -k " + kernel_number + " > " + kernel_name + "_stats.txt"
    subprocess.run(analyze_command, shell=True, check=True)

    file = kernel_name + "_stats.txt"

    metric_dict = {}
    
    metric_dict["Runtime"] = float(get_line_with_substring(file, kernel_name).split('│')[4].strip())

    for name, prof_metric in metrics.items():
        line = get_line_with_substring(file, prof_metric)
        line_split = line.split('│')
        val = line_split[3].strip()
        if val == "":
            val = 0
        metric_value = float(val)
        metric_dict[name] = metric_value
    return metric_dict


def profile_nsys(executable, kernel_name):
    subprocess.run(["nsys profile --gpu-metrics-device=all -o nsys_out.qdrep --force-overwrite true " + executable], shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
   
    #TODO: check if gpu_kern_sum or gpukernsum is available
    subprocess.run(["nsys stats --report gpukernsum,gpumemtimesum  nsys_out.qdrep > nsys_reports.txt"], shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    kernel_line = get_line_with_substring("nsys_reports.txt", kernel_name)
    dtoh_line = get_line_with_substring("nsys_reports.txt", "DtoH")
    htod_line = get_line_with_substring("nsys_reports.txt", "HtoD")

    kernel_line_list = kernel_line.split()

    kernel_time = int(kernel_line_list[1].replace(',', ''))

    dtoh_list = dtoh_line.split()
    
    dtoh_time = int(dtoh_list[1].replace(',', ''))  

    htod_list = htod_line.split()
    htod_time = int(htod_list[1].replace(',', '')) 

    metric_dict = {}

    metric_dict["Runtime"] = kernel_time
    metric_dict["HtoD copy time"] = htod_time
    metric_dict["DtoH copy time"] = dtoh_time


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

    # get #shared memory lds/stores (smsp__inst_executed_op_shared_st.sum, smsp__inst_executed_op_shared_ld.sum)
    ncu_metrics["Shared Memory Stores"] = "smsp__inst_executed_op_shared_st.sum"
    ncu_metrics["Shared Memory Loads"] = "smsp__inst_executed_op_shared_ld.sum"


    # get shared memory throughput: l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.per_second, l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second
    ncu_metrics["Shared Memory Store Throughput"] = "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.per_second"
    ncu_metrics["Shared Memory Load Troughput"] = "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second"

    # get shared memory bank conflicts: l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
    ncu_metrics["Shared Memory Store Bank Conflicts"] = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
    ncu_metrics["Shared Memory Load Bank Conflicts"] = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"


    # global store/load instructions
    ncu_metrics["Global Memory Load Instructions"] = "smsp__inst_executed_op_global_ld.sum"
    ncu_metrics["Global Memory Store Instructions"] = "smsp__inst_executed_op_global_st.sum"

    # global store/load requests
    ncu_metrics["Global Memory Load Requests"] = "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
    ncu_metrics["Global Memory Store Requests"] = "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"

    # global store/load Throughput
    ncu_metrics["Global Memory Load Throughput"] = "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second"
    ncu_metrics["Global Memory Store Throughput"] = "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second"

    # TODO register spills. could be done using GPU Scout analysis. Or indicatory metrics as l1 accesses

    # TODO coalescing 
    
    metrics_string = ','.join(value for value in ncu_metrics.values() if isinstance(value, str))
    profile_command = "ncu --metrics " + metrics_string + " " + executable + " > ncu_out.txt"
    subprocess.run(profile_command, shell=True, check=True, stdout=subprocess.DEVNULL)


    # get all metrics from profilig output and store them in dict
    for name, prof_metric in ncu_metrics.items():
        metric_line = get_line_with_substring("ncu_out.txt", prof_metric)
        metric = float(metric_line.split()[-1].replace(',', ''))
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

    # Parse the arguments
    args = parser.parse_args()

    # Check that an even number of arguments (for executable-name pairs) was provided
    if len(args.exec_name_pairs) % 2 != 0:
        raise ValueError("Please provide pairs of executables and their corresponding names.")

    # Separate executables and their names
    exec_name_pairs = [(args.exec_name_pairs[i], args.exec_name_pairs[i + 1]) for i in range(0, len(args.exec_name_pairs), 2)]
    
    results = {}

    print("Profiling for: " + args.gpu)
    for executable, name in exec_name_pairs:
        print("Profiling " + executable)
        res_dict = {}
        if args.gpu == "amd":
            res_dict = profile_amd(executable, name)
            # Number of waves = #Teams/32
        elif args.gpu == "nvidia":
            res_dict = profile_nsys(executable, name)

        results[executable] = res_dict

    
    keys = results[exec_name_pairs[0][0]].keys()
    del_keys = []

    for key in keys:
        # Get the value of the key from the first executable's dictionary
        first_value = results[exec_name_pairs[0][0]][key]
    
        # Compare this value with the same key in the other dictionaries
        all_same = True
        for executable, name in exec_name_pairs[1:]:
            # If the value differs from the first one, set all_same to False
            if results[executable][key] != first_value:
                all_same = False
                break
    
        # If all values for this key are the same, remove it from all dictionaries
        if all_same:
            del_keys.append(key)

    for key in del_keys:
        for executable, _ in exec_name_pairs:
            del results[executable][key]

    
    test_name = args.test_name

    stripped_dict = {key.rsplit('/', 1)[-1]: value for key, value in results.items()}
    pd.set_option('display.float_format', '{:.2f}'.format)
    df = pd.DataFrame(stripped_dict)
    if test_name is not None:
        if not test_name.endswith('.csv'):
            test_name += '.csv'
        df.to_csv(test_name, index=True)
    print(df)

if  __name__ == "__main__":
    main()
