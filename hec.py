import os
import subprocess
import argparse
import re
import tarfile
import shlex
import shutil
import signal
import glob
import time

def capture_include_and_library_paths():
    # Run `make` in dry-run mode to capture output without compiling
    try:
        result = subprocess.run(
            ["make", "-n", "OPTIMIZE=yes"],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running make: {e}")
        return []

    # Capture output from `make`
    output = result.stdout

    # Regular expressions to match `-I` and `-L` options
    include_pattern = r"-I\s*([^\s]+)"
    library_pattern = r"-L\s*([^\s]+)"
    d_pattern = r"-D\s*([^\s]+)"
    o_pattern = r"-O\s*([^\s]+)"

    # Find all occurrences of `-I` and `-L` paths
    include_paths = re.findall(include_pattern, output)
    library_paths = re.findall(library_pattern, output)
    d_paths = re.findall(d_pattern, output)
    o_flag = re.findall(o_pattern, output)

    return include_paths, library_paths, d_paths, o_flag

def halve_numbers_in_string(s):
    def halve(match):
        number = int(match.group())
        return str(number)
        #return str(number // 4) if (number // 4) > 0 else str(1)

    # Use re.sub to replace each number in the string with half its value
    return re.sub(r'\d+', halve, s)


# process files in input so they are replaced with full paths
def process_files(output, base_path):
    # Regex to detect .bmp and .raw file paths
    file_pattern = r"(?:(?:\.\./|\.?/)?(?:[\w\-/]+(?:\.bmp|\.raw))|(?:/[^\s]+(?:\.bmp|\.raw)))"
    
    # Find all .bmp and .raw file paths
    detected_paths = re.findall(file_pattern, output)
    resolved_paths = {}
    for path in detected_paths:
        if path.startswith("../") or path.startswith("./") or not os.path.isabs(path):
            # Resolve to an absolute path if it's a relative path
            abs_path = os.path.abspath(os.path.join(base_path, path))
            resolved_paths[path] = abs_path
        else:
            # Path is already absolute
            resolved_paths[path] = path

    # Replace relative .bmp and .raw paths with absolute paths in the output
    for rel_path, abs_path in resolved_paths.items():
        output = output.replace(rel_path, abs_path)

    return output

def get_make_commands(name):
    try:
        # Run make with -n to list all commands executed by run
        result = subprocess.run(["make", "-n", "run", "program="+name], capture_output=True, text=True, check=True)

        # Filter out lines that look like shell commands
        commands = [line.strip() for line in result.stdout.splitlines() if line.strip() and not line.startswith("make")]

        # Process commands to resolve file paths
        resolved_commands = []
        base_path = os.getcwd()  # You can adjust this to another base path if needed

        for command in commands:
            # Replace relative .bmp and .raw file paths with absolute paths
            updated_command = process_files(command, base_path)
            # Append the updated command without halving numbers
            resolved_commands.append(updated_command)
        
        return resolved_commands

    except subprocess.CalledProcessError as e:
        print("An error occurred while trying to get make run commands", e)
        return []


def reduce_parameters(command):
    # reduces all int parameters by factor 2
    parts = shlex.split(command)  # Split the command into parts
    reduced_parts = []
    for part in parts:
        if part.isdigit():  # Check if the part is a number
            reduced_parts.append(str(max(1, int(int(part) // 2))))  # Reduce by 2, minimum 1
        else:
            reduced_parts.append(part)
    return " ".join(reduced_parts)

def remove_compilation_commands(command_list):
    return [cmd for cmd in command_list if cmd.startswith("./") and not (".o" in cmd and not ".out" in cmd)]



def main():
    parser = argparse.ArgumentParser(description='Script to check for HeC benchmarks available in sycl, omp and cuda/hip and builds and then profiles them')

    parser.add_argument('--gpu', choices=['nvidia', 'amd'], default="nvidia", help='Optional argument specifying GPU type (nvidia or amd)') 
    
    # arguments for passing compilers and flags
    parser.add_argument("--cuda_compiler", type=str, help="CUDA compiler")
    parser.add_argument("--cuda_flags", type=str, help="CUDA compilation flags")
    parser.add_argument("--hip_compiler", type=str, help="HIP compiler")
    parser.add_argument("--hip_flags", type=str, help="HIP compilation flags")
    parser.add_argument("--omp_compiler", type=str, help="OpenMP compiler")
    parser.add_argument("--omp_flags", type=str, help="OpenMP compilation flags")
    parser.add_argument("--sycl_compiler", type=str, help="SYCL compiler")
    parser.add_argument("--sycl_flags", type=str, help="SYCL compilation flags")
    parser.add_argument("--arch", default="sm_70", help="architecture to compile for, e.g sm_70, gfx908,..." )
    args = parser.parse_args()

    # Repository URL and local path
    repo_url = "https://github.com/zjin-lcf/HeCBench.git"
    local_path = "HeCBench"

    # Check if the repository is already cloned
    if not os.path.isdir(local_path):
        # Clone the repository
        print(f"Cloning repository from {repo_url}...")
        subprocess.run(["git", "clone", repo_url])

    # Untar files in HeCBench
    for root, dirs, files in os.walk("HeCBench"):
        for file in files:
            if file.endswith('.tar.gz'):
                tar_path = os.path.join(root, file)
                try:
                    with tarfile.open(tar_path, 'r:gz') as tar:
                        tar.extractall(path=root, filter='data')

                except Exception as e: 
                    print("Failed to extract tar file " + tar_path)

    # Check for suffixes in src directory
    suffixes = [ "-sycl", "-omp"]
    if args.gpu == "amd":
        suffixes.append("-hip")
    elif args.gpu == "nvidia":
        suffixes.append("-cuda")

    matching_basenames = []

    src_path = os.path.join(local_path, "src")

    if os.path.isdir(src_path):
        # Get the base names of directories without suffixes
        potential_basenames = set(item.rsplit("-", 1)[0] for item in os.listdir(src_path)
                                  if os.path.isdir(os.path.join(src_path, item)))
        
        # Check if all suffixes are present for each base name
        for basename in potential_basenames:
            if all(os.path.isdir(os.path.join(src_path, f"{basename}{suffix}")) for suffix in suffixes):
                matching_basenames.append(basename)

    matching_basenames.sort()
    
    print("Directories with all three suffixes:", matching_basenames)

    # create directory for compilation output
    build_dir = "build"
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
        
    if not os.path.exists("results"):
        os.makedirs("results")

    results_path = os.path.abspath("results")
    failed_benchmarks = []
    successfull_benchmarks = []
    timeout_benchmarks = []
    #remove all benchmarks already profiled from benchmark list
    # benchmarks are already profiled if a .csv exists in results with the name
    matching_basenames = [name for name in matching_basenames if not os.path.isfile(os.path.join(results_path, f"{name}.csv"))]

    
    print("Removed all benchmarks that already have results in results directory, the following benchmarks will be profiled:") 
    print(matching_basenames)
    # compile for each suffix with proper flags
    for name in matching_basenames:
        run_scale_factor = 1
        error_dir = os.path.join(results_path, name)
        os.makedirs(error_dir, exist_ok=True)
        
       
        os.environ["ARCH"] = args.arch
        
        os.chdir("HeCBench/src")

        # for openmp: go into subdir and compile with appropriate flags 
        os.chdir(name + "-omp")
        
        # some benchmarks do not have a Makefile and have to be skipped
        if not os.path.isfile("Makefile"):
            print("No Makefile found for OpenMP implementation, continuing with next benchmark")
            os.chdir("../../../")
            continue
        
        # if Makefile exists make variables consistent
        with open("Makefile", "r") as file:
            content = file.read()
        
        # Replace 
        content = re.sub(r'\bNVCC\b', 'CC', content)
        content = re.sub(r'\bNVCC_FLAGS\b', 'CFLAGS', content)

        # Write the updated Makefile back
        with open("Makefile", "w") as file:
            file.write(content)
 
        include_paths, library_paths, d_paths, o_flag = capture_include_and_library_paths()
        cflags = " ".join([f"-I{path}" for path in include_paths] + [f"-L{path}" for path in library_paths] + [f"-D{path}" for path in d_paths] + [f"-O{flag}" for flag in o_flag])
        command = ["make"]
        command.append("CC=" + args.omp_compiler)
        command.append("CFLAGS=" + args.omp_flags + " " + cflags)
        
        command.append("program=omp OPTIMIZE=yes")
        subprocess.run(["make", "clean"],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # run make command and store logs
        error_file_omp = os.path.join(error_dir, "omp.log")
        with open(error_file_omp, "w") as err_file:
            err_omp = subprocess.run(command, check=False, stdout=err_file, stderr=err_file)
        
        run_commands_omp=get_make_commands("omp")
        
        os.chdir("../")

        # sycl compilation and get run commands
        os.chdir(name + "-sycl")
        include_paths, library_paths, d_paths, o_flag = capture_include_and_library_paths()
        
        cflags = " ".join([f"-I{path}" for path in include_paths] + [f"-L{path}" for path in library_paths] + [f"-D{path}" for path in d_paths] + [f"-O{flag}" for flag in o_flag])
        command = ["make"]
        if args.sycl_compiler is not None:
            command.append("CC=" + args.sycl_compiler)
        if args.sycl_flags is not None:
            command.append("CFLAGS=" + args.sycl_flags + " " + cflags)
        if args.gpu == "nvidia":
            command.append("CUDA=yes")
        else: 
            command.append("HIP=yes")
        command.append("program=sycl OPTIMIZE=yes")
        subprocess.run(["make", "clean"],stdout=subprocess.PIPE, stderr=subprocess.PIPE)

         # run make command and store logs
        error_file_sycl = os.path.join(error_dir, "sycl.log")
        with open(error_file_sycl, "w") as err_file:
            err_sycl = subprocess.run(command, check=False, stdout=err_file, stderr=err_file)
        
        run_commands_sycl=get_make_commands("sycl")

        os.chdir("../")
        
        
        # cuda/hip compilation and get run commands
        run_commands_low_level = None
        low_level_name = ""
        if args.gpu == "amd":
            low_level_name = "hip"
            os.chdir(name + "-hip")
            include_paths, library_paths, d_paths, o_flag = capture_include_and_library_paths()
            cflags = " ".join([f"-I{path}" for path in include_paths] + [f"-L{path}" for path in library_paths] + [f"-D{path}" for path in d_paths] + [f"-O{flag}" for flag in o_flag])
            command = ["make"]
            if args.hip_compiler is not None:
                command.append("CC=" + args.hip_compiler)
            if args.hip_flags is not None:
                command.append("CFLAGS=" + args.hip_flags + " " + cflags)
            command.append("program=hip")
            subprocess.run(["make", "clean"],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # run make command and store logs
            error_file_low_level = os.path.join(error_dir, "hip.log")
            with open(error_file_low_level, "w") as err_file:
                err_low_level = subprocess.run(command, check=False, stdout=err_file, stderr=err_file)

            run_commands_low_level=get_make_commands("hip")

        else:
            low_level_name = "cuda"
            os.chdir(name + "-cuda")
            include_paths, library_paths, d_paths, o_flag = capture_include_and_library_paths()
            cflags = " ".join([f"-I{path}" for path in include_paths] + [f"-L{path}" for path in library_paths] + [f"-D{path}" for path in d_paths] + [f"-O{flag}" for flag in o_flag])
            command = ["make"]
            if args.cuda_compiler is not None:
                command.append("CC=" + args.cuda_compiler)
            if args.cuda_flags is not None:
                command.append("CFLAGS=" + args.cuda_flags + " " + cflags)
            command.append("program=cuda")
            command.append("ARCH="+ args.arch)
            subprocess.run(["make", "clean"],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # run make command and store logs
            error_file_low_level = os.path.join(error_dir, "cuda.log")
            with open(error_file_low_level, "w") as err_file:
                err_low_level = subprocess.run(command, check=False, stdout=err_file, stderr=err_file)

            run_commands_low_level=get_make_commands("cuda")
        
        os.chdir("../")
        os.chdir("../")
        os.chdir("../")

        run_commands_low_level = remove_compilation_commands(run_commands_low_level)
        run_commands_omp = remove_compilation_commands(run_commands_omp)
        run_commands_sycl = remove_compilation_commands(run_commands_sycl)
        print("Profiling the following programs")
        print(run_commands_low_level)
        print(run_commands_omp)
        print(run_commands_sycl)

        if len(run_commands_low_level) != len(run_commands_omp) or len(run_commands_omp) != len(run_commands_sycl):
            print("Number of commands in make run do not match in " + name + ". Skipping benchmark")
            failed_benchmarks.append(name)
            continue


        #profile everything, if multiple run in make run profile each individually
        if err_low_level.returncode != 0:
            print("Compilation of cuda/hip for " + name + " failed, continuing with next benchmark")
            print("For build log check " + error_dir)
            failed_benchmarks.append(name)
            continue
        if err_omp.returncode != 0:
            print("Complation of omp for " + name + " failed, continuing with next benchmark")
            print("For build log check " + error_dir)
            failed_benchmarks.append(name)
            continue
        if err_sycl.returncode != 0:
            print("Compilation of sycl for " + name + " failed, continuing with next benchmark")
            print("For build log check " + error_dir)
            failed_benchmarks.append(name)
            continue

        if len(run_commands_omp) == 1:
            retries = 0
            result = None
            while retries < 5:
                try: 
                    result = subprocess.Popen(["python", "profiling.py",
                        "--gpu", args.gpu,
                        "--accumulate",
                        "--verbose", "0" ,
                        "--test_name", name,
                        "HeCBench/src/" + name + "-omp/" + run_commands_omp[0],
                        "omp offloading kernel", 
                        "HeCBench/src/" + name + "-sycl/" + run_commands_sycl[0],
                        "sycl kernel", 
                        "HeCBench/src/" + name + "-" + low_level_name + "/" + run_commands_low_level[0], 
                        low_level_name + "_kernel"], start_new_session=True)
                    result.wait(timeout=3600)
                    break

                # if a timeout is reached the process has to be killed, otherwise the profiling tool will continue to execute and block resources
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(result.pid), signal.SIGTERM)
                    time.sleep(1)
                    retries += 1
                    run_commands_omp[0] = reduce_parameters(run_commands_omp[0])
                    run_commands_sycl[0] = reduce_parameters(run_commands_sycl[0])
                    run_commands_low_level[0] = reduce_parameters(run_commands_low_level[0])
            
            if retries == 5:
                print("Benchmark " + name + " timeouted, continuing with next one")
                timeout_benchmarks.append(name)
                continue
            if result.returncode != 0:
                print("Profiling of benchmark " + name + " failed")
                failed_benchmarks.append(name)
                continue
 

        else:
            # if multiple run commands exists profile them individually and store in <benchmarkname>_i.csv 
            for i in range(len(run_commands_omp)):
                retries = 0
                while retries < 5:
                    try:
                         result = subprocess.Popen(["python", "profiling.py",
                            "--gpu", args.gpu,
                            "--accumulate",
                            "--verbose", "0" ,
                            "--test_name", name + "_"+ str(i),
                            "HeCBench/src/" + name + "-omp/" + run_commands_omp[i],
                            "omp offloading kernel", 
                            "HeCBench/src/" + name + "-sycl/" + run_commands_sycl[i],
                            "sycl kernel", 
                            "HeCBench/src/" + name + "-" + low_level_name + "/" + run_commands_low_level[i], 
                            low_level_name + "_kernel"
                            ], start_new_session=True)
                         result.wait(timeout=3600)
                         break
                                                            
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(result.pid), signal.SIGTERM)
                        retries += 1
                        run_commands_omp[0]
                        run_commands_omp[0] = reduce_parameters(run_commands_omp[0])
                        run_commands_sycl[0] = reduce_parameters(run_commands_sycl[0])
                        run_commands_low_level[0] = reduce_parameters(run_commands_low_level[0])
                        print(run_commands_omp[0])

                if retries == 5:
                    print("Benchmark " + name + " timeouted, continuing with next one")
                    failed_benchmarks.append(name)
                    continue
                if result.returncode != 0:
                    print("Profiling of benchmark " + name + " failed")
                    failed_benchmarks.append(name)
                    continue
                   
    
        successfull_benchmarks.append(name)
        print("Successfull benchmarks so far:")
        print(successfull_benchmarks)
        print("Build Failed so far for :")
        print(failed_benchmarks)
        print("Timeouted Benchmarks so far:")
        print(timeout_benchmarks)


    print("Benchmarking finished")
    print("Build failed for:")
    print(failed_benchmarks)
    print("Check build logs in result directory")
    print("Timeouted Benchmarks:")
    print(timeout_benchmarks)



if  __name__ == "__main__":
    main()

