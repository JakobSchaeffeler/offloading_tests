import os
import subprocess
import argparse

def get_make_commands():
    try:
        # Run make with -n o list all commands executed by run
        result = subprocess.run("make -n run", capture_output=True, text=True, check=True)

        # Filter out lines that look like shell commands
        commands = [line.strip() for line in result.stdout.splitlines() if line.strip() and not line.startswith("make")]

        return commands

    except subprocess.CalledProcessError as e:
        print("An error occurred while trying to get make run commands", e)
        return []


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

    # Output the list of matching base names
    print("Directories with all three suffixes:", matching_basenames)

    # TODO for now reduce number of benchmarks
    matching_basenames = matching_basenames[0:5]
    
    # create directory for compilation output
    build_dir = "build"
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    # compile for each suffix with proper flags
    for name in matching_basenames: 
        os.environ["ARCH"] = args.arch
        
        os.chdir("HeCBench/src")

        # for openmp: go into subdir and compile with appropriate flags 
        os.chdir(name + "-omp")
        err_omp = subprocess.run(["make " + ("CC=" + args.omp_compiler) if args.omp_compiler is not None else "" + (" CFLAGS=" + args.omp_flags) if args.omp_flags is not None else ""], check=True) 
        run_commands_omp=get_make_commands()
    
        
        os.chdir("../")

        # sycl compilation and get run commands
        os.chdir(name + "-sycl")
        err_sycl = subprocess.run(["make " + ("CC=" + args.sycl_compiler) if args.sycl_compiler is not None else "" + (" CFLAGS=" + args.sycl_flags) if args.sycl_flags is not None else ""], check=True)
        
        run_commands_sycl=get_make_commands()

        os.chdir("../")

        
        # cuda/hip compilation and get run commands
        run_commands_low_level = None
        low_level_name = ""
        if args.gpu == "amd":
            low_level_name = "hip"
            os.chdir(name + "-hip")
            err_low_level = subprocess.run(["make " + ("CC=" + args.hip_compiler) if args.hip_compiler is not None else "" + (" CFLAGS=" + args.hip_flags) if args.hip_flags is not None else ""], check=True)
        else:
            low_level_name = "cuda"
            os.chdir(name + "-cuda")
            err_low_level = subprocess.run(["make " + ("CC=" + args.cuda_compiler) if args.cuda_compiler is not None else "" + (" CFLAGS=" + args.cuda_flags) if args.cuda_flags is not None else ""], check=True)
       
        run_commands_low_level=get_make_commands()

        if len(run_commands_omp) != len(run_commands_sycl) or len(run_commands_sycl) != len(run_commands_low_level):
            print("Did not find matching number of commands executed for different models " + name)

        #profile everything, if multiple run in make run profile each individually

        if len(run_commands_omp) == 1:
            subprocess.run("python profiling.py --verbose 0 --gpu " + args.gpu + "HeCBench/src/ + " + name + "-omp/" + run_commands_omp[0] + " omp offloading"  + "HeCBench/src/ + " + name + "-sycl/" + run_commands_sycl[0] + " SyCL" + "HeCBench/src/ + " + name + "-" + low_level_name + "/" + run_commands_low_level[0] + " " + low_level_name + " --test_name " + name)
        else: 
            for i in range(len(run_commands_omp)):
                subprocess.run("python profiling.py --verbose 0 --gpu " + args.gpu + "HeCBench/src/ + " + name + "-omp/" + run_commands_omp[i] + " omp offloading"  + "HeCBench/src/ + " + name + "-sycl/" + run_commands_sycl[i] + " SyCL" + "HeCBench/src/ + " + name + "-" + low_level_name + "/" + run_commands_low_level[i] + " " + low_level_name + " --test_name \"" + name + " " + run_commands_omp[i] + "\"" )
        
    



if  __name__ == "__main__":
    main()

