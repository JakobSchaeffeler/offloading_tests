import os
import subprocess
import argparse
import re

def capture_include_and_library_paths():
    # Run `make` in dry-run mode to capture output without compiling
    try:
        result = subprocess.run(
            ["make", "-n"],
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


    # Find all occurrences of `-I` and `-L` paths
    include_paths = re.findall(include_pattern, output)
    library_paths = re.findall(library_pattern, output)
    d_paths = re.findall(d_pattern, output)

    return include_paths, library_paths, d_paths

def halve_numbers_in_string(s):
    def halve(match):
        number = int(match.group())
        return str(number // 4) if (number // 4) > 0 else str(1)

    # Use re.sub to replace each number in the string with half its value
    return re.sub(r'\d+', halve, s)

def get_make_commands(name):
    try:
        # Run make with -n o list all commands executed by run
        result = subprocess.run(["make", "-n", "run", "program="+name], capture_output=True, text=True, check=True)

        # Filter out lines that look like shell commands
        commands = [line.strip() for line in result.stdout.splitlines() if line.strip() and not line.startswith("make")]
        
        halve_list = []
        for command in commands:
            halve_list.append(halve_numbers_in_string(command))
        return halve_list

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

    matching_basenames.sort()
    
    print("Directories with all three suffixes:", matching_basenames)
    # TODO for now reduce number of benchmarks
    #matching_basenames = ["aidw"]
    
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
        
        include_paths, library_paths, d_paths = capture_include_and_library_paths()
        cflags = " ".join([f"-I{path}" for path in include_paths] + [f"-L{path}" for path in library_paths] + [f"-D{path}" for path in d_paths])
        command = ["make"]
        if args.omp_compiler is not None:
            command.append("CC=" + args.omp_compiler)
        if args.omp_flags is not None:
            command.append("CFLAGS=" + args.omp_flags + " " + cflags)
        
        command.append("program=omp")
        subprocess.run(["make", "clean"])
        err_omp = subprocess.run(command, check=True) 
        
        run_commands_omp=get_make_commands("omp")
        
        os.chdir("../")

        # sycl compilation and get run commands
        os.chdir(name + "-sycl")
        include_paths, library_paths, d_paths = capture_include_and_library_paths()
        #print(include_paths)
        #print(library_paths)
        cflags = " ".join([f"-I{path}" for path in include_paths] + [f"-L{path}" for path in library_paths] + [f"-D{path}" for path in d_paths])
        #print(cflags)
        command = ["make"]
        if args.sycl_compiler is not None:
            command.append("CC=" + args.sycl_compiler)
        if args.sycl_flags is not None:
            command.append("CFLAGS=" + args.sycl_flags + " " + cflags)
        if args.gpu == "nvidia":
            command.append("CUDA=yes")
        else: 
            command.append("HIP=yes")
        command.append("program=sycl")
        subprocess.run(["make", "clean"])

        err_sycl = subprocess.run(command, check=True) 
        
        run_commands_sycl=get_make_commands("sycl")

        os.chdir("../")
        
        
        # cuda/hip compilation and get run commands
        run_commands_low_level = None
        low_level_name = ""
        if args.gpu == "amd":
            low_level_name = "hip"
            os.chdir(name + "-hip")
            include_paths, library_paths, d_paths = capture_include_and_library_paths()
            cflags = " ".join([f"-I{path}" for path in include_paths] + [f"-L{path}" for path in library_paths] + [f"-D{path}" for path in d_paths])
            command = ["make"]
            if args.hip_compiler is not None:
                command.append("CC=" + args.hip_compiler)
            if args.hip_flags is not None:
                command.append("CFLAGS=" + args.hip_flags + " " + cflags)
            command.append("program=hip")
            subprocess.run(["make", "clean"])
            err_low_level = subprocess.run(command, check=True) 
            run_commands_low_level=get_make_commands("hip")

        else:
            low_level_name = "cuda"
            os.chdir(name + "-cuda")
            command = ["make"]
            if args.cuda_compiler is not None:
                command.append("CC=" + args.cuda_compiler)
            if args.cuda_flags is not None:
                command.append("CFLAGS=" + args.cuda_flags)
            command.append("program=cuda")
    
            subprocess.run(["make", "clean"])

            err_low_level = subprocess.run(command, check=True) 
 
            run_commands_low_level=get_make_commands("cuda")
        
        os.chdir("../")
        os.chdir("../")
        os.chdir("../")


        print(run_commands_low_level)
        print(run_commands_omp)
        print(run_commands_sycl)
        #if len(run_commands_omp) != len(run_commands_sycl) or len(run_commands_sycl) != len(run_commands_low_level):
        #    print("Did not find matching number of commands executed for different models " + name)

        #profile everything, if multiple run in make run profile each individually

        if len(run_commands_omp) == 1:
            #subprocess.run("python profiling.py --verbose 0 --gpu " + args.gpu + "HeCBench/src/ + " + name + "-omp/" + run_commands_omp[0] + " omp offloading"  + "HeCBench/src/ + " + name + "-sycl/" + run_commands_sycl[0] + " SyCL" + "HeCBench/src/ + " + name + "-" + low_level_name + "/" + run_commands_low_level[0] + " " + low_level_name + " --test_name " + name)
            subprocess.run(["python", "profiling.py",
                "--gpu", args.gpu,
                "--accumulate",
                "--verbose", "0" ,
                "--test_name", name,
                "HeCBench/src/" + name + "-omp/" + run_commands_omp[0],
                "omp offloading kernel", 
                "HeCBench/src/" + name + "-sycl/" + run_commands_sycl[0],
                "sycl kernel", 
                "HeCBench/src/" + name + "-" + low_level_name + "/" + run_commands_low_level[0], 
                low_level_name + "_kernel"
                ])
        else: 
            for i in range(len(run_commands_omp)):
                #subprocess.run("python profiling.py --verbose 0 --gpu " + args.gpu + "HeCBench/src/ + " + name + "-omp/" + run_commands_omp[i] + " omp offloading"  + "HeCBench/src/ + " + name + "-sycl/" + run_commands_sycl[i] + " SyCL" + "HeCBench/src/ + " + name + "-" + low_level_name + "/" + run_commands_low_level[i] + " " + low_level_name + " --test_name \"" + name + " " + run_commands_omp[i] + "\"" )
                subprocess.run(["python", "profiling.py",
                        "--gpu", args.gpu,
                        "--accumulate",
                        "--verbose", "0" ,
                        "--test_name", name,
                        "HeCBench/src/" + name + "-omp/" + run_commands_omp[i],
                        "omp offloading kernel", 
                        "HeCBench/src/" + name + "-sycl/" + run_commands_sycl[i],
                        "sycl kernel", 
                        "HeCBench/src/" + name + "-" + low_level_name + "/" + run_commands_low_level[i], 
                        low_level_name + "_kernel"
                        ])


    



if  __name__ == "__main__":
    main()

