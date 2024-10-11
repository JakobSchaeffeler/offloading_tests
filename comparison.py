# -*- coding: utf-8 -*-
import subprocess
import os
import argparse
import pandas as pd

#TODO add optional argument to verbose, only print values that differ by at least x%
def main():

    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Pass profiling csv file  and metrics that are not allowed to differ')

    #parser.add_argument("--verbose", action="store_true", help="If verbose print all output that differs as [INFO]")
    parser.add_argument("--verbose", nargs="?", const=True, type=float, 
                    help="If verbose is set without a value, print all items that differ. If a number is passed, only print items that differ by at least that percentage.") 
    parser.add_argument("--metrics", nargs="+", help="List of metrics that are not allowed to differ")
    
    parser.add_argument("csv_file", type=str, help="Path to the profiling CSV file")

    # Parse the arguments
    args = parser.parse_args()
    

    df = pd.read_csv(args.csv_file)
    test = os.path.basename(args.csv_file)
    test = test.replace(".csv", "")
    print("Processing test results of " + test)
    if args.metrics:
        for metric in args.metrics:
            if metric in df.iloc[:, 0].values:  # assuming first column contains metric names
                # Filter the row for the given metric
                row = df[df.iloc[:, 0] == metric].iloc[:, 1:]  # skipping the first column (metric name)
                
                # Check if all values in the row are the same across benchmarks
                if row.nunique().values[0] != 1:  # nunique() gives number of unique values, != 1 means they differ
                    print(f"[ERROR] {metric} differs across benchmarks")
    if args.verbose is not None:
        metric_column = df.columns[0]  # First column is the metric name

        # Iterate over each metric in the CSV
        for index, row in df.iterrows():
            metric_name = row[metric_column]  # Get the metric name (from the first column)
            benchmark_values = row.iloc[1:]  # Skip the first column (metric names), get the benchmark values

            # Check if values differ across benchmarks
            if benchmark_values.nunique() != 1:  # nunique() gives number of unique values
                if args.verbose is True:
                    print(f"[INFO] {metric_name} differs")
                    result_df = pd.DataFrame({
                    "Benchmark": benchmark_columns,  # Column names (benchmarks)
                    "Value": benchmark_values.values  # Corresponding values for the metric
                    })
                    print(result_df.to_string(index=True)) 
                else:
                    # if percentage is passed, check if min and max value differ by at least args.verbose %
                    max_value = benchmark_values.max()
                    min_value = benchmark_values.min()
                    percentage_diff = ((max_value - min_value) / min_value) * 100  # Calculate percentage difference
                    if percentage_diff >= args.verbose:
                        print(f"[INFO] {metric_name} differs")
                        result_df = pd.DataFrame({
                        "Benchmark": benchmark_columns,  # Column names (benchmarks)
                        "Value": benchmark_values.values  # Corresponding values for the metric
                        })
                        print(result_df.to_string(index=True))


if  __name__ == "__main__":
    main()
