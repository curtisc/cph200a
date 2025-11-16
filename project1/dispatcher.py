import argparse
import json
import random
import os, subprocess
from csv import DictWriter
import multiprocessing

def add_main_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config_path",
        type=str,
        default="grid_search.json",
        help="Location of config file"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of processes to run in parallel"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Location of experiment logs and results"
    )

    parser.add_argument(
        "--grid_search_results_path",
        default="grid_results.csv",
        help="Where to save grid search results"
    )

    return parser

def get_experiment_list(config: dict) -> (list[dict]):
    '''
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item, but the config contains a list, this will return one job for each item in the list.
    :config - experiment_config

    returns: jobs - a list of dicts, each of which encapsulates one job.
        *Example: {learning_rate: 0.001 , batch_size: 16 ...}
    '''
    jobs = [{}]

    # Go through the tree of possible jobs and enumerate into a list of jobs
    for key, value in config.items():
        if isinstance(value, list):
            new_jobs = []
            for v in value:
                for job in jobs:
                    new_job = job.copy()
                    new_job[key] = v
                    new_jobs.append(new_job)
            jobs = new_jobs
        else:
            for job in jobs:
                job[key] = value

    print(jobs)

    return jobs

def worker(args: argparse.Namespace, job_queue: multiprocessing.Queue, done_queue: multiprocessing.Queue):
    '''
    Worker thread for each worker. Consumes all jobs and pushes results to done_queue.
    :args - command line args
    :job_queue - queue of available jobs.
    :done_queue - queue where to push results.
    '''
    # Reseed random number generator for this worker process
    import numpy as np
    import time

    while not job_queue.empty():
        params = job_queue.get()
        if params is None:
            return
        done_queue.put(
            launch_experiment(args, params))


def launch_experiment(args: argparse.Namespace, experiment_config: dict) ->  dict:
    '''
    Launch an experiment and direct logs and results to a unique filepath.
    :configs: flags to use for this model run. Will be fed into
    scripts/main.py

    returns: flags for this experiment as well as result metrics
    '''

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    # Launch the experiment
    log_path = os.path.join(args.log_dir, "log_{}.txt".format(random.randint(0, 1000000)))
    results_path = os.path.join(args.log_dir, "results_{}.json".format(random.randint(0, 1000000)))
    command = ["python3", "main.py", "--results_path", results_path]
    for key, value in experiment_config.items():
        command.append("--{}".format(key))
        command.append("{}".format(value))
    with open(log_path, "w") as log_file:
        print(command)
        subprocess.run(command, stdout=log_file, stderr=log_file)
    print("Launched experiment with config: {}. Logs: {}, Results: {}".format(experiment_config, log_path, results_path))

    # Parse the results from the experiment and return them as a dict
    if os.path.isfile(results_path):
        results = json.load(open(results_path, "r"))
        results.update(experiment_config)
    else:
        results = experiment_config.copy()

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_main_args(parser)
    args = parser.parse_args()
    return args

def main(args: argparse.Namespace) -> dict:
    print(args)
    config = json.load(open(args.config_path, "r"))
    print("Starting grid search with the following config:")
    print(config)

    # From config, generate a list of experiments to run
    experiments = get_experiment_list(config)
    random.shuffle(experiments)

    job_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    for exper in experiments:
        job_queue.put(exper)

    print("Launching dispatcher with {} experiments and {} workers".format(len(experiments), args.num_workers))

    # Define worker fn to launch an experiment as a separate process.
    for _ in range(args.num_workers):
        multiprocessing.Process(target=worker, args=(args, job_queue, done_queue)).start()

    # Accumulate results into a list of dicts
    grid_search_results = []
    for _ in range(len(experiments)):
        grid_search_results.append(done_queue.get())

    keys = grid_search_results[0].keys()

    print("Saving results to {}".format(args.grid_search_results_path))

    writer = DictWriter(open(args.grid_search_results_path, 'w'), keys)
    writer.writeheader()
    writer.writerows(grid_search_results)

    print("Done")

if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)