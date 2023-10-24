from loguru import logger
import subprocess
import time
import queue
import argparse

def read_commands_from_file(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Initialize the logger
logger.add("task_runner.log", rotation="1 day", level="INFO") 

# Argument parser
parser = argparse.ArgumentParser(description='Run multiple commands in parallel.')
parser.add_argument('-f', '--file', type=str, help='File containing commands to run')
parser.add_argument('-ng', '--num_gpus', type=int, default=4, help='Number of GPUs available')
args = parser.parse_args()

# Define number of GPUs and initialize a queue to hold available GPU IDs
NUM_GPUS = args.num_gpus
available_gpus = queue.Queue()
for i in range(NUM_GPUS):
    available_gpus.put(i)

# Read commands from file
commands = read_commands_from_file(args.file)

# Define how many tasks should run concurrently
CONCURRENT_TASKS = NUM_GPUS

running_tasks = []

while commands or running_tasks:
    while len(running_tasks) < CONCURRENT_TASKS and commands:
        next_command = commands.pop(0)
        next_gpu = available_gpus.get()
        p = subprocess.Popen(next_command + f" -gpu {next_gpu}", shell=True)
        running_tasks.append((p, next_gpu))
        logger.info(f"Started task {next_command} on GPU {next_gpu}")

    for p, gpu_id in running_tasks.copy():
        if p.poll() is not None:  # Task has finished
            running_tasks.remove((p, gpu_id))
            available_gpus.put(gpu_id)
            logger.info(f"Completed task on GPU {gpu_id}")

    time.sleep(1)