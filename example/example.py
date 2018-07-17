import time

import pandas as pd

from engine import map_jobs, map_reduce_jobs, print_progress
from example.indicators import ppsr, on_balance_volume, simple_moving_average

current_milli_time = lambda: int(round(time.time() * 1000))


def main():
    df = pd.read_csv('dataframe.csv')
    jobs = [df.copy() for _ in range(1, 50)]

    i = 0
    for job in jobs:
        job['Symbol'] = str(i)
        i += 1

    print('start_map_jobs')
    measure_jobs_callback(start_map_jobs, jobs.copy())
    print('start_map_reduce_jobs')
    measure_jobs_callback(start_map_reduce_jobs, jobs.copy())
    print('start synchronous jobs')
    measure_jobs_callback(start_synchronous_jobs, jobs.copy())


def start_map_reduce_jobs(jobs):
    indicators = map_reduce_jobs(
        report_progress=True,
        redux=dict.update,
        redux_in_place=True,
        func=handle_task_dict,
        molecules=('jobs', jobs),
        threads=8,
        batches=4
    )


def start_synchronous_jobs_dict(jobs):
    indicators = handle_task_dict(jobs)


def start_map_jobs(jobs):
    indicators = map_jobs(
        report_progress=True,
        func=handle_task,
        molecules=('jobs', jobs),
        threads=8,
        batches=4
    )


def start_synchronous_jobs(jobs):
    indicators = handle_task_sync(jobs)


def handle_task_sync(jobs):
    time0 = time.time()

    i = 1
    for job in jobs:
        ppsr(job)
        on_balance_volume(job, 5)
        simple_moving_average(job, 5)
        print_progress(i, len(jobs), time0, 'handle_task_sync')
        i += 1
    return jobs


def handle_task(jobs):
    for i in jobs:
        ppsr(i)
        on_balance_volume(i, 5)
        simple_moving_average(i, 5)
    return jobs


def handle_task_dict(jobs):
    results = {}
    for job in jobs:
        ppsr(job)
        on_balance_volume(job, 5)
        simple_moving_average(job, 5)
        symbol = job['Symbol'].iloc(0)
        results[symbol] = job
    return results


def measure_jobs_callback(callback, jobs):
    start = current_milli_time()
    callback(jobs)
    end = current_milli_time()
    print(end - start)


if __name__ == '__main__':
    main()
