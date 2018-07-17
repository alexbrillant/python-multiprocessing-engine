import copy
import copyreg
import datetime as dt
import multiprocessing as mp
import sys
import time
import types

import pandas as pd


def _pickle_method(method):
    """
    Pickle methods in order to assign them to different
    processors using multiprocessing module. It tells the engine how
    to pickle methods.
    :param method: method to be pickled
    """
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class

    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    """
    Unpickle methods in order to assign them to different
    processors using multiprocessing module. It tells the engine how
    to unpickle methods.
    :param func_name: func name to unpickle
    :param obj: pickled object
    :param cls: class method
    :return: unpickled function
    """
    func = None
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get(obj, cls)


copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)


def map_reduce_jobs(func, molecules, threads=24, batches=1, linear_molecules=True, redux=None,
                    redux_args={}, redux_in_place=False, report_progress=False, **kargs):
    """
    Parallelize jobs and combine them into a single output
    :param func: function to be parallelized
    :param molecules[0]: Name of argument used to pass the molecule
    :param molecules[1]: List of atoms that will be grouped into molecules
    :param threads: number of threads
    :param batches: number of parallel batches (jobs per core)
    :param linear_molecules: Whether partition will be linear or double-nested
    :param redux: callabck to the function that carries out the reduction.
    :param redux_args: this is a dictionnary that contains the keyword arguments that must
    :param redux_in_place: a boolean, indicating wether the redux operation should happen in-place or not.
    For example, redux=dict.update and redux=list.append require redux_in_place=True,
    since appending a list and updating a dictionnary are both in place operations.
    :param kargs: any other argument needed by func
    :param report_progress: Whether progressed will be logged or not
    :return results combined into a single output
    """
    parts = __create_parts(batches, linear_molecules, molecules, threads)
    jobs = __create_jobs(func, kargs, molecules, parts)
    out = __process_jobs_redux(jobs, redux=redux, redux_args=redux_args, redux_in_place=redux_in_place, threads=threads,
                               report_progress=report_progress)

    return out


def map_jobs(func, molecules, threads=24, batches=1, linear_molecules=True, report_progress=False,
             **kargs):
    """
    Parallelize jobs, return a DataFrame or Series
    :param func: function to be parallelized
    :param molecules: pandas object
    :param molecules[0]: Name of argument used to pass the molecule
    :param molecules[1]: List of atoms that will be grouped into molecules
    :param threads: number of threads that will be used in parallel (one processor per thread)
    :param batches: number of parallel batches (jobs per core)
    :param linear_molecules: whether partition will be linear or double-nested
    :param report_progress: whether progressed will be logged or not
    :param kargs: any other argument needed by func
    """
    parts = __create_parts(batches, linear_molecules, molecules, threads)
    jobs = __create_jobs(func, kargs, molecules, parts)
    out = __process_jobs(jobs, threads, report_progress)

    return __create_output(out)


def __create_parts(batches, linear_molecules, molecules, threads):
    """
    Create partitions of atoms to be executed on each processor
    :param batches: number of parallel batches (jobs per core)
    :param linear_molecules: Whether partition will be linear or double-nested
    :param molecules: pandas object
    :param threads: number of threads that will be used in parallel (one processor per thread)
    :return: partitions array
    """
    if linear_molecules:
        return __linear_parts(len(molecules[1]), threads * batches)
    else:
        return __nested_parts(len(molecules[1]), threads * batches)


def __create_output(out):
    """
    Create DataFrame or Series output if needed
    :param out: result array
    :return: return the result as a DataFrame or Series if needed
    """
    import pandas as pd
    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series()
    else:
        return out
    for i in out:
        df0 = df0.append(i)
    return df0.sort_index()


def __process_jobs(jobs, threads, report_progress):
    """
    Process jobs
    :param jobs: jobs to process
    :param threads: number of threads that will be used in parallel (one processor per thread)
    :param report_progress: Whether progressed will be logged or not
    :return: result output
    """
    if threads == 1:
        out = __process_jobs_sequentially_for_debugging(jobs)
    else:
        out = __process_jobs_in_parallel(jobs=jobs, threads=threads, report_progress=report_progress)
    return out


def __create_jobs(func, kargs, molecules, parts):
    """
    Create jobs
    :param func: function to be executed
    :param kargs: any other argument needed by the function
    :param parts: partitionned list of atoms to be passed to the function
    """
    jobs = []
    for i in range(1, len(parts)):
        job = {molecules[0]: molecules[1][parts[i - 1]: parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)
    return jobs


def __process_jobs_in_parallel(jobs, task=None, threads=24, report_progress=False):
    """
    Process jobs with a multiprocess Pool
    :param jobs: jobs to be processed (data to be passed to task)
    :param task: func to be executed for each jobs
    :param threads: number of threads to create
    :param report_progress: Whether progressed will be logged or not
    """
    if task is None:
        task = jobs[0]['func'].__name__

    pool = mp.Pool(processes=threads)
    outputs, out, time0 = pool.imap_unordered(__expand_call, jobs), [], time.time()

    __map_outputs(jobs, out, outputs, task, time0, report_progress)

    pool.close()
    pool.join()

    return out


def __map_outputs(jobs, out, outputs, task, time0, report_progress):
    """
    Map outputs
    :param jobs: jobs to be processed (data to be passed to task)
    :param out: single output
    :param outputs: outputs
    :param task: task
    :param time0: start time
    :param report_progress: Whether progressed will be logged or not
    """
    for i, out_ in enumerate(outputs, 1):
        out.append(out_)
        if report_progress:
            print_progress(i, len(jobs), time0, task)


def __process_jobs_redux(jobs, task=None, threads=24, redux=None, redux_args={}, redux_in_place=False,
                         report_progress=False):
    """
    Process jobs and combine them into a single output(redux),
    :param jobs: jobs to run in parallel
    :param task: current task
    :param threads: number of threads
    :param redux: callabck to the function that carries out the reduction.
    :param redux_args: this is a dictionnary that contains the keyword arguments that must
    be passed to redux (if any).
    :param redux_in_place: a boolean, indicating wether the redux operation should happen in-place or not.
    For example, redux=dict.update and redux=list.append require redux_in_place=True,
    since appending a list and updating a dictionnary are both in place operations.
    :param report_progress: Whether progressed will be logged or not
    :return: job result array
    """
    if task is None:
        task = jobs[0]['func'].__name__

    pool = mp.Pool(processes=threads)
    imap = pool.imap_unordered(__expand_call, jobs)
    out = None
    if out is None and redux is None:
        redux = list.append
        redux_in_place = True

    time0 = time.time()

    out = __map_reduce_outputs(imap, jobs, out, redux, redux_args, redux_in_place, task, time0, report_progress)

    pool.close()
    pool.join()

    if isinstance(out, (pd.Series, pd.DataFrame)):
        out = out.sort_index()

    return out


def __map_reduce_outputs(imap, jobs, out, redux, redux_args, redux_in_place, task, time0, report_progress):
    """
    Map reduce outputs
    :param imap: job output iterator
    :param jobs: jobs to run in parallel
    :param out: output
    :param redux: callabck to the function that carries out the reduction.
    :param redux_args: this is a dictionnary that contains the keyword arguments that must
    :param redux_in_place: a boolean, indicating whether the redux operation should happen in-place or not.
    :param task: task to be executed
    :param time0: start time
    :param report_progress: Whether progressed will be logged or not
    :return:
    """
    for i, out_ in enumerate(imap, 1):
        out = __reduce_output(out, out_, redux, redux_args, redux_in_place)
        if report_progress:
            print_progress(i, len(jobs), time0, task)
    return out


def __reduce_output(out, out_, redux, redux_args, redux_in_place):
    """
    Reduce output into a single output with the redux function
    :param out: output
    :param out_: current output
    :param redux: callabck to the function that carries out the reduction.
    :param redux_args: this is a dictionnary that contains the keyword arguments that must
    :param redux_in_place: a boolean, indicating whether the redux operation should happen in-place or not.
    For example, redux=dict.update and redux=list.append require redux_in_place=True,
    since appending a list and updating a dictionnary are both in place operations.
    :return:
    """
    if out is None:
        if redux is None:
            out = [out_]
        else:
            out = copy.deepcopy(out_)
    else:
        if redux_in_place:
            redux(out, out_, **redux_args)
        else:
            out = redux(out, out_, **redux_args)
    return out


def print_progress(job_number, job_len, time0, task):
    """
    Report jobs progress
    :param job_number: job index
    :param job_len: number of jobs
    :param time0: multiprocessing start timestamp
    :param task: task to process
    """
    percentage = float(job_number) / job_len
    minutes = (time.time() - time0) / 60.
    minutes_remaining = minutes * (1 / percentage - 1)
    msg = [percentage, minutes, minutes_remaining]
    timestamp = str(dt.datetime.fromtimestamp(time.time()))

    msg = timestamp + ' ' + str(round(msg[0] * 100, 2)) + '% ' + task + ' done after ' + \
          str(round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'

    if job_number < job_len:
        sys.stderr.write(msg + '\r')
    else:
        sys.stderr.write(msg + '\n')

    return


def __process_jobs_sequentially_for_debugging(jobs):
    """
    Simple function that processes jobs sequentially for debugging
    :param jobs: jobs to process
    :return: result array of jobs
    """
    out = []
    for job in jobs:
        out_ = __expand_call(job)
        out.append(out_)
    return out


def __expand_call(kargs):
    """
    Pass the job (molecule) to the callback function
    Expand the arguments of a callback function, kargs['func']
    :param kargs: argument needed by callback func
    """
    func = kargs['func']
    del kargs['func']
    out = func(**kargs)
    return out


def __linear_parts(number_of_atoms, number_of_threads):
    """
    Partition a list of atoms in subset of equal size between the number of processors and the number of atoms.
    :param number_of_atoms: number of atoms (individual tasks to execute and group into molecules)
    :param number_of_threads: number of threads to create
    :return: return partitions or list of list of atoms (molecules)
    """
    parts = pd.np.linspace(0, number_of_atoms, min(number_of_threads, number_of_atoms) + 1)
    parts = pd.np.ceil(parts).astype(int)

    return parts


def __nested_parts(number_of_atoms, number_of_threads, upper_triangle=False):
    """
    Partition of atoms with an inner loop
    :param number_of_atoms: number of atoms (individual tasks to execute and group into molecules)
    :param number_of_threads: number of threads to create
    :param upper_triangle:
    :return: return partitions or list of list of atoms (molecules)
    """
    parts = [0]
    number_of_threads_ = min(number_of_threads, number_of_atoms)

    for num in range(number_of_threads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + number_of_atoms * (number_of_atoms + 1.) / number_of_threads_)
        part = (-1 + part ** .5) / 2.
        parts.append(part)

    parts = pd.np.round(parts).astype(int)
    if upper_triangle:
        parts = pd.np.cumsum(pd.np.diff(parts)[::-1])
        parts = pd.np.append(pd.np.array([0]), parts)

    return parts
