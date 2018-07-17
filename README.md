# python-multiprocessing-engine

## map_jobs

Parallelize jobs, return a DataFrame or Series
```python
indicators = map_jobs(
    func=handle_task,
    molecules=('jobs', jobs),
    threads=8,
    batches=4
)
```

## map_reduce_jobs

Parallelize jobs and combine them into a single output

```python
indicators = map_reduce_jobs(
    redux=dict.update,
    redux_in_place=True,
    func=handle_task_dict,
    molecules=('jobs', jobs),
    threads=8,
    batches=4
)
```

