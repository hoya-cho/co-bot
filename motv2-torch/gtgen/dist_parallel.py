import os
from time import sleep, time
import torch.multiprocessing as mp

from gtgen.gtgen_abstract import ProgressCallBackAbstract


def setup(master_addr="localhost", master_port=30100, world_size=1, rank=0):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    


def distribute_worker(
    is_train_phase: bool, task_worker, ngpus_per_node, args, progress_callback
):
    if progress_callback is None:
        print("### dist_parallel:  no progress_callback")
        mp.spawn(fn=task_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        return

    from multiprocessing import get_context

    py_mp = get_context("spawn")

    callback_stop_queues = []
    for i in range(ngpus_per_node):
        callback_queue = py_mp.SimpleQueue()
        stop_queue = py_mp.SimpleQueue()
        callback_stop_queues.append((callback_queue, stop_queue))

    args.callback_stop_queues = callback_stop_queues

    context = mp.spawn(
        fn=task_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args), join=False
    )

    print("### dist_parallel:  start receive progress callback from child process")

    recv_callback_indices = set(range(len(callback_stop_queues)))

    def child_to_main_callback():
        temp_callback_indices = recv_callback_indices.copy()
        for rank in temp_callback_indices:
            callback_queue = callback_stop_queues[rank][0]
            while not callback_queue.empty():
                callback_kwargs = callback_queue.get()
                print(f"callback from the {rank} training:\n{callback_kwargs}\n")
                if len(callback_kwargs) == 0:  # finished training
                    print(f"rank={rank} training is finished")
                    recv_callback_indices.remove(rank)
                    break

                if is_train_phase and rank > 0:
                    continue

                # when training, call main callback if rank is zero
                callback_kwargs.update({"rank": rank})
                progress_callback(**callback_kwargs)

    while len(recv_callback_indices) > 0:  # until all child processes are terminated
        child_to_main_callback()

        if progress_callback.is_stopping():
            print(
                "### dist_parallel:  distribute_worker. stop signal is recevied from main process"
            )
            break

        sleep(1)

    print("### dist_parallel:  distribute_worker. set stop all child processes")

    for _, stop_queue in callback_stop_queues:
        stop_queue.put(True)

    # when receive callback messages from child processes, send to callback of main process(this)
    # Loop on join until it returns True or raises an exception.
    print("### dist_parallel:  distribute_worker. join child process")
    while not context.join():
        pass

    print("### dist_parallel:  distribute_worker. finished all child process\n")


class DistributedChildProgressCallback(ProgressCallBackAbstract):
    def __init__(self, rank, args):
        self.send_callback_queue, self.recv_stop_queue = args.callback_stop_queues[rank]

    def __call__(self, **kwargs) -> None:
        print(f"DistributedChildProgressCallback: {kwargs}\n", flush=True)

        #self.send_callback_queue.put(kwargs)
        # print("DistributedChildProgressCallback: queue put.end\n")

    def is_stopping(self) -> bool:
        return not self.recv_stop_queue.empty()

    def end_progress(self) -> None:
        self.send_callback_queue.put({})


class SingleChildProgressCallback(ProgressCallBackAbstract):
    def __init__(self, progress_callback:ProgressCallBackAbstract) -> None:
        self.progress_callback = progress_callback

    def __call__(self, **kwargs) -> None:
        if self.progress_callback is not None:
            self.progress_callback(**kwargs)
        else:
            print("no callback.")

    def end_progress(self):
        pass

    def is_stopping(self):
        return False


def get_child_progress_callback(distributed, rank: int, args, progress_callback):
    if distributed and progress_callback is not None:
        progress_callback = DistributedChildProgressCallback(rank, args)
    else:
        progress_callback = SingleChildProgressCallback(progress_callback)

    return progress_callback