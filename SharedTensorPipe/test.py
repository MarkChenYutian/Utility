import time
import torch
import signal
import random
import multiprocessing as mp
from multiprocessing.connection import PipeConnection
from multiprocessing.synchronize import Event
from dataclasses import dataclass

from tqdm import tqdm

from .tensor_pipe import SharedTensorPipe, SharedTensorSend, SharedTensorRecv


try:
   mp.set_start_method('spawn', force=True)
   print("spawned")
except RuntimeError:
   pass


@dataclass
class SimPayload:
    create_time: float
    payload1   : torch.Tensor
    payload2   : list[torch.Tensor]


def ping_pong_benchmark_send(sender: SharedTensorSend[SimPayload], stop_event: Event, NUM_TENSOR: int):
    while not (stop_event.is_set()):
        length = random.randint(50, 2000)
        x = SimPayload(
            time.time(),
            torch.randn((length, 64), dtype=torch.float32),
            [torch.randn((length, 3, 3), dtype=torch.float32) for _ in range(NUM_TENSOR)]
        )
        sender.send(x, blocking=True)


def ping_pong_benchmark_recv(receiver: SharedTensorRecv[SimPayload], stop_event: Event):
    with tqdm(total=None, unit="msg", smoothing=0.05, desc="throughput") as bar:
        while not (stop_event.is_set()):
            with receiver.recv() as x: bar.update(1)
            # print(f"{time.time() - x.create_time : 3f}ms")


def benchmark_sharedtensor(NUM_TENSOR: int):
    maximum_payload = SimPayload(
        0.0,
        torch.empty((2000, 256), dtype=torch.float32),
        [torch.empty((2000, 3, 3), dtype=torch.float32) for _ in range(NUM_TENSOR)]
    )
    
    sender, receiver = SharedTensorPipe(maximum_payload)
    stop_event       = mp.Event()
    
    def _graceful_shutdown(signum, frame):
        print("\nCtrl-C detected — stopping …", flush=True)
        stop_event.set()          # tell children to exit
        producer.join(timeout=1)
        consumer.join(timeout=1)
        # if they failed to exit in time, kill them hard
        for p in (producer, consumer):
            if p.is_alive():
                p.terminate()
    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    
    producer = mp.Process(target=ping_pong_benchmark_send, args=(sender,   stop_event, NUM_TENSOR))
    consumer = mp.Process(target=ping_pong_benchmark_recv, args=(receiver, stop_event))
    
    producer.start()
    consumer.start()
    
    producer.join()
    consumer.join()


# Reference (slow but simple implementation) #


def ping_pong_ref_send(sender: PipeConnection, stop_event: Event, NUM_TENSOR: int):
    while not stop_event.is_set():
        length = random.randint(50, 2000)
        x = SimPayload(
            time.time(),
            torch.randn((length, 64), dtype=torch.float32),
            [torch.randn((length, 3, 3), dtype=torch.float32) for _ in range(NUM_TENSOR)]
        )
        sender.send(x)


def ping_pong_ref_recv(receiver: PipeConnection, stop_event: Event):
    with tqdm(total=None, unit="msg", smoothing=0.05, desc="throughput") as bar:
        while not stop_event.is_set():
            x = receiver.recv()
            bar.update(1)
        # print(f"{time.time() - x.create_time : 3f}ms")


def benchmark_ref(NUM_TENSOR: int):    
    sender, receiver = mp.Pipe(duplex=True)
    stop_event       = mp.Event()
    
    def _graceful_shutdown(signum, frame):
        print("\nCtrl-C detected — stopping …", flush=True)
        stop_event.set()          # tell children to exit
        producer.join(timeout=1)
        consumer.join(timeout=1)
        # if they failed to exit in time, kill them hard
        for p in (producer, consumer):
            if p.is_alive():
                p.terminate()
    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    
    producer = mp.Process(target=ping_pong_ref_send, args=(sender,   stop_event, NUM_TENSOR))
    consumer = mp.Process(target=ping_pong_ref_recv, args=(receiver, stop_event))
    
    producer.start()
    consumer.start()
    
    producer.join()
    consumer.join()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("test", type=str, choices=["exp", "ref"])
    parser.add_argument("--num_tensor", type=int, default=12)
    args   = parser.parse_args()
    
    
    match args.test:
        case "exp": benchmark_sharedtensor(args.num_tensor)
        case "ref": benchmark_ref(args.num_tensor)
