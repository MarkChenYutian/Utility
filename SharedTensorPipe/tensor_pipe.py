import torch
import typing as T
import multiprocessing as mp
from contextlib import contextmanager
from multiprocessing.connection import PipeConnection
from dataclasses import dataclass, is_dataclass, fields, replace

if T.TYPE_CHECKING:
    from _typeshed import DataclassInstance
    T_Dataclass = DataclassInstance
else:
    T_Dataclass = object



@dataclass(kw_only=True)
class TensorMetaData:
    shape: torch.Size
    dtype: torch.dtype


def dump_tensor(tensor: torch.Tensor) -> tuple[TensorMetaData, torch.IntTensor]:
    data_tensor = tensor.cpu()
    return (
        TensorMetaData(shape=data_tensor.shape, dtype=data_tensor.dtype),
        data_tensor.flatten().view(torch.uint8)
    )


def load_tensor(metadata: TensorMetaData, data: torch.IntTensor) -> torch.Tensor:
    return data.view(metadata.dtype).reshape(metadata.shape)


def dehydrate(data: T.Any) -> tuple[T.Any, list[torch.Tensor]]:
    """Given a dataclass, walk through the dataclass and dehydrate all tensors"""
    if   isinstance(data, torch.Tensor):
        metadata, buffer = dump_tensor(data)
        return metadata, [buffer]
    
    elif isinstance(data, list):
        results = [dehydrate(d) for d in data]
        return (
            [r for r, _ in results],
            [buf for _, blist in results for buf in blist]
        )
    
    elif isinstance(data, tuple):
        results = [dehydrate(d) for d in data]
        return (
            tuple(r for r, _ in results),
            [buf for _, blist in results for buf in blist]
        )
    
    elif isinstance(data, dict):
        results = { k : dehydrate(v) for k, v in data }
        return (
            { k : v_stub for k, (v_stub, _) in results.items() },
            [buf for _, blist in results.values() for buf in blist]
        )

    elif is_dataclass(data) and (not isinstance(data, type)):
        replaced_fields = {}
        buffers         = []
        
        for f in fields(data):
            metadata, new_bufs      = dehydrate(getattr(data, f.name))
            replaced_fields[f.name] = metadata
            buffers.extend(new_bufs)
        
        return replace(data, **replaced_fields), buffers

    else:
        return data, []


def hydrate(data: T.Any, buffer: list[torch.Tensor], cur_idx: int = 0) -> tuple[T.Any, int]:
    """Hydrate the data based on dehydrate method"""
    if   isinstance(data, TensorMetaData):
        return load_tensor(data, buffer[cur_idx]), cur_idx + 1
    
    elif isinstance(data, list):
        result = []
        for stub in data:
            item, cur_idx = hydrate(stub, buffer, cur_idx)
            result.append(item)
        return result, cur_idx
    
    elif isinstance(data, tuple):
        result = []
        for stub in data:
            item, cur_idx = hydrate(stub, buffer, cur_idx)
            result.append(item)
        return tuple(result), cur_idx
    
    elif isinstance(data, dict):
        result = dict()
        for key, stub in data.items():
            item, cur_idx = hydrate(stub, buffer, cur_idx)
            result[key] = item
        return result, cur_idx
    
    elif is_dataclass(data) and (not isinstance(data, type)):
        replaced_fields = {}
        
        for f in fields(data):
            item, cur_idx = hydrate(getattr(data, f.name), buffer, cur_idx)
            replaced_fields[f.name] = item

        return replace(data, **replaced_fields), cur_idx
    
    else:
        return data, cur_idx


def prologue_comm(data: T.Any, buffer: torch.Tensor | None) -> tuple[tuple[T.Any, list[int]], torch.Tensor]:
    stub, buffers = dehydrate(data)
    offset        = [0]
    for b in buffers: offset.append(offset[-1] + b.numel())
    
    if buffer is None:
        return (stub, offset), torch.cat(buffers, dim=0)
    else:
        num_send = offset[-1]
        num_buf  = buffer.numel()
        
        if num_send > num_buf:
            raise RuntimeError(f"Trying to send a data pack larger than the pre-defined maximum size ({num_send=} > {num_buf=}).")
        
        # NOTE: this is optional, for debugging use only
        # buffer.narrow(dim=0, start=num_send, length=num_buf-num_send).fill_(255) 
        write_idx = 0
        for buf in buffers:
            step = buf.numel()
            buffer.narrow(dim=0, start=write_idx, length=step).copy_(buf)
            write_idx += step
        
        offset.append(num_buf)
        return (stub, offset), buffer


def epilogue_comm(data: tuple[T.Any, list[int]], buffer: torch.Tensor) -> T.Any:
    stub, offset  = data
    packed_buffer = torch.nested.nested_tensor_from_jagged(values=buffer, offsets=torch.tensor(offset, dtype=torch.long))
    
    orig_data, _     = hydrate(stub, [buf for buf in packed_buffer])
    return orig_data


T_Datatype = T.TypeVar("T_Datatype")


class SharedTensorSend(T.Generic[T_Datatype]):
    def __init__(self, pipe: PipeConnection, max_data_example: T_Datatype):
        self.free_queue  = pipe
        
        # First send, will create shared memory buffer.
        (stub, offset), packed_buffer = prologue_comm(max_data_example, buffer=None)
        
        self.free_queue.send(((stub, offset), packed_buffer))

    def send(self, data: T_Datatype, blocking: bool=True) -> bool:
        if not blocking:
            if not self.free_queue.poll(): return False
        
        free_buffer       = self.free_queue.recv()
        (stub, offset), _ = prologue_comm(data, free_buffer)
        
        self.free_queue.send(((stub, offset), free_buffer))
        return True
        

class SharedTensorRecv(T.Generic[T_Datatype]):
    def __init__(self, pipe: PipeConnection):
        self.data_queue = pipe
        
        # Receive the initial connection from Sender.
        (stub, offset), buffer = self.data_queue.recv()
        
        # Send back the buffer for first "real" communication.
        self.data_queue.send(buffer)
    
    @contextmanager
    def recv(self) -> T.Generator[T_Datatype, None, None]:
        (stub, offset), buffer = self.data_queue.recv()
        
        try:
            data = epilogue_comm((stub, offset), buffer)
            yield data
        except Exception as e:
            raise e from None
        finally:
            # Return the buffer to sender for next pass
            self.data_queue.send(buffer)
    
    def try_recv(self) -> T_Datatype | None:
        if not self.data_queue.poll(): return None
        
        (stub, offset), buffer = self.data_queue.recv()
        local_buffer           = buffer.clone()
        self.data_queue.send(buffer)
        
        return epilogue_comm((stub, offset), local_buffer)


def SharedTensorPipe(max_data_pack: T_Datatype) -> tuple[SharedTensorSend, SharedTensorRecv]:
    pipe_send, pipe_recv = mp.Pipe(duplex=True)
    print(f"Initializing shared tensor buffer for {type(max_data_pack)}...", end="", flush=True)
    
    sender = SharedTensorSend(pipe_send, max_data_pack)
    recver = SharedTensorRecv(pipe_recv)
    
    print("Done.")
    return sender, recver


__all__ = ["SharedTensorPipe", "SharedTensorSend", "SharedTensorRecv"]
