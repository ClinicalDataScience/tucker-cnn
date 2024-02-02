from abc import ABC, abstractmethod
import time

import numpy as np
import torch

from tuckercnn.utils import eprint


class Timer:
    execution_times = []
    verbose = True
    warm_up_rounds = 1
    use_cuda = True

    def __init__(self):
        self.clock: BaseClock = CUDAClock() if Timer.use_cuda else CPUClock()

    def __enter__(self):
        self.clock.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        ms = self.clock.stop()

        if Timer.verbose:
            eprint(f'({str(self.clock)}) Elapsed time: {ms:7.2f}ms')

        Timer.execution_times.append(ms)

    @classmethod
    def get_exec_times(cls):
        return cls.execution_times[cls.warm_up_rounds :]

    @classmethod
    def report(cls) -> None:
        if len(cls.execution_times) == 0:
            print('Nothing to report. No times were recorded.')
            return

        mean = float(np.mean(cls.get_exec_times()))
        std = float(np.std(cls.get_exec_times()))

        print(
            f'Execution took {mean:.2f}±{std:.2f}ms over '
            f'{len(cls.execution_times) - cls.warm_up_rounds} function calls.'
        )

        if cls.warm_up_rounds == 1:
            print('1 function call was ignored in this report due to warm up.')
        elif cls.warm_up_rounds > 1:
            print(
                f'{cls.warm_up_rounds} function calls were ignored in '
                f'this report due to warm up.'
            )

    @classmethod
    def reset(cls) -> None:
        cls.execution_times.clear()


class BaseClock(ABC):
    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class CUDAClock(BaseClock):
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self.start_event.record()

    def stop(self) -> float:
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)

    def __str__(self) -> str:
        return 'CUDA'


class CPUClock(BaseClock):
    def __init__(self):
        self.start_time = 0

    def start(self) -> None:
        self.start_time = time.time()

    def stop(self) -> float:
        return (time.time() - self.start_time) * 1000

    def __str__(self) -> str:
        return 'CPU'
