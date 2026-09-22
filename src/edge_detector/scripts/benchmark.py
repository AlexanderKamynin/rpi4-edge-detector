from dataclasses import asdict, dataclass
from statistics import fmean, median
from time import perf_counter_ns

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BenchmarkResult:
    params: int
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"params={self.params / 1e6:7.3f} M | "
            f"median={self.median_ms:8.3f} ms | "
            f"mean={self.mean_ms:8.3f} ms | "
            f"min={self.min_ms:8.3f} ms | "
            f"max={self.max_ms:8.3f} ms"
        )


def configure_cpu(num_threads: int = 4) -> None:
    torch.set_num_threads(num_threads)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def feature_shapes(
    model: nn.Module,
    x: Tensor,
) -> tuple[tuple[int, ...], ...]:
    model.eval()

    with torch.inference_mode():
        outputs = model(x)

    return tuple(tuple(output.shape) for output in outputs)


def benchmark_module(
    model: nn.Module,
    x: Tensor,
    warmup: int = 20,
    runs: int = 100,
) -> BenchmarkResult:
    model.eval()

    times = []

    with torch.inference_mode():
        for _ in range(warmup):
            model(x)

        for _ in range(runs):
            start = perf_counter_ns()
            model(x)
            end = perf_counter_ns()

            times.append((end - start) / 1e6)

    return BenchmarkResult(
        params=count_parameters(model),
        median_ms=median(times),
        mean_ms=fmean(times),
        min_ms=min(times),
        max_ms=max(times),
    )
