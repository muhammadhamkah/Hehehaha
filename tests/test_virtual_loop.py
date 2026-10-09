import asyncio
import time

from backtest.virtual_loop import LoopClock, VirtualTimeEventLoop


def test_virtual_time_runs_timers_without_real_waiting():
    loop = VirtualTimeEventLoop(start_ms=1_000_000)
    clock = LoopClock(loop)
    log = []

    async def ticker():
        for _ in range(5):
            await asyncio.sleep(0.25)
            log.append(("tick", clock.now_ms()))

    async def waiter():
        ev = asyncio.Event()
        try:
            await asyncio.wait_for(ev.wait(), 3600)      # one simulated hour
        except asyncio.TimeoutError:
            log.append(("timeout", clock.now_ms()))

    async def main():
        await asyncio.gather(ticker(), waiter())
        await loop.sleep_until_ms(1_000_000 + 7_200_000)
        log.append(("end", clock.now_ms()))

    t0 = time.perf_counter()
    loop.run_until_complete(main())
    loop.close()
    assert time.perf_counter() - t0 < 2.0
    assert log[:5] == [("tick", 1_000_000 + 250 * i) for i in range(1, 6)]
    assert log[5] == ("timeout", 1_000_000 + 3_600_000)
    assert log[6] == ("end", 1_000_000 + 7_200_000)


def test_speed_scaling_sleeps_real_time():
    loop = VirtualTimeEventLoop(start_ms=0, speed=10.0)
    t0 = time.perf_counter()
    loop.run_until_complete(asyncio.sleep(1.0))       # 1s simulated at 10x -> ~0.1s real
    loop.close()
    assert 0.08 < time.perf_counter() - t0 < 0.6


def test_epoch_scale_timestamps_do_not_livelock():
    # Regression: with epoch-based float time, a 1ms timer at t=1.7e12 ms never fired.
    loop = VirtualTimeEventLoop(start_ms=1_717_200_000_000)
    clock = LoopClock(loop)

    async def main():
        for _ in range(2000):
            await asyncio.sleep(0.001)
        return clock.now_ms()

    t0 = time.perf_counter()
    end = loop.run_until_complete(asyncio.wait_for(main(), 10))
    loop.close()
    assert end == 1_717_200_000_000 + 2000
    assert time.perf_counter() - t0 < 3
