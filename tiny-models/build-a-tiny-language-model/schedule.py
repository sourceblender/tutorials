"""The learning-rate schedule, shared by Day 3's short run and the full trainer.

Linear warm-up from near zero to the peak over the first `warmup` steps, then cosine decay from the peak down to
10% of it at the last step. Step `warmup` is exactly the peak; when total > warmup, the last step is exactly 10%.
If the run is no longer than the warm-up (total <= warmup), it is all warm-up and never decays.
"""
import math

SCHEDULE_ID = "linear-warmup-cosine-v2"  # stored in checkpoints, so a resume can't silently switch schedules


def lr_at(step, total, peak, warmup):
    if warmup > 0 and step <= warmup:
        return peak * step / warmup
    decay = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, decay))))
