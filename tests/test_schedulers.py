from  unet import schedulers


class TestWarmupLinearDecaySchedule:

    def test_schedule(self):
        warmup_steps = 2
        total_steps = 10
        learning_rate = 10
        min_lr = 0.0
        scheduler = schedulers.WarmupLinearDecaySchedule(warmup_steps, total_steps, learning_rate, min_lr=min_lr)

        assert scheduler(step=0) == 0
        assert 0 < scheduler(step=warmup_steps - 1) < learning_rate
        assert scheduler(step=warmup_steps) == learning_rate

        assert min_lr < scheduler(step=warmup_steps + 1) < learning_rate
        assert min_lr < scheduler(step=total_steps - 1) < learning_rate
        assert scheduler(step=total_steps) == min_lr
        assert scheduler(step=total_steps + 1) == min_lr
