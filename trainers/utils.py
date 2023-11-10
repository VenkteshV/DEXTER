

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from data.datastructures.hyperparameters.dpr import DprHyperParams


def get_optimizer(config: DprHyperParams, num_instances, model):
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, eps=config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=(config.warmup*num_instances) // config.batch_size,
            num_training_steps=num_instances // config.batch_size
        )
    return optimizer, scheduler