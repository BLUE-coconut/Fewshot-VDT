import random
import mindspore
import mindspore.numpy as mnp
import numpy as np


def fix_randseed(seed):
    if seed is None:
        seed = int(random.random() * 1e5)

    random.seed(seed)

    np.random.seed(seed)

    mindspore.set_seed(seed)

    print(f"Random seed fixed to: {seed}")


def mean(x):

    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_device(batch):

    for key, value in batch.items():

        if isinstance(value, np.ndarray):
            batch[key] = mindspore.Tensor(value)
 
        elif isinstance(value, mindspore.Tensor):
            pass 
            
    return batch


def to_numpy(tensor):

    if isinstance(tensor, mindspore.Tensor):

        return tensor.asnumpy().copy()
    else:
        return tensor