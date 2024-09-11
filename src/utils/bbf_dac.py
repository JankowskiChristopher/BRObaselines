from typing import Tuple


def bbf_sac_args_to_name(size_tuple: Tuple[int, int]) -> str:
    bbf_sac_size = {
        (1, 128): 'xs',
        (1, 256): 's',
        (1, 512): 'm',
        (2, 512): 'l',
        (3, 1024): 'xl',
    }
    if size_tuple in bbf_sac_size:
        return bbf_sac_size[size_tuple]
    raise ValueError(f"Size tuple {size_tuple} not found in bbf_dac_size")
