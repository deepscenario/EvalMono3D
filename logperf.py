import itertools
from tabulate import tabulate
from termcolor import colored
from tabulate import tabulate
import logging

logger = logging.getLogger("default")


def print_ap_category_histogram(dataset, results):
    """
    Prints AP performance for each category.
    Args:
        results: dictionary; each entry contains information for a dataset
    """
    num_classes = len(results)
    N_COLS = 9
    data = list(itertools.chain(*[[
        cat,
        out["AP2D"],
        out["AP3D"],
    ] for cat, out in results.items()]))
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "AP2D", "AP3D"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    logger.info("Performance for each of {} categories on {}:\n".format(num_classes, dataset) + colored(table, "cyan"))
