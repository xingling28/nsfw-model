import sys
from os.path import abspath, dirname, join

import fire

ROOT_DIR = join(dirname(abspath(__file__)), "..")

sys.path.append(ROOT_DIR)

from bot.init_task import init_task  # noqa
from bot.time_task import time_task  # noqa

commands = {
    "time": time_task,
    "init": init_task,
}


def main():
    return fire.Fire(commands)


if __name__ == "__main__":
    main()
