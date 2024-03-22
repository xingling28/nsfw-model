from freezegun import freeze_time

from bot.time_task import time_task


@freeze_time("2020-01-01")
def test_time_task():
    assert time_task() == "Data atual: 01/01/2020"
