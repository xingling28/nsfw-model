from datetime import date


def time_task(*args, **kwargs) -> str:
    return "Data atual: %s" % date.today().strftime("%d/%m/%Y")
