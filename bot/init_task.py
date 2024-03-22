from os import rename
from os.path import join
from string import Template


def init_task():
    """Initialize the repository for a new package.

    This task will rename the package folder and the tests folder to the
    package name. It will also update the setup.py file with the package name
    and description.
    """
    try:
        from package import ROOT_DIR  # type: ignore
    except ModuleNotFoundError:
        print("Package already initialized")
        return

    package = input("Package name: ")
    description = input("Package description: ")

    data = {
        "package": package,
        "description": description,
    }

    with open(join(ROOT_DIR, "setup.py")) as f:
        setup = f.read()

    with open(join(ROOT_DIR, "setup.py"), "w") as f:
        f.write(Template(setup).substitute(data))

    with open(join(ROOT_DIR, "README.md"), "w") as f:
        with open(join(ROOT_DIR, "README.md.template")) as template:
            f.write(Template(template.read()).substitute(data))

    rename(join(ROOT_DIR, "tests/package"), join(ROOT_DIR, f"tests/{package}"))
    rename(join(ROOT_DIR, "package"), join(ROOT_DIR, package))

    print("Package initialized!")
