from setuptools import setup

setup(
    url="http://github.com/xingling28/opennsfw2",
    name="opennsfw2",
    author="Developers",
    version="0.0.1",
    zip_safe=False,
    packages=["opennsfw2", "bot"],
    description="bhky/opennsfw2 fork for detection of potential pornographic content on images and videos",
    author_email="developers@devops.com.br",
    entry_points={
        "console_scripts": ["bot = bot.__main__:main"],
    },
)
