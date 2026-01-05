from setuptools import setup

with open("requirements.txt") as fid:
    requirements = [l.strip() for l in fid.readlines() if l.strip()]

from gabliterate import __version__

setup(
    name="gabliteration",
    version=__version__,
    author="Gökdeniz Gülmez",
    description="Automated Gabliteration",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://arxiv.org/abs/2512.18901",
    packages=["gabliterate.automated"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "gabliterate=gabliterate.automated:main",
        ],
    }
)