from setuptools import setup

with open("requirements.txt") as fid:
    requirements = [l.strip() for l in fid.readlines() if l.strip()]

setup(
    name="gabliteration",
    version="0.1.0",
    author="Gökdeniz Gülmez",
    description="Automated Gabliteration Optimizer",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://arxiv.org/abs/2512.18901",
    packages=["gabliteration"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "gabliterate=gabliteration.automated_gabliteration:main",
        ],
    }
)