import sys
try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="molpy",
    version="0.0.1",
    author="Roy Kid",
    author_email="lijichen365@126.com",
    license="MIT",
    description="molecule data structure",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Roy-Kid/molpy",
    project_urls={
        "Bug Tracker": "https://github.com/Roy-Kid/molpy/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3-Clause License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(  # find packages in molpy/molpy
        where='.',
        exclude=('tests*', 'docs*', 'examples*')
    ),
    # package_dir={"": "molpy"},  # indicate all the code under molpy/molpy
    python_requires=">=3.8",
    # extras_require={"test": ["pytest"]},
    include_package_data=True,
    # cmake_install_dir="molpy/cpp",
)

