#import setuptools
from __future__ import print_function
import numpy
# from numpy.distutils.core import setup, Extension
from setuptools import setup, find_packages

setup(name = "iocp2k",
      version = "0.1",
      description = "Read the cp2k snapshots",
      author = "Antonio Siciliano",
      packages = ["AtomicSnap"],
      package_dir = {"AtomicSnap": "Modules"},
      license = "GPLv3")


def readme():
    with open("README.md") as f:
        return f.read()
