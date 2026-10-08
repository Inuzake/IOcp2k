from setuptools import setup, find_packages

setup(name="IOcp2k",
      version="0.1.0",
      author="Antonio Sicilinao",
      author_email="antonio.siciliano@ens.psl.eu",
      description="IO for cp2k MD",
      packages = ["AtomicSnap"],
      package_dir = {"AtomicSnap": "Modules"},
      #long_description=open("README.md").read(),
      #ilong_description_content_type="text/markdown",
      url="https://github.com/AntonioSiciliano/IOcp2k.git",
      # packages=find_packages(),
      classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
      ],
      python_requires = '>=3.6',
      install_requires = ["numpy", "scipy", "ase", "MDAnalysis"])
