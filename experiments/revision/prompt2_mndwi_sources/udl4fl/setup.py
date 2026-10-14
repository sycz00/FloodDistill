from setuptools import setup, find_packages
import codecs
import os.path

with open("README.md", "r") as fh:
    long_description = fh.read()


def parse_requirements_file(filename):
    with open(filename, encoding="utf-8") as fid:
        requires = [l.strip() for l in fid.readlines() if l]
    return requires


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(name="udl4fl",
      version=get_version("udl4fl/__init__.py"),
      author="Enrique Portalés-Julià, Gonzalo Mateo-García, Luis Gómez-Chova",
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=find_packages(".", exclude=["tests"]),
      description="Lightweight code to run inference with the flood segmentation models proposed in \
          in the paper \"Understanding flood detection models across Sentinel-1 and Sentinel-2 modalities and benchmark datasets\"",
      install_requires=parse_requirements_file("requirements.txt"),
      url="https://github.com/kipoju/udl4fl",
      classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: GIS",
        "Development Status :: 5 - Production/Stable", 
    ],
      keywords=["raster reading", "rasterio"],
)
