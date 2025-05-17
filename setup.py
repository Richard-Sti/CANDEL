# Copyright (C) 2025 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from setuptools import setup, find_packages

setup(
    name="candel",
    version="0.1.0",
    author="Richard Stiskalek",
    author_email="richard.stiskalek@protonmail.com",
    description="Calibration And Normalization of the DistancE Ladder",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/richard-sti/candel",
    packages=find_packages(include=["my_package", "my_package.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.10",
    install_requires=[
        "jax",
        "jaxlib",
        "numpyro",
        "numpy",
        "scipy",
        "h5py",
        "tomli",
        "corner",
        "interpax",
        "astropy",
        "quadax",
        "matplotlib",
        "corner",
        "tomli_w",
        "scienceplots",
        "jax-cosmo",
    ],
    include_package_data=True,
    zip_safe=False,
)