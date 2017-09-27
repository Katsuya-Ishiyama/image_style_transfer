# -*- coding: utf-8 -*-

from setuptools import (find_packages, setup)

REQUIRED_PACKAGES = [
    'Pillow>=4.2.1',
    'tensorflow==1.2.0'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Image Style Transfer package.'
)

