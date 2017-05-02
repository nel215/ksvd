#!/usr/bin/env python
# coding: utf-8
from setuptools import setup


setup(
    name='ksvd',
    version='0.0.3',
    description='An K-SVD implementaion written in Python.',
    author='nel215',
    author_email='otomo.yuhei@gmail.com',
    url='https://github.com/nel215/ksvd',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    packages=['ksvd'],
    install_requires=[
        "numpy",
        "scikit-learn",
    ],
    keywords=['machine learning'],
)
