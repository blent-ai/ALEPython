#!/usr/bin/env python

from setuptools import setup, find_packages

with open('README.md') as f:
	long_description = f.read()

with open('requirements.txt', 'r') as f:
	required = f.read().splitlines()

setup(name='alepython',
      version='0.1.0',
      description='Python Accumulated Local Effects (ALE) package.',
      author='Maxime Jumelle',
      author_email='maxime@blent.ai',
      license="Apache 2",
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/MaximeJumelle/alepython/',
      install_requires=required,
      keywords='alepython',
      packages=find_packages(include=['alepython', 'alepython.*']),
     )