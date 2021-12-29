#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Manuel Vimercati",
    author_email='manuel.vimercati@unimib.it',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A framework to perform entity typing task in different setup and with different networks",
    entry_points={
        'console_scripts': [
            'entity_typing_framework=entity_typing_framework.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='entity_typing_framework',
    name='entity_typing_framework',
    packages=find_packages(include=['entity_typing_framework', 'entity_typing_framework.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/NooneBug/entity_typing_framework',
    version='0.1.0',
    zip_safe=False,
)
