from setuptools import setup, find_packages

setup(
    name='swe_wrapper',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here, e.g.
        'numpy',
        'dedalus',
    ],
    author='Lars Stietz',
    author_email='lars.stietz@tuhh.de',
    description='A project for solving Bayesian inverse problems',
    url='https://collaborating.tuhh.de/l_stz/bayesian-inverse-problems',
    python_requires='>=3.10'
)