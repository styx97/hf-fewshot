from setuptools import setup, find_packages

setup(
    name='hf_fewshot',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
    ]
)