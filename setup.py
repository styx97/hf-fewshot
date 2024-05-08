from setuptools import setup, find_packages

setup(
    name='hf_fewshot',
    version='0.1.2',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'hf_fewshot = hf_fewshot.classifiers:main',
        ],
    }, 
    install_requires=[
        "transformers",
        "torch",
        "accelerate",
        "openai", 
        "bitsandbytes",
    ]
)