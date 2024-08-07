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
        "transformers=4.40.2",
        "torch=2.3.0",
        "accelerate=0.30.1",
        "openai=1.30.1", 
        "bitsandbytes=0.43.1",
        "sentencepiece=0.2.0",
        "autoawq=0.2.5"
    ]
)