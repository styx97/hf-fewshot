from setuptools import setup, find_packages

setup(
    name='hf_fewshot',
    version='0.1.3',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'hf_fewshot = hf_fewshot.classifiers:main',
            'hf_chat = hf_fewshot.agents:main',
            'hf_pairwise = hf_pairwise.get_bt_scores:main',
        ],
    },
    install_requires=[
        "transformers>=4.51.3",
        "torch>=2.3.0",
        "optimum>=1.22.0",
        "accelerate>=0.30.1",
        "openai>=1.30.1",
        "bitsandbytes>=0.43.1",
        "sentencepiece>=0.2.0",
        "autoawq>=0.2.5", 
        "pynvml>=11.5.3", 
        "python-dotenv>=1.0.1"
    ]
)