from setuptools import setup

setup(
    name='liter-AI',
    packages=["literai"],
    version='1.0.0',
    description='Generate visual podcasts using open source models',
    author='Jeffrey Quesnelle',
    author_email='jq@jeffq.com',
    url='https://github.com/jquesnelle/liter-AI/',
    license='MIT',
    install_requires=[
        'transformers',
        'accelerate',
        'torch',
        'protobuf==3.20.0',
        'tqdm',
        'diffusers'
    ],
    entry_points={
        'console_scripts': [
            'literai = literai.__main__:main',
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ]
)