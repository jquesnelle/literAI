from setuptools import setup
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('literai/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name='literAI',
    packages=["literai"],
    version=main_ns['__version__'],
    description='Generate visual podcasts about novels using open source models',
    author='Jeffrey Quesnelle',
    author_email='jq@jeffq.com',
    url='https://github.com/jquesnelle/literAI/',
    license='MIT',
    install_requires=[
        'transformers',
        'accelerate',
        'torch',
        'protobuf<=3.19.6',
        'tqdm',
        'diffusers'
    ],
    extras_require = {
        'gcloud': ["google-cloud-storage"]
    },
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