import os
import re
import codecs
from setuptools import setup, find_packages

current_path = os.path.abspath(os.path.dirname(__file__))


def read_file(*parts):
    with codecs.open(os.path.join(current_path, *parts), 'r', 'utf8') as reader:
        return reader.read()


def get_requirements(*parts):
    with codecs.open(os.path.join(current_path, *parts), 'r', 'utf8') as reader:
        return list(map(lambda x: x.strip(), reader.readlines()))


def find_version(*file_paths):
    version_file = read_file(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')


setup(
    name='torch-same-pad',
    version=find_version('torch_same_pad', '__init__.py'),
    packages=find_packages(),
    url='https://github.com/CyberZHG/torch-same-pad',
    license='MIT',
    author='CyberZHG',
    author_email='CyberZHG@users.noreply.github.com',
    description='Paddings used for converting TensorFlow conv/pool layers to PyTorch.',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    install_requires=get_requirements('requirements.txt'),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)