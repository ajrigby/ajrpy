from setuptools import setup, find_packages
from pathlib import Path                                     

this_directory = Path(__file__).parent                       

setup(
    name="ajrpy",
    version='0.0.1',
    author="Andrew Rigby",
    author_email="<ajrigby.astro@gmail.com>",
    description='A basic package containing useful tools for astronomy',
    long_description=(this_directory / "README.md").read_text(),
    long_description_content_type='text/markdown',
    license='MIT',
    packages=find_packages(['ajrpy', 'ajrpy.*']),
    install_requires=['astropy', 'matplotlib', 'numpy', 'scipy', 'tqdm'],
)
