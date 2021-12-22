from setuptools import setup, find_packages, Extension

setup(
    name='kmeans_pp',
    version=1,
    author="Assaf Harel & Maya Baruch",
    author_email="",
    description="A simple K-Means algorithm",
    install_requires=['invoke'],
    packages=find_packages(),
    license="BSD",
    ext_modules=[
        Extension("mykmeanssp", ["kmeans.c"])
    ]
)