import setuptools

with open("README.md", "rb") as fin:
    long_description = fin.read().decode("utf-8")

req = ["numpy",
       "scipy",
       "h5py",
       "opt_einsum"
       ]

setuptools.setup(
    name="renormalizer",
    version="0.0.5",
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shuaigroup/Renormalizer",
    python_requires='>=3.6',
    install_requires=req,
    license="Apache",
)


# How to publish the library to pypi
# python setup.py sdist
# twine upload -s dist/*
