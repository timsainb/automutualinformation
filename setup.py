from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="automutualinformation",
    packages=find_packages(),
    version="0.1.2",
    description="Auto Mutual Information (Sequential Mutual Information) for temporal data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tim Sainburg",
    license="MIT",
    rl="https://github.com/timsainb/automi",
    install_requires=[
        "numpy",
        "scipy",
        "tqdm",
        "scikit-learn",
        "joblib",
        "matplotlib",
        "colorednoise",
        "pandas",
        "lmfit",
    ],
)
