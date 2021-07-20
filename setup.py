from setuptools import setup


with open("README.md") as f:
    long_description = f.read()

setup(
    name="rtsmplx",
    version="0.0.1",
    description="Real time SMPL-X model",
    long_description=long_description,
    author="Thomas Ackermann",
    author_email="t.ackermann@stud.uni-heidelberg.de",
    packages=["rtsmplx"],
)



