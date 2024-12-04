from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f.readlines()]


setup(
    name="crowd_calibrator",
    version="1.0",
    author="Urja Khurana",
    author_email="u.khurana@vu.nl",
    url="",
    description="Code for calibrating language models according to subjectivity.",
    packages=find_packages(),
    install_requires=requirements,
)