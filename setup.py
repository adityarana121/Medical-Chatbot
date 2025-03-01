from setuptools import setup, find_packages

setup(
    name="AI Medical ChatBot",
    version="0.0.1",
    author="Aditya Rana",
    author_email="adityaranarana2006@gmail.com",
    packages=find_packages(),
    install_requires=[
        "click",
        "requests",
    ],
    entry_points={
        'console_scripts': [
            'myapp=app:cli',
        ],
    },
)