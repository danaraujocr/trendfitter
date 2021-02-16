import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="trendfitter", # Replace with your own username
    version="0.0.3",
    author="Daniel Rodrigues",
    author_email="dan.araujocr@gmail.com",
    description="Latent Variable Modelling made easy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danaraujocr/trendfitter",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = ['numpy', 'pandas', 'sklearn', 'scipy'] 
)