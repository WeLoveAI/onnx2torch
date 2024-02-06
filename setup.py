from setuptools import find_packages, setup

with open("VERSION", "r") as f:
    version = f.read().strip()

with open("o2t/version.py", "w") as f:
    f.write(f'__version__ = "{version}"\n')

setup(
    name="o2t",
    version=version,
    description="onnx2torch: onnx to native pytorch converter",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/inisis/onnx2torch",
    author="inisis",
    author_email="desmond.yao@buaa.edu.cn",
    project_urls={
        "Bug Tracker": "https://github.com/inisis/onnx2torch/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="MIT",
    install_requires=["loguru", "torch", "onnx"],
    packages=find_packages(exclude=("tests", "tests.*")),
    entry_points={"console_scripts": ["onnx2torch=o2t.cli:main"]},
    zip_safe=True,
    python_requires=">=3.6",
)
