from setuptools import setup, find_packages

setup(
    name="text_classification_project",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],  # dependencies handled by requirements.txt
    entry_points={
        "console_scripts": [
            "tc-train = src.train:main",
            "tc-predict = src.predict:main",
        ],
    },
)
