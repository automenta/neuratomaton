from setuptools import setup, find_packages

setup(
    name="ana-nlp",
    version="1.0.0",
    description="Adaptive Neural Automaton - Advanced SSM with Associative Memory",
    author="ANA Research Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tensorboard>=2.13.0",
        "pytest>=7.0.0",
        "tqdm>=4.64.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.0.0",
        "umap-learn>=0.5.0",
        "optuna>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "experiments": [
            "datasets",
            "transformers",
            "tiktoken",
        ]
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "ana-research=ana.cli:main",
        ],
    },
)
