from setuptools import setup, find_packages

setup(
    name="robohusky",
    version="0.1.0",
    description="An open platform for training, serving, and evaluating large foundation models for robots.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="",  # Add your name
    author_email="",  # Add your email
    url="https://github.com/YaoMarkMu/Robothusky",
    packages=find_packages(exclude=["assets*", "benchmark*", "docs", "dist*", "playground*", "scripts*", "tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.8",
    install_requires=[
        "accelerate", "fastapi", "gradio>=3.23", "markdown2[all]", "numpy",
        "prompt_toolkit>=3.0.0", "requests", "rich>=10.0.0", "sentencepiece",
        "shortuuid", "transformers>=4.29.0", "tokenizers>=0.12.1", "torch",
        "uvicorn", "wandb", "httpx", "shortuuid", "pydantic", "nh3",
    ],
    extras_require={
        "dev": ["black>=23.3.0", "pylint>=2.8.2"]
    },
    project_urls={
        "Homepage": "https://github.com/YaoMarkMu/Robothusky",
        "Bug Tracker": "https://github.com/YaoMarkMu/Robothusky/issues",
    },
)
