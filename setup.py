from setuptools import setup, find_packages

setup(
    name="agentcompiler",
    version="0.1.0",
    description="A compiler for agentic AI workloads: parallelism extraction, LLM call merging, speculative branch execution",
    author="Mahdi Rashidiyan",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[],   # zero dependencies: pure asyncio + stdlib
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
