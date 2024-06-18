from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()
package_data = {
    "": ["*.cpp", "*.cu"],
}

optional_packages = {
    "tf" : ['tensorflow>=2.2.0', 'tensorflow-text', 'tensorflow-hub']
}

setup(
    name="dexter-cqa",
    version="1.0.8",
    author="Venktesh V, Deepali Prabhu",
    author_email="venkyviswa12@gmail.com",
    description="A Benchmark for Complex Heterogeneous Question answering",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/VenkteshV/BCQA",
    download_url="",
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'sentence-transformers',
        'pytrec_eval',
        'faiss_cpu',
        'elasticsearch==7.9.1',
        'data',
        'toml',
        'zope.interface',
        'transformers==4.30.0',
        'protobuf',
        'openai',
        'annoy',
        'pytrec_eval',
        'joblib',
        'tqdm',
        'pandas',
        "ujson",
        "gitpython"
    ],
    extras_require = optional_packages,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    package_data=package_data,
    keywords="Information Retrieval Transformer Networks Complex Question Answering BERT PyTorch Question Answering IR NLP deep learning"
)
