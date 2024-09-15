from setuptools import setup, find_packages

setup(
    name="eagle",
    version="0.0.1",
    packages=find_packages(include=["eagle", "eagle.*"]),
    package_data={
        "eagle": ["*"],
        "eagle.model": ["*"],
        "eagle.model.language_model": ["*"],
    },
    py_modules=[
        "eagle.conversation",
        "eagle.constants",
        "eagle.model.builder",
        "eagle.model.language_model",
        "eagle.utils",
        "eagle.mm_utils",
    ],
    install_requires=[
        # Add any dependencies required by the eagle module
    ],
    include_package_data=True,
)
