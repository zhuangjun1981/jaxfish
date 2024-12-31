from setuptools import setup, find_packages
import io
import os
import re

here = os.path.abspath(os.path.dirname(__file__))


def read(*filenames, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


long_description = read("README.md")


def prepend_find_packages(*roots):
    """
    Recursively traverse nested packages under the root directories
    """
    packages = []

    for root in roots:
        packages += [root]
        packages += [root + "." + s for s in find_packages(root)]

    return packages


# find meta data
def extract_meta(text, key_word):
    match = re.search(rf'__{key_word}__\s*=\s*"([^"]+)"', text)
    if match:
        return match.group(1)
    raise RuntimeError(f"Unable to find {key_word} string.")


def find_meta(f_path):
    with open(f_path, "r") as f:
        init_file = f.read()
    version = extract_meta(init_file, "version")
    license = extract_meta(init_file, "license")
    author = extract_meta(init_file, "author")
    author_email = extract_meta(init_file, "author_email")
    url = extract_meta(init_file, "url")
    downloadUrl = extract_meta(init_file, "downloadUrl")

    return version, license, author, author_email, url, downloadUrl


version, license, author, author_email, url, download = find_meta(
    os.path.join(here, "jaxfish", "__init__.py")
)


setup(
    name="jaxfish",
    version=version,
    url=url,
    author=author,
    tests_require=["pytest"],
    install_requires=[
        # "jax",
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "h5py",
        # "PyQt5-tools",
        "pyyaml",
        "jupyter",
        # "PyQt5",
    ],
    author_email=author_email,
    description="jaxfish brain network simulation",
    long_description=long_description,
    packages=prepend_find_packages("jaxfish"),
    include_package_data=True,
    package_data={"": ["*.md", "*.txt", "*.cfg"]},
    platforms="any",
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ],
)
