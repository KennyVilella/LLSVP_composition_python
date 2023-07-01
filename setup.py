import setuptools

setuptools.setup(
    name="llsvp",
    author="Kenny Vilella",
    author_email="kenny.vilella@gmail.com",
    description="Calculates the material properties of potential LLSVP compositions",
    url="https://github.com/KennyVilella/LLSVP_composition_python",
    project_urls={
        "Source Code": "https://github.com/KennyVilella/LLSVP_composition_python",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
