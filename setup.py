import setuptools

setuptools.setup(
    name='LLSVP_composition',
    author='Kenny Vilella',
    author_email='kenny.vilella@gmail.com',
    description='Calculates the material properties of potential LLSVP compositions',
    url='',
    project_urls={
        'Source Code': '',
    },
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    python_requires='>=3.6',
)
