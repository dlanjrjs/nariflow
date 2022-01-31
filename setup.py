from setuptools import setup, find_packages

setup(
    name='nariflow',
    version='0.1.0',
    url='https://github.com/dlanjrjs/nariflow.git',
    author='dlanjrjs',
    install_requires=[
        'numpy==1.20.1'
    ],
    packages = find_packages(include = ('core','thirdparty','*')),
    package_data={'' : ['*']},
    include_package_data = True
)

