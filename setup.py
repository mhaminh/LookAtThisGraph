from setuptools import setup

setup(
    name='LookAtThisGraph',
    version='0.1.0',
    author='Martin Ha Minh',
    author_email='martin.haminh@icecube.wisc.edu',
    packages=['lookatthisgraph'],
    # scripts=['bin/script1','bin/script2'],
    # url='http://pypi.python.org/pypi/PackageName/',
    license='MIT',
    description='Utilities and models for IceCube event reconstruction using graph neural networks',
    package_data = {
        'lookatthisgraph.resources': ['*.pkl']

    }

    # long_description=open('README.txt').read(),
  #    install_requires=[
#        "Django >= 1.1.1",
#        "pytest",
#    ],
)
