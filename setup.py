from setuptools import setup

with open('requirements.txt') as reqs_file:
    reqs = reqs_file.read().splitlines()

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='convnetwork',
    url='https://github.com/AdamChomicki/ConvNetwork',
    author='AdamChomicki',
    author_email='ad.chomicki@gmail.com',
    # Needed to actually package something
    packages=['convnetwork'],
    # Needed for dependencies
    install_requires= reqs,
    # strongly suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='An example of a python package from pre-existing code and existing gan implementation',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
