setuptools.setup(
    name='ConvNetwork',
    version='0.0.3',
    author='AdamChomicki',
    author_email='ad.chomicki@gmail.com',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/AdamChomicki/ConvNetwork.git',
    project_urls = {
        "Bug Tracker": "https://github.com/AdamChomicki/ConvNetwork.git"
    },
    license='MIT',
    packages=['toolbox'],
    install_requires=['requests'],
)