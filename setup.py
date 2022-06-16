import setuptools



setuptools.setup(
    name='skin-cancer-detection',
    version='0.0.1',
    author='Mariam alaa',
    author_email='mariamalaa2019@gmail.com',
    description='skin cancer detection model',
    long_description="",
    long_description_content_type="text/markdown",
    url='https://github.com/mariamalaa98/skin-cancer-detection',
    license='MIT',
    packages=['skin_cancer'],
    package_dir={
        'skin_cancer': 'src/skin_cancer'},
    install_requires=['torch','torchvision','numpy','opencv-python','pandas','gdown'],
)