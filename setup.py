from setuptools import setup

with open('README.md') as f:
    README = f.read()

setup(
    name='inverse-canopy',
    url='https://git@git.space.openpra.org/openpra/inverse-canopy.git',
    version='0.0.3',
    author='Arjun Earthperson',
    author_email='arjun@openpra.org',
    license='AGPL3',
    description='',
    long_description=README,
    install_requires=[
        'tensorflow==2.15.0',
        'tensorflow-probability==0.23.0',
        'numpy',
        'scipy',
        'matplotlib'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.6',
    entry_points={
        # 'console_scripts': [
        #     'inverse-canopy=inverse_canopy:main',
        # ],
    },
)
