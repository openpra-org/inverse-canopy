from setuptools import setup

with open('README.md') as f:
    README = f.read()

setup(
    name='inverse-canopy',
    url='ssh://git@git.space.openpra.org:2222/openpra/inverse-canopy.git',
    version='0.0.1',
    author='Arjun Earthperson',
    author_email='arjun@openpra.org',
    license='MIT',
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
        'console_scripts': [
            'inverse-canopy=inverse-canopy:main',
        ],
    },
)
