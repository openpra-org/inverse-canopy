from setuptools import setup

with open('README.md') as f:
    README = f.read()

setup(
    name='inverse-canopy',
    url='https://openpra.org',
    version='0.0.6',
    author='Arjun Earthperson',
    author_email='arjun@openpra.org',
    license='AGPL-3.0',
    description='',
    long_description_content_type='text/markdown',
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
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Programming Language :: Python :: 3.8',
    ],
    packages=['inverse_canopy'],
    python_requires='>=3.6',
    entry_points={
    },
)
