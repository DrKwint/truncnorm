from setuptools import setup


def _parse_requirements(requirements_txt_path):
    with open(requirements_txt_path) as fp:
        return fp.read().splitlines()


setup(
    name='truncnorm',
    version='0.1.0',
    license='MIT',
    description='Truncated Normal and Student\'s t-distribution',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Eleanor Quint',
    author_email='equint@cse.unl.edu',
    packages=['truncnorm'],
    install_requires=_parse_requirements('requirements.txt'),
    requires_python='>=3.6',
    # PyPI package information.
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
)
