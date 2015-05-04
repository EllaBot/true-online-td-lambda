import os

from setuptools import setup

def read(*paths):
    """Build a file path from *paths* and return the contents."""
    with open(os.path.join(*paths), 'r') as f:
        return f.read()

setup(
    name='true_online_td_lambda',
    version='0.0.1',
    description='Implementation of True Online TD(lambda) with a Fourier Basis function approximator.',
    long_description=(read('README.rst')),
    url='https://github.com/EllaBot/true-online-td-lambda/',
    license='MIT',
    author='Rishi Shah, Nick Walker',
    py_modules=['true_online_td_lambda'],
    include_package_data=True,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires=['nose', 'numpy', 'scipy', 'matplotlib']
)
