from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='block_movement_pruning',
      version='0.1',
      description='block_movement_pruning is a python package for experimenting on block-sparse pruned version of popular networks.',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.0',
        'Topic :: Text Processing',
      ],
      keywords='',
      url='',
      author='',
      author_email='',
      license='MIT',
      packages=['block_movement_pruning'],
      entry_points={
          'console_scripts': ['block_movement_pruning_run=block_movement_pruning.command_line:train_command'],
      },
      include_package_data=True,
      zip_safe=False)