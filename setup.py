from setuptools import setup


def readme():
    with open('README.md', 'r', encoding="UTF-8") as f:
        README = f.read()
    return README


def requirements():
    with open('requirements.txt', 'r', encoding='UTF-8') as f:
        REQUIREMENTS = f.read().split('\n')
    return REQUIREMENTS


setup(name='agentclpr',
      packages=[
          'agentclpr', 'agentclpr.infer'
      ],
      include_package_data=True,
      version='1.0.0',
      install_requires=requirements(),
      license='Apache License 2.0',
      description='An easy-to-use Chinese license plate recognition system.',
      url='https://github.com/AgentMaker/AgentCLPR',
      author='jm12138',
      long_description=readme(),
      long_description_content_type='text/markdown')