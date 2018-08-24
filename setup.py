from setuptools import setup
from setuptools import find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext

cmdclass = { }
ext_modules = [ ]

ext_modules += [
    Extension("genetic_selection.GeneticSelectionCV", ["genetic_selection/__init__.pyx"]),
]
cmdclass.update({ 'build_ext': build_ext })

setup(name='sklearn-genetic',
      version='0.1.2',
      description='Genetic feature selection module for scikit-learn',
      url='http://github.com/netzkontrast/sklearn-genetic',
      original_author='Manuel Calzolari',
      author='Michael Schimmer',
      author_email='',
      license='GPLv3',
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      install_requires=['scikit-learn>=0.18', 'deap>=1.0.2', 'joblib'],
      packages=find_packages())
