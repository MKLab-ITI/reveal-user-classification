__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from setuptools import setup
from setuptools.extension import Extension
try:
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True
import numpy
C_OPT_FLAG = "-O3"


def readme():
    with open("README.md") as f:
        return f.read()

cmdclass = dict()
ext_modules = list()


# Either use cython to translate the .pyx files, or compile the distributed .c files as translated by the author.
if USE_CYTHON:
    ext_modules.append(Extension(name="reveal_user_classification.embedding.arcte.cython_opt.arcte",
                                 sources=["reveal_user_classification/embedding/arcte/cython_opt/arcte.pyx"],
                                 extra_compile_args=[C_OPT_FLAG, '-I/user/local/include/python3.3'],
                                 include_dirs=[numpy.get_include()]))
    ext_modules.append(Extension(name="reveal_user_classification.eps_randomwalk.cython_opt.push",
                                 sources=["reveal_user_classification/eps_randomwalk/cython_opt/push.pyx"],
                                 extra_compile_args=[C_OPT_FLAG, '-fopenmp', '-I/user/local/include/python3.3'],
                                 extra_link_args=['-fopenmp'],
                                 include_dirs=[numpy.get_include()]))
    ext_modules.append(Extension(name="reveal_user_classification.eps_randomwalk.cython_opt.transition",
                                 sources=["reveal_user_classification/eps_randomwalk/cython_opt/transition.pyx"],
                                 extra_compile_args=[C_OPT_FLAG, '-fopenmp', '-I/user/local/include/python3.3'],
                                 extra_link_args=['-fopenmp'],
                                 include_dirs=[numpy.get_include()]))
    ext_modules.append(Extension(name="reveal_user_classification.eps_randomwalk.cython_opt.similarity",
                                 sources=["reveal_user_classification/eps_randomwalk/cython_opt/similarity.pyx"],
                                 extra_compile_args=[C_OPT_FLAG, '-fopenmp', '-I/user/local/include/python3.3'],
                                 extra_link_args=['-fopenmp'],
                                 include_dirs=[numpy.get_include()]))
    cmdclass.update({"build_ext": build_ext})
else:
    ext_modules.append(Extension(name="reveal_user_classification.embedding.arcte.cython_opt.arcte",
                                 sources=["reveal_user_classification/embedding/arcte/cython_opt/arcte.c"],
                                 extra_compile_args=[C_OPT_FLAG, '-I/user/local/include/python3.3'],
                                 include_dirs=[numpy.get_include()]))
    ext_modules.append(Extension(name="reveal_user_classification.eps_randomwalk.cython_opt.push",
                                 sources=["reveal_user_classification/eps_randomwalk/cython_opt/push.c"],
                                 extra_compile_args=[C_OPT_FLAG, '-fopenmp', '-I/user/local/include/python3.3'],
                                 extra_link_args=['-fopenmp'],
                                 include_dirs=[numpy.get_include()]))
    ext_modules.append(Extension(name="reveal_user_classification.eps_randomwalk.cython_opt.transition",
                                 sources=["reveal_user_classification/eps_randomwalk/cython_opt/transition.c"],
                                 extra_compile_args=[C_OPT_FLAG, '-fopenmp', '-I/user/local/include/python3.3'],
                                 extra_link_args=['-fopenmp'],
                                 include_dirs=[numpy.get_include()]))
    ext_modules.append(Extension(name="reveal_user_classification.eps_randomwalk.cython_opt.similarity",
                                 sources=["reveal_user_classification/eps_randomwalk/cython_opt/similarity.c"],
                                 extra_compile_args=[C_OPT_FLAG, '-fopenmp', '-I/user/local/include/python3.3'],
                                 extra_link_args=['-fopenmp'],
                                 include_dirs=[numpy.get_include()]))

setup(
    name='reveal-user-classification',
    version='0.1.10',
    author='Georgios Rizos',
    author_email='georgerizos@iti.gr',
    packages=['reveal_user_classification',
              'reveal_user_classification.datautil',
              'reveal_user_classification.embedding',
              'reveal_user_classification.embedding.arcte',
              'reveal_user_classification.embedding.arcte.cython_opt',
              'reveal_user_classification.eps_randomwalk',
              'reveal_user_classification.eps_randomwalk.benchmarks',
              'reveal_user_classification.eps_randomwalk.benchmarks.time_comparisons',
              'reveal_user_classification.eps_randomwalk.cython_opt',
              'reveal_user_classification.experiments',
              'reveal_user_classification.experiments.asu_experiments',
              'reveal_user_classification.entry_points'],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    url='https://github.com/MKLab-ITI/reveal-user-classification',
    license='Apache',
    description='Performs user classification into labels using a set of seed Twitter users with known labels and'
                'the structure of the interaction network between them.',
    long_description=readme(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],
    keywords="online-social-network user-classification Reveal-FP7",
    entry_points={
        'console_scripts': ['user_network_profile_classifier=reveal_user_classification.entry_points.user_network_profile_classifier:main'],
    },
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "Cython",
        "python-louvain",
        "reveal-user-annotation"
    ],
)
