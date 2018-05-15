""" Setup up the required packages

Setup all required packages. Configured for the das4 cluster fs4.das4.science.uva.nl (See https://www.cs.vu.nl/das4/).

First create clean environment using:
conda create --name name_environment python=3.6
source activate name_environment

To update one pytorch or my forks set the relevant update flag to True.

Example:
    python setup.py --upgrade_ignite True
"""


import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--update_torch', type=bool, default=False, help='')
parser.add_argument('--update_ignite', type=bool, default=False, help='')
parser.add_argument('--update_vision', type=bool, default=False, help='')
config = parser.parse_args()


def git_clone(repo_name, git_url=None):
    import os
    os.chdir("/var/scratch/blokland")
    os.system(f"rm -f -r {repo_name}")
    url = git_url if git_url else "https://mavanb:lollie12345@github.com/mavanb/{}".format(repo_name)
    os.system(f"git clone --recursive {url}")


def install_pytorch(update):
    if not update:
        try:
            import torch
        except:
            install_pytorch(True)
    else:
        os.system("export CMAKE_PREFIX_PATH='$(dirname $(which conda))/../'")
        os.system("conda install -y numpy pyyaml mkl mkl-include setuptools cmake cffi typing")
        os.system("conda install -y -c pytorch magma-cuda90")
        git_clone("pytorch", "https://github.com/pytorch/pytorch")
        os.system("cd pytorch && python setup.py install && cd ..")
        time.sleep(10)
        print("###### Installed torch ######")
        try:
            import torch
        except:
            print("###### Installed torch failed, try again with intel mkl ######")
            # somehow mkl is not working on conda's env, this intel version does work
            os.system("conda install -y -c intel mkl")
            time.sleep(10)
            try:
                import torch
                print("###### Installed torch with mkl ######")
            except:
                print("###### Torch mkl intel also failed ######")
                exit(1)


def install_ignite(update):
    # install own fork of ignite
    if not update:
        try:
            import ignite
        except:
            install_ignite(True)
    else:
        os.chdir("/var/scratch/blokland")
        git_clone("ignite")
        os.system("cd ignite && python setup.py install && cd ..")


def install_vision(update):
    # install own fork of ignite
    if not update:
        try:
            import torchvision.datasets.smallnorb
        except:
            install_vision(True)
    else:
        os.chdir("/var/scratch/blokland")
        git_clone("vision")
        os.system("cd vision && python setup.py install && cd ..")


install_pytorch(config.update_torch)
install_vision(config.update_vision)
install_ignite(config.update_ignite)

# install visdom
try:
  import visdom
except:
  os.system("pip install visdom")

# install configargparse
try:
  import configargparse
except:
    os.chdir("/var/scratch/blokland")
    os.system(f"rm -f -r ConfigArgParse-0.13.0")
    os.system(f"rm -f ConfigArgParse-0.13.0.tar.gz")
    os.system("wget https://pypi.python.org/packages/77/61/ae928ce6ab85d4479ea198488cf5ffa371bd4ece2030c0ee85ff668deac5/ConfigArgParse-0.13.0.tar.gz#md5=6d3427dce78a17fb48222538f579bdb8")
    os.system("tar -xzf ConfigArgParse-0.13.0.tar.gz")
    os.system("cd ConfigArgParse-0.13.0 && python setup.py install && cd ..")

try:
    import line_profiler
except:
    os.system("pip install line_profiler")

try:
    import matplotlib
except:
    os.system("conda install -y matplotlib")

# test all packages:
try:
    time.sleep(3)
    import torch
    import torchvision.datasets.smallnorb
    import visdom
    import configargparse
    import line_profiler
    import matplotlib
    print("###### All packages are succesfully installed ######")
except:
    print("###### Failed to install all packages ######")
    exit(1)

""" Setup up the required packages

Setup all required packages. Configured for the das4 cluster fs4.das4.science.uva.nl (See https://www.cs.vu.nl/das4/).

First create clean environment using:
conda create --name name_environment python=3.6
source activate name_environment

To update one pytorch or my forks set the relevant update flag to True.

Example:
    python setup.py --upgrade_ignite True
"""


import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--update_torch', type=bool, default=False, help='')
parser.add_argument('--update_ignite', type=bool, default=False, help='')
parser.add_argument('--update_vision', type=bool, default=False, help='')
config = parser.parse_args()


def git_clone(repo_name, git_url=None):
    import os
    os.chdir("/var/scratch/blokland")
    os.system(f"rm -f -r {repo_name}")
    url = git_url if git_url else "https://mavanb:lollie12345@github.com/mavanb/{}".format(repo_name)
    os.system(f"git clone --recursive {url}")


def install_pytorch(update):
    if not update:
        try:
            import torch
        except:
            install_pytorch(True)
    else:
        os.system("export CMAKE_PREFIX_PATH='$(dirname $(which conda))/../'")
        os.system("conda install -y numpy pyyaml mkl mkl-include setuptools cmake cffi typing")
        os.system("conda install -y -c pytorch magma-cuda90")
        git_clone("pytorch", "https://github.com/pytorch/pytorch")
        os.system("cd pytorch && python setup.py install && cd ..")
        time.sleep(10)
        print("###### Installed torch ######")
        try:
            import torch
        except:
            print("###### Installed torch failed, try again with intel mkl ######")
            # somehow mkl is not working on conda's env, this intel version does work
            os.system("conda install -y -c intel mkl")
            time.sleep(10)
            try:
                import torch
                print("###### Installed torch with mkl ######")
            except:
                print("###### Torch mkl intel also failed ######")
                exit(1)


def install_ignite(update):
    # install own fork of ignite
    if not update:
        try:
            import ignite
        except:
            install_ignite(True)
    else:
        os.chdir("/var/scratch/blokland")
        git_clone("ignite")
        os.system("cd ignite && python setup.py install && cd ..")


def install_vision(update):
    # install own fork of ignite
    if not update:
        try:
            import torchvision.datasets.smallnorb
        except:
            install_vision(True)
    else:
        os.chdir("/var/scratch/blokland")
        git_clone("vision")
        os.system("cd vision && python setup.py install && cd ..")


install_pytorch(config.update_torch)
install_vision(config.update_vision)
install_ignite(config.update_ignite)

# install visdom
try:
  import visdom
except:
  os.system("pip install visdom")

# install configargparse
try:
  import configargparse
except:
    os.chdir("/var/scratch/blokland")
    os.system(f"rm -f -r ConfigArgParse-0.13.0")
    os.system(f"rm -f ConfigArgParse-0.13.0.tar.gz")
    os.system("wget https://pypi.python.org/packages/77/61/ae928ce6ab85d4479ea198488cf5ffa371bd4ece2030c0ee85ff668deac5/ConfigArgParse-0.13.0.tar.gz#md5=6d3427dce78a17fb48222538f579bdb8")
    os.system("tar -xzf ConfigArgParse-0.13.0.tar.gz")
    os.system("cd ConfigArgParse-0.13.0 && python setup.py install && cd ..")

try:
    import line_profiler
except:
    os.system("pip install line_profiler")

try:
    import matplotlib
except:
    os.system("conda install -y matplotlib")

# test all packages:
try:
    time.sleep(3)
    import torch
    import torchvision.datasets.smallnorb
    import visdom
    import configargparse
    import line_profiler
    import matplotlib
    print("###### All packages are succesfully installed ######")
except:
    print("###### Failed to install all packages ######")
    exit(1)

