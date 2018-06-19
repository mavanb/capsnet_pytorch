""" Setup up the required packages

Setup all required packages. Configured for the das4 cluster fs4.das4.science.uva.nl (See https://www.cs.vu.nl/das4/).

First create clean environment using:
conda create --name name_environment python=3.6
source activate name_environment

To update one pytorch or my forks set the relevant update flag to True.

Example local:
    python setup.py install
    im
Example das4:
    python setup.py install --torch_source True --folder "/var/scratch/blokland"

To tests if the packages are succesfully isntalled run:

python setup.py tests

"""
import os
import argparse
import time

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(help='use tests or install mode')

parser_test = subparsers.add_parser('tests', help='tests whether all packages are installed successfully')
parser_test.set_defaults(cmd='tests')

parser_install = subparsers.add_parser('install', help='install all required packages')
parser_install.add_argument('--update_torch', type=bool, default=False, help='')
parser_install.add_argument('--update_ignite', type=bool, default=False, help='')
parser_install.add_argument('--update_vision', type=bool, default=False, help='')
parser_install.add_argument('--folder', type=str, default="/home/blokland", help='Location of all source files')
parser_install.add_argument('--torch_source', type=bool, default=False, help='Install torch from source. Should be done on das4')
parser_install.set_defaults(cmd='install')

config = parser.parse_args()


if config.cmd is "tests":
    # tests all packages:
    try:
        import torch
        import torchvision.datasets.smallnorb
        import visdom
        import configargparse
        import line_profiler
        import matplotlib
        print("###### All packages are succesfully installed ######")
        exit(0)
    except Exception as e:
        print("###### Failed to install all packages ######")
        exit(1)


def git_clone(repo_name, git_url=None):
    import os
    os.chdir(config.folder)
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
        if config.torch_source:
            print("###### Installing torch from source ######")
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
        else:
            print("###### Installing torch no CPU ######")
            os.system("conda install -y pytorch-cpu torchvision-cpu -c pytorch")


def install_ignite(update):
    # install own fork of ignite
    if not update:
        try:
            import ignite
        except:
            install_ignite(True)
    else:
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
        git_clone("vision")
        os.system("cd vision && python setup.py install && cd ..")


install_pytorch(config.update_torch)
install_vision(config.update_vision)
install_ignite(config.update_ignite)

# install visdom
try:
    import visdom
except:
    ## on das4 first installing this: conda install -c conda-forge visdom, fixed the SSL error
    os.system("pip install visdom")

# install configargparse
try:
  import configargparse
except:
    os.chdir(config.folder)
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


if config.update_vision:
    print("Warning: Updated torchvision. Don't forget to remove previously saved data files.")