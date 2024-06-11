# Modified from: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/registry.py

import os.path as osp
import glob
import importlib


class Registry():
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('dataset', 'data')

    To register an object:

    .. code-block:: python

        @DATASET_REGISTRY.register()
        class MyDataset():
            ...

    Or:

    .. code-block:: python

        DATASET_REGISTRY.register(MyDataset)

    To retrieve a registered object:
    .. code-python:: python
        DATA_REGISTRY.get('MyDataset')

    This will register all files in data folder that ended with '_dataset.py' and then
    use can retrieve the registered object using its class name.
    Normally used with Models, Datasets, Metrics, and Losses.
    """

    def __init__(self, name, root):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._root = root
        self._obj_map = {}
        self._registered = False

    def _do_register(self, name, obj):
        assert (name not in self._obj_map), (f"An object named '{name}' was already registered "
                                             f"in '{self._name}' registry!")
        self._obj_map[name] = obj

    def register(self, obj=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not.
        See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            ret = self._obj_map.get(name)
            print(f'Name {name} is not found, use name: {name}!')
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()

    def scan_and_register(self):
        if self._registered:
            return
        python_files = glob.glob(osp.join(self._root, f'*_{self._name}.py'))
        python_files = [osp.basename(x.replace('.py', '')) for x in python_files]
        [importlib.import_module(f'{self._root}.{file_name}') for file_name in python_files]
        self._registered = True


DATASET_REGISTRY = Registry('dataset', 'data')
MODEL_REGISTRY = Registry('model', 'models')
LOSS_REGISTRY = Registry('loss', 'losses')
METRIC_REGISTRY = Registry('metric', 'metrics')
