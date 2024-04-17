from typing import TypeVar, Type

T = TypeVar('T', bound='RegisterClassRoot')


class RegisterClassRoot(object):
    __HIERARCHY = dict()
    __ABSPATH = ""
    
    @classmethod
    def Instantiate(cls: Type[T], name: str, *args, **kwargs) -> T:
        target_cls = cls._read_hierarchy(name)
        return target_cls(*args, **kwargs)
    
    def __init_subclass__(cls, **kwargs) -> None:
        cls.__HIERARCHY = {"": cls}
        
        checkbase = list(filter(lambda x: issubclass(x, RegisterClassRoot), cls.__bases__))
        assert len(checkbase) == 1, "Does not support diamond inheritance in RegisterClassTree"
        direct_pcls = checkbase[0]
        cls.__ABSPATH = direct_pcls.__ABSPATH + "/" + cls.__name__
        
        for pcls in cls.mro()[1:]:
            if not issubclass(pcls, RegisterClassRoot):
                continue
            pcls._write_hierarchy(cls.__ABSPATH, cls)
    
    @classmethod
    def _write_hierarchy(cls, k: str, v: type):
        cls.__HIERARCHY[k.replace(cls.__ABSPATH, "")] = v
    @classmethod
    def _read_hierarchy(cls, k: str):
        if k not in cls.__HIERARCHY:
            print(f"Error: get {k}, expect to be one of {list(cls.__HIERARCHY.keys())}")
        return cls.__HIERARCHY[k]
    @classmethod
    def _get_hierarchy(cls):
        return cls.__HIERARCHY
    @classmethod
    def show_hierarchy(cls, indent=0):
        print("| "*indent + cls.__name__)
        for ccls in cls.__subclasses__():
            ccls.show_hierarchy(indent=indent+1)
