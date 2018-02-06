#
# from abc import ABC, abstractmethod
#
# class Base(ABC):
#
#     @property
#     @abstractmethod
#     def x(self):
#         return self.__x
#
# class Sub(Base):
#
#     def __init__(self):
#         self.x = 3
#         super(Sub, self).__init__()
#
#     def swim(self):
#         print("hoi")
#
#     @property
#     def x(self):
#         print("ik kom hier")
#         return self.__x
#
#     @x.setter
#     def x(self, x):
#         print("hier ook:)")
#         self.__x = x
#
# sub = Sub()
#
# print("opvargen")
# print(sub.x)

import torch


class A:

    def f(self):
        raise NotImplementedError


class B(A):
    def __init__(self):
        super(B, self).__init__()


#
# class A:
#     def h(self):
#         self.f()
#
#     def f(self):
#         print("A")
#
#
# class B(A):
#     def g(self):
#         self.h()
#
# b = B()
#
# def newf():
#     print("new f")
# b.f = newf
# b.g()
