import collections
import os
import functools
import inspect
from typing import Union, get_origin, get_args
import types

# def check_parameters(func):
#     """
#     参数检查装饰器，需要配合函数注解表达式（Function Annotations）使用。
#     使用 inspect.signature 来处理复杂的函数签名，并格式化错误信息。
#     """
#     # 定义错误信息模板
#     error_msg = 'Argument "{argument}" must be of type {expected!r}, but got {got!r}, value {value!r}'
    
#     # 获取函数的签名和参数
#     sig = inspect.signature(func)
#     parameters = sig.parameters  # 获取有序参数字典
#     arg_keys = tuple(parameters.keys())  # 获取参数名称

#     # 定义一个命名元组，用于存储每个参数的注解、名称和实际值
#     CheckItem = collections.namedtuple('CheckItem', ('anno', 'arg_name', 'value'))

#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         # 创建一个列表，用于存储要检查的参数信息
#         check_list = []

#         # 遍历位置参数 *args 并收集其注解和实际值
#         for i, value in enumerate(args):
#             arg_name = arg_keys[i]
#             anno = parameters[arg_name].annotation
#             check_list.append(CheckItem(anno, arg_name, value))

#         # 遍历关键字参数 **kwargs 并收集其注解和实际值
#         for arg_name, value in kwargs.items():
#             anno = parameters[arg_name].annotation
#             check_list.append(CheckItem(anno, arg_name, value))

#         # 检查每个参数的类型是否符合注解要求
#         for item in check_list:
#             if item.anno is not inspect.Parameter.empty:  # 如果参数有注解
#                 anno_origin = get_origin(item.anno)  # 获取泛型的原始类型
#                 anno_args = get_args(item.anno)  # 获取泛型的参数类型
                
#                 # 如果是联合类型（Union）
#                 if anno_origin in (Union, types.UnionType):
#                     if not any(
#                         isinstance(item.value, arg) if not get_origin(arg) else (
#                             isinstance(item.value, get_origin(arg)) and all(isinstance(v, get_args(arg)[0]) for v in item.value)
#                         )
#                         for arg in anno_args
#                     ):
#                         raise TypeError(
#                             error_msg.format(
#                                 argument=item.arg_name,
#                                 expected=item.anno,
#                                 got=type(item.value),
#                                 value=item.value
#                             )
#                         )
#                 # 如果是泛型类型（如 List[datetime]）
#                 elif anno_origin:
#                     if not isinstance(item.value, anno_origin) or not all(isinstance(v, anno_args[0]) for v in item.value):
#                         raise TypeError(
#                             error_msg.format(
#                                 argument=item.arg_name,
#                                 expected=item.anno,
#                                 got=type(item.value),
#                                 value=item.value
#                             )
#                         )
#                 else:
#                     if not isinstance(item.value, item.anno):
#                         raise TypeError(
#                             error_msg.format(
#                                 argument=item.arg_name,
#                                 expected=item.anno,
#                                 got=type(item.value),
#                                 value=item.value
#                             )
#                         )

#         # 调用原始函数
#         return func(*args, **kwargs)

#     return wrapper


def check_parameters(func):
    """
    参数检查装饰器，需要配合函数注解表达式（Function Annotations）使用。
    使用 inspect.signature 来处理复杂的函数签名，并格式化错误信息。
    """
    # 定义错误信息模板
    error_msg = 'Argument "{argument}" must be of type {expected!r}, but got {got!r}, value {value!r}'
    
    # 获取函数的签名和参数
    sig = inspect.signature(func)
    parameters = sig.parameters  # 获取有序参数字典
    arg_keys = tuple(parameters.keys())  # 获取参数名称

    # 定义一个命名元组，用于存储每个参数的注解、名称和实际值
    CheckItem = collections.namedtuple('CheckItem', ('anno', 'arg_name', 'value'))

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 创建一个列表，用于存储要检查的参数信息
        check_list = []

        # 遍历位置参数 *args 并收集其注解和实际值
        for i, value in enumerate(args):
            param = parameters[arg_keys[i]]
            # 跳过VAR_POSITIONAL参数（例如 *args）
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                continue
            anno = param.annotation
            check_list.append(CheckItem(anno, arg_keys[i], value))

        # 遍历关键字参数 **kwargs 并收集其注解和实际值
        for arg_name, value in kwargs.items():
            if arg_name in parameters:
                param = parameters[arg_name]
                # 跳过VAR_KEYWORD参数（例如 **kwargs）
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    continue
                anno = param.annotation
                check_list.append(CheckItem(anno, arg_name, value))
            # 如果传入的关键字参数不在签名中，直接忽略

        # 检查每个参数的类型是否符合注解要求
        for item in check_list:
            if item.anno is not inspect.Parameter.empty:  # 如果参数有注解
                anno_origin = get_origin(item.anno)  # 获取泛型的原始类型
                anno_args = get_args(item.anno)  # 获取泛型的参数类型
                
                # 如果是联合类型（Union）
                if anno_origin in (Union, types.UnionType):
                    if not any(
                        isinstance(item.value, arg) if not get_origin(arg) else (
                            isinstance(item.value, get_origin(arg)) and all(isinstance(v, get_args(arg)[0]) for v in item.value)
                        )
                        for arg in anno_args
                    ):
                        raise TypeError(
                            error_msg.format(
                                argument=item.arg_name,
                                expected=item.anno,
                                got=type(item.value),
                                value=item.value
                            )
                        )
                # 如果是泛型类型（如 List[datetime]）
                elif anno_origin:
                    if not isinstance(item.value, anno_origin) or not all(isinstance(v, anno_args[0]) for v in item.value):
                        raise TypeError(
                            error_msg.format(
                                argument=item.arg_name,
                                expected=item.anno,
                                got=type(item.value),
                                value=item.value
                            )
                        )
                else:
                    if not isinstance(item.value, item.anno):
                        raise TypeError(
                            error_msg.format(
                                argument=item.arg_name,
                                expected=item.anno,
                                got=type(item.value),
                                value=item.value
                            )
                        )

        # 调用原始函数
        return func(*args, **kwargs)

    return wrapper


def _determine_processes(num_processes) -> int:
    if num_processes is None:
        num_processes = int(os.cpu_count() * 0.5)
    elif num_processes < 1 and num_processes > 0:
        num_processes = int(os.cpu_count() * num_processes)
    elif num_processes > 1:
        num_processes = int(num_processes)
        
    if num_processes < 1:
        num_processes = 1
        
    return num_processes