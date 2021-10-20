
from functools import wraps

def check_properties(instance, **props):
    """ a decorate to check if the instances has method required properties before method is execute
        For example, if move() method need to check item.position, then decorate move() with
        ...: item.check_properties(position: 'required')
        ...: def move(self, x, y, z):
    """
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for k,v in props.items():
                if v == 'required':
                    if getattr(instance, k, None) is None:
                        raise AttributeError(f'this method requires {instance} has property {k}')
            return func(*args, **kwargs)
        return wrapper
    return decorate