import importlib

try:
    backend = importlib.import_module("jax")
    library_name = "jax"
    torch_device = "cpu"
except:
    backend = None
    library_name = None
    torch_device = None

def use(lib_name, torch_device_name = "cpu"):
    global backend, library_name, torch_device

    torch_device = torch_device_name

    if lib_name == "jax":
        try:
            backend = importlib.import_module("jax")
            library_name = "jax"
            print("Using JAX.")
        except ImportError:
            raise ImportError("JAX is not installed")
    elif lib_name == "torch":
        try:
            backend = importlib.import_module("torch")
            library_name = "torch"
            print("Using PyTorch.")
        except ImportError:
            raise ImportError("PyTorch is not installed")
    else:
        raise ValueError("Unsupported library. Use 'jax' or 'torch'.")

def get_backend():
    if backend is None:
        raise RuntimeError("Backend not set. Use nao.use('jax') or nao.use('torch') first.")
    return backend

def get_library_name():
    return library_name

def get_torch_device():
    return torch_device
