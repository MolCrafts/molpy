# Frame

The `Frame` is a static data structure designed for aligning and handling data efficiently, particularly suited for numerical calculations with NumPy, as well as for reading and writing data, without altering the underlying topology. It serves as a standardized format for integration with other components and calculators.

::: note
    In contrast, the `Struct` is a dynamic data structure optimized for editing molecular structures. 

The `Frame` is built upon a pandas DataFrame, functioning as a multi-key directory. Let's play it with a simple example:

``` python
import molpy as mp

# Create a Frame with some data
frame = mp.Frame(
    "atoms"={
        'name': ["H", "O", "C"],
    },
    "bonds"={
        'i': [0, 1],
        'j': [1, 2],
    }
)
x = frame["atoms", "x"]  # pd.DataFrame
```
Usually, frame is created by IO modules.
