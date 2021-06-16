import os
from typing import List, Optional, Union
import numpy as np
from hub.core import compression

import numpy as np

from hub.core.typing import StorageProvider
from hub.core.compression import BaseImgCodec, BaseNumCodec
from hub.util.exceptions import ArrayShapeInfoNotFound
from hub.util.dataset import get_compressor


def sample_from_index_entry(
    key: str,
    storage: StorageProvider,
    index_entry: dict,
    dtype: str,
) -> np.ndarray:
    """Get the un-chunked sample from a single `index_meta` entry.

    Args:
        key (str): Key relative to `storage` where this instance.
        storage (StorageProvider): Storage of the sample.
        index_entry (dict): Index metadata of sample with `chunks_names`, `start_byte` and `end_byte` keys.
        dtype (str): Data type of the sample.

    Returns:
        Numpy array from the bytes of the sample.
    """

    chunk_names = index_entry["chunk_names"]
    shape = index_entry["shape"]
    compressor = get_compressor(index_entry["compression"])
    # sample has no data
    if len(chunk_names) <= 0:
        return np.zeros(shape, dtype=dtype)

    b = bytearray()
    for chunk_name in chunk_names:
        chunk_key = os.path.join(key, "chunks", chunk_name)
        last_b_len = len(b)
        b.extend(storage[chunk_key])

    start_byte = index_entry["start_byte"]
    end_byte = last_b_len + index_entry["end_byte"]

    return array_from_buffer(
        memoryview(b),
        dtype,
        compressor,
        shape,
        start_byte,
        end_byte,
    )


def array_from_buffer(
    b: memoryview,
    dtype: str,
    compressor: Union[BaseImgCodec, BaseNumCodec, None],
    shape: tuple = None,
    start_byte: int = 0,
    end_byte: Optional[int] = None,
) -> np.ndarray:
    """Reconstruct a sample from bytearray (memoryview) only using the bytes `b[start_byte:end_byte]`. By default all
    bytes are used.

    Args:
        b (memoryview): Bytes that should be decompressed and converted to array.
        dtype (str): Data type of the sample.
        compressor (BaseImgCodec/BaseNumCodec/None): Compressor applied on the sample.
        shape (tuple): Array shape from index entry.
        start_byte (int): Get only bytes starting from start_byte.
        end_byte (int, optional): Get only bytes up to end_byte.

    Returns:
        Numpy array from the bytes of the sample.

    Raises:
        ArrayShapeInfoNotFound: If no info about sample shape is in meta.
    """

    partial_b = b[start_byte:end_byte]
    if compressor is not None:
        if isinstance(compressor, BaseImgCodec):
            partial_b = compressor.decode_single_image(partial_b)
            array = partial_b
        else:
            partial_b = compressor.decode(partial_b)
    array = np.frombuffer(partial_b, dtype=dtype)
    if shape is not None:
        array = array.reshape(shape)
    else:
        raise ArrayShapeInfoNotFound()
    return array
