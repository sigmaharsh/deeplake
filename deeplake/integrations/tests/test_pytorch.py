import deeplake
import numpy as np
import pickle
import pytest

from deeplake.util.remove_cache import get_base_storage
from deeplake.util.exceptions import DatasetUnsupportedPytorch, TensorDoesNotExistError
from deeplake.tests.common import (
    requires_torch,
    convert_data_according_to_torch_version,
)
from deeplake.core.dataset import Dataset
from deeplake.core.index.index import IndexEntry
from deeplake.core.storage.memory import MemoryProvider
from deeplake.constants import KB

from PIL import Image  # type: ignore

try:
    from torch.utils.data._utils.collate import default_collate
except ImportError:
    pass

# ensure tests have multiple chunks without a ton of data
PYTORCH_TESTS_MAX_CHUNK_SIZE = 5 * KB


def double(sample):
    return sample * 2


def to_tuple(sample, t1, t2):
    return sample[t1], sample[t2]


def reorder_collate(batch):
    x = [((x["a"], x["b"]), x["c"]) for x in batch]
    return default_collate(x)


def dict_to_list(sample):
    return [sample["a"], sample["b"], sample["c"]]


def my_transform_collate(batch):
    x = [(c, a, b) for a, b, c in batch]
    return default_collate(x)


def pytorch_small_shuffle_helper(start, end, dataloader):
    for _ in range(2):
        all_values = []
        for i, batch in enumerate(dataloader):
            value = batch["image"].numpy()[0][0][0]
            value2 = batch["image2"].numpy()[0][0][0]
            assert value == value2
            all_values.append(value)
            np.testing.assert_array_equal(
                batch["image"].numpy(), value * np.ones(batch["image"].shape)
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), value2 * np.ones((1, 12, 12))
            )

    assert set(all_values) == set(range(start, end))


@requires_torch
def test_pytorch_small(local_ds):
    ds = local_ds
    with ds:
        ds.create_tensor("image", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(([i * np.ones((i + 1, i + 1)) for i in range(16)]))
        ds.commit()
        ds.create_tensor("image2", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = ds.pytorch(num_workers=0)
        return

    dl = ds.pytorch(num_workers=2, batch_size=1)

    assert len(dl.dataset) == 16

    for _ in range(2):
        for i, batch in enumerate(dl):
            np.testing.assert_array_equal(
                batch["image"].numpy(), i * np.ones((1, i + 1, i + 1))
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), i * np.ones((1, 12, 12))
            )

    dls = ds.pytorch(num_workers=2, batch_size=1, shuffle=True)
    pytorch_small_shuffle_helper(0, 16, dls)

    sub_ds = ds[5:]

    sub_dl = sub_ds.pytorch(num_workers=0)

    for i, batch in enumerate(sub_dl):
        np.testing.assert_array_equal(
            batch["image"].numpy(), (5 + i) * np.ones((1, 6 + i, 6 + i))
        )
        np.testing.assert_array_equal(
            batch["image2"].numpy(), (5 + i) * np.ones((1, 12, 12))
        )

    sub_dls = sub_ds.pytorch(num_workers=0, batch_size=1, shuffle=True)
    pytorch_small_shuffle_helper(5, 16, sub_dls)

    sub_ds2 = ds[8:12]

    sub_dl2 = sub_ds2.pytorch(num_workers=0, batch_size=1)

    for _ in range(2):
        for i, batch in enumerate(sub_dl2):
            np.testing.assert_array_equal(
                batch["image"].numpy(), (8 + i) * np.ones((1, 9 + i, 9 + i))
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), (8 + i) * np.ones((1, 12, 12))
            )

    sub_dls2 = sub_ds2.pytorch(num_workers=2, batch_size=1, shuffle=True)
    pytorch_small_shuffle_helper(8, 12, sub_dls2)

    sub_ds3 = ds[:5]

    sub_dl3 = sub_ds3.pytorch(num_workers=0, batch_size=1)

    for _ in range(2):
        for i, batch in enumerate(sub_dl3):
            np.testing.assert_array_equal(
                batch["image"].numpy(), (i) * np.ones((1, i + 1, i + 1))
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), (i) * np.ones((1, 12, 12))
            )

    sub_dls3 = sub_ds3.pytorch(num_workers=0, batch_size=1, shuffle=True)
    pytorch_small_shuffle_helper(0, 5, sub_dls3)


@requires_torch
def test_pytorch_transform(local_ds):
    import torch
    ds = local_ds

    with ds:
        ds.create_tensor("image", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(([i * np.ones((i + 1, i + 1)) for i in range(16)]))
        ds.checkout("alt", create=True)
        ds.create_tensor("image2", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = ds.pytorch(num_workers=0)
        return

    dl = ds.pytorch(
        num_workers=2,
        transform=to_tuple,
        batch_size=1,
        transform_kwargs={"t1": "image", "t2": "image2"},
    )

    for _ in range(2):
        for i, batch in enumerate(dl):
            if torch.__version__ < "2.0.0":
                actual_image, actual_image2 = batch[0]

                expected_image = i * np.ones((i + 1, i + 1))
                expected_image2 = i * np.ones((12, 12))
                np.testing.assert_array_equal(actual_image, expected_image)
                np.testing.assert_array_equal(actual_image2, expected_image2)
            else:
                actual_image, actual_image2 = batch

                expected_image = i * np.ones((1, i + 1, i + 1))
                expected_image2 = i * np.ones((1, 12, 12))
                np.testing.assert_array_equal(actual_image, expected_image)
                np.testing.assert_array_equal(actual_image2, expected_image2)

    dls = ds.pytorch(
        num_workers=0,
        transform=to_tuple,
        batch_size=1,
        shuffle=True,
        transform_kwargs={"t1": "image", "t2": "image2"},
    )

    for _ in range(2):
        all_values = []
        for i, batch in enumerate(dls):
            if torch.__version__ < "2.0.0":
                actual_image, actual_image2 = batch[0]

                value = actual_image[0][0]
                value2 = actual_image2[0][0]
                assert value == value2
                all_values.append(value)

                expected_image = value * np.ones(actual_image.shape)
                expected_image2 = value * np.ones(actual_image2.shape)
                np.testing.assert_array_equal(actual_image, expected_image)
                np.testing.assert_array_equal(actual_image2, expected_image2)
            else:
                actual_image, actual_image2 = batch

                value = actual_image[0][0][0]
                value2 = actual_image2[0][0][0]
                assert value == value2
                all_values.append(value)

                expected_image = value * np.ones(actual_image.shape)
                expected_image2 = value * np.ones(actual_image2.shape)
                np.testing.assert_array_equal(actual_image, expected_image)
                np.testing.assert_array_equal(actual_image2, expected_image2)


@requires_torch
def test_pytorch_transform_dict(local_ds):
    ds = local_ds
    with ds as ds:
        ds.create_tensor("image", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(([i * np.ones((i + 1, i + 1)) for i in range(16)]))
        ds.create_tensor("image2", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))
        ds.create_tensor("image3", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image3.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = ds.pytorch(num_workers=0)
        return

    dl = ds.pytorch(
        num_workers=2, transform={"image": double, "image2": None}, batch_size=1
    )

    assert len(dl.dataset) == 16

    for _ in range(2):
        for i, batch in enumerate(dl):
            assert set(batch.keys()) == {"image", "image2"}
            np.testing.assert_array_equal(
                batch["image"].numpy(), 2 * i * np.ones((1, i + 1, i + 1))
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), i * np.ones((1, 12, 12))
            )

    for _ in range(2):
        for i, (image, image2) in enumerate(dl):
            np.testing.assert_array_equal(
                image.numpy(), 2 * i * np.ones((1, i + 1, i + 1))
            )
            np.testing.assert_array_equal(image2.numpy(), i * np.ones((1, 12, 12)))


@requires_torch
def test_pytorch_with_compression(local_ds: Dataset):
    ds = local_ds
    # TODO: chunk-wise compression for labels (right now they are uncompressed)
    with ds:
        images = ds.create_tensor(
            "images",
            htype="image",
            sample_compression="png",
            max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE,
        )
        labels = ds.create_tensor(
            "labels", htype="class_label", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE
        )

        assert images.meta.sample_compression == "png"

        images.extend(np.ones((16, 12, 12, 3), dtype="uint8"))
        labels.extend(np.ones((16, 1), dtype="uint32"))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = ds.pytorch(num_workers=0)
        return

    dl = ds.pytorch(num_workers=0, batch_size=1)
    dls = ds.pytorch(num_workers=0, batch_size=1, shuffle=True)

    for dataloader in [dl, dls]:
        for _ in range(2):
            for batch in dataloader:
                X = batch["images"].numpy()
                T = batch["labels"].numpy()
                assert X.shape == (1, 12, 12, 3)
                assert T.shape == (1, 1)


@requires_torch
def test_custom_tensor_order(local_ds):
    ds = local_ds
    with ds:
        tensors = ["a", "b", "c", "d"]
        for t in tensors:
            ds.create_tensor(t, max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
            ds[t].extend(np.random.random((3, 4, 5)))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = ds.pytorch(num_workers=0)
        return

    with pytest.raises(TensorDoesNotExistError):
        dl = ds.pytorch(num_workers=0, tensors=["c", "d", "e"])

    dl = ds.pytorch(num_workers=0, tensors=["c", "d", "a"], return_index=False)

    for i, batch in enumerate(dl):
        c1, d1, a1 = batch
        a2 = batch["a"]
        c2 = batch["c"]
        d2 = batch["d"]
        assert "b" not in batch
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(a1[0], ds.a.numpy()[i])
        np.testing.assert_array_equal(c1[0], ds.c.numpy()[i])
        np.testing.assert_array_equal(d1[0], ds.d.numpy()[i])
        batch = pickle.loads(pickle.dumps(batch))
        c1, d1, a1 = batch
        a2 = batch["a"]
        c2 = batch["c"]
        d2 = batch["d"]
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(a1[0], ds.a.numpy()[i])
        np.testing.assert_array_equal(c1[0], ds.c.numpy()[i])
        np.testing.assert_array_equal(d1[0], ds.d.numpy()[i])

    dls = ds.pytorch(num_workers=0, tensors=["c", "d", "a"], return_index=False)
    for i, batch in enumerate(dls):
        c1, d1, a1 = batch
        a2 = batch["a"]
        c2 = batch["c"]
        d2 = batch["d"]
        assert "b" not in batch
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(d1, d2)
        batch = pickle.loads(pickle.dumps(batch))
        c1, d1, a1 = batch
        a2 = batch["a"]
        c2 = batch["c"]
        d2 = batch["d"]
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(d1, d2)


@requires_torch
def test_readonly_with_two_workers(local_ds):
    local_ds.create_tensor("images", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
    local_ds.create_tensor("labels", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
    local_ds.images.extend(np.ones((10, 12, 12)))
    local_ds.labels.extend(np.ones(10))

    base_storage = get_base_storage(local_ds.storage)
    base_storage.flush()
    base_storage.enable_readonly()
    ds = Dataset(storage=local_ds.storage, read_only=True, verbose=False)

    ptds = ds.pytorch(num_workers=2)
    # no need to check input, only care that readonly works
    for _ in ptds:
        pass


@requires_torch
def test_corrupt_dataset(local_ds, corrupt_image_paths, compressed_image_paths):
    local_ds.storage.clear()
    img_good = deeplake.read(compressed_image_paths["jpeg"][0])
    img_bad = deeplake.read(corrupt_image_paths["jpeg"])
    with local_ds:
        local_ds.create_tensor("image", htype="image", sample_compression="jpeg")
        for i in range(3):
            for i in range(10):
                local_ds.image.append(img_good)
            local_ds.image.append(img_bad)
    num_samples = 0
    num_batches = 0
    dl = local_ds.pytorch(num_workers=0, batch_size=2, return_index=False)
    for (batch,) in dl:
        num_batches += 1
        num_samples += len(batch)
    assert num_samples == 30
    assert num_batches == 15


@requires_torch
def test_pytorch_local_cache(local_ds):
    ds = local_ds
    with ds:
        ds.create_tensor("image", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(([i * np.ones((i + 1, i + 1)) for i in range(16)]))
        ds.create_tensor("image2", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = ds.pytorch(num_workers=0)
        return

    epochs = 2

    for _ in range(epochs):
        dl = ds.pytorch(num_workers=1, batch_size=1, use_local_cache=True)
        for i, batch in enumerate(dl):
            np.testing.assert_array_equal(
                batch["image"].numpy(), i * np.ones((1, i + 1, i + 1))
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), i * np.ones((1, 12, 12))
            )

        dls = ds.pytorch(
            num_workers=2,
            batch_size=1,
            shuffle=True,
            use_local_cache=True,
        )
        pytorch_small_shuffle_helper(0, 16, dls)


@requires_torch
def test_groups(local_ds, compressed_image_paths):
    img1 = deeplake.read(compressed_image_paths["jpeg"][0])
    img2 = deeplake.read(compressed_image_paths["png"][0])
    with local_ds:
        local_ds.create_tensor(
            "images/jpegs/cats", htype="image", sample_compression="jpeg"
        )
        local_ds.create_tensor(
            "images/pngs/flowers", htype="image", sample_compression="png"
        )
        for _ in range(10):
            local_ds.images.jpegs.cats.append(img1)
            local_ds.images.pngs.flowers.append(img2)
    dl = local_ds.pytorch(return_index=False, num_workers=0)
    for cat, flower in dl:
        np.testing.assert_array_equal(cat[0], img1.array)
        np.testing.assert_array_equal(flower[0], img2.array)

    with local_ds:
        local_ds.create_tensor(
            "arrays/x",
        )
        local_ds.create_tensor(
            "arrays/y",
        )
        for _ in range(10):
            local_ds.arrays.x.append(np.random.random((2, 3)))
            local_ds.arrays.y.append(np.random.random((4, 5)))

    dl = local_ds.images.pytorch(return_index=False, num_workers=0)
    for sample in dl:
        cat = sample["jpegs/cats"]
        flower = sample["pngs/flowers"]
        np.testing.assert_array_equal(cat[0], img1.array)
        np.testing.assert_array_equal(flower[0], img2.array)


@requires_torch
def test_string_tensors(local_ds):
    with local_ds:
        local_ds.create_tensor("strings", htype="text")
        local_ds.strings.extend([f"string{idx}" for idx in range(5)])

    ptds = local_ds.pytorch(batch_size=1)
    for idx, batch in enumerate(ptds):
        np.testing.assert_array_equal(batch["strings"], [f"string{idx}"])

    ptds2 = local_ds.pytorch(batch_size=None)
    for idx, batch in enumerate(ptds2):
        np.testing.assert_array_equal(batch["strings"], f"string{idx}")


@pytest.mark.slow
@requires_torch
def test_pytorch_large(local_ds):
    arr_list_1 = [np.random.randn(1500, 1500, i) for i in range(5)]
    arr_list_2 = [np.random.randn(400, 1500, 4, i) for i in range(5)]
    label_list = list(range(5))

    with local_ds as ds:
        ds.create_tensor("img1")
        ds.create_tensor("img2")
        ds.create_tensor("label")
        ds.img1.extend(arr_list_1)
        ds.img2.extend(arr_list_2)
        ds.label.extend(label_list)

    ptds = local_ds.pytorch(num_workers=0)
    for idx, batch in enumerate(ptds):
        np.testing.assert_array_equal(batch["img1"][0], arr_list_1[idx])
        np.testing.assert_array_equal(batch["img2"][0], arr_list_2[idx])
        np.testing.assert_array_equal(batch["label"][0], idx)


def view_tform(sample):
    return sample


@pytest.mark.slow
@requires_torch
@pytest.mark.parametrize(
    "index,shuffle",
    [
        (slice(2, 7), True),
        (slice(3, 10, 2), False),
        (slice(None, 10), True),
        (slice(None, None, -1), False),
        (slice(None, None, -2), True),
        ([2, 3, 4], False),
        ([2, 4, 6, 8], True),
        ([2, 2, 4, 4, 6, 6, 7, 7, 8, 8, 9, 9, 9], True),
        ([4, 3, 2, 1], False),
        (3, True),
        (np.random.randint(0, 10, 100).tolist(), False),
    ],
)
def test_pytorch_view(local_ds, index, shuffle):
    arr_list_1 = [np.random.randn(15, 15, 5) for _ in range(10)]
    arr_list_2 = [np.random.randn(40, 15, 4, 2) for _ in range(10)]
    label_list = list(range(10))

    with local_ds as ds:
        ds.create_tensor("img1")
        ds.create_tensor("img2")
        ds.create_tensor("label")
        ds.img1.extend(arr_list_1)
        ds.img2.extend(arr_list_2)
        ds.label.extend(label_list)

    ptds = local_ds[index].pytorch(transform=view_tform, shuffle=shuffle, num_workers=0)
    idxs = list(IndexEntry(index).indices(len(local_ds)))
    for _ in range(2):
        for idx, batch in enumerate(ptds):
            idx = idxs[idx]
            if not shuffle:
                np.testing.assert_array_equal(batch["img1"][0], arr_list_1[idx])
                np.testing.assert_array_equal(batch["img2"][0], arr_list_2[idx])
                np.testing.assert_array_equal(batch["label"][0], idx)
    ptds = local_ds[index].pytorch(
        transform=view_tform,
        shuffle=shuffle,
        batch_size=2,
        drop_last=True,
        num_workers=0,
    )
    for _ in range(2):
        for batch in ptds:
            pass


@pytest.mark.slow
@requires_torch
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("buffer_size", [0, 10])
def test_pytorch_collate(local_ds, shuffle, buffer_size):
    local_ds.create_tensor("a")
    local_ds.create_tensor("b")
    local_ds.create_tensor("c")
    for _ in range(100):
        local_ds.a.append(0)
        local_ds.b.append(1)
        local_ds.c.append(2)

    ptds = local_ds.pytorch(
        batch_size=4,
        shuffle=shuffle,
        collate_fn=reorder_collate,
        buffer_size=buffer_size,
        num_workers=0,
    )
    for batch in ptds:
        assert len(batch) == 2
        assert len(batch[0]) == 2
        np.testing.assert_array_equal(batch[0][0], np.array([0, 0, 0, 0]).reshape(4, 1))
        np.testing.assert_array_equal(batch[0][1], np.array([1, 1, 1, 1]).reshape(4, 1))
        np.testing.assert_array_equal(batch[1], np.array([2, 2, 2, 2]).reshape(4, 1))


@pytest.mark.slow
@requires_torch
@pytest.mark.parametrize("shuffle", [True, False])
def test_pytorch_transform_collate(local_ds, shuffle):
    local_ds.create_tensor("a")
    local_ds.create_tensor("b")
    local_ds.create_tensor("c")
    for _ in range(100):
        local_ds.a.append(0 * np.ones((300, 300)))
        local_ds.b.append(1 * np.ones((300, 300)))
        local_ds.c.append(2 * np.ones((300, 300)))

    ptds = local_ds.pytorch(
        batch_size=4,
        shuffle=shuffle,
        collate_fn=my_transform_collate,
        transform=dict_to_list,
        buffer_size=10,
        num_workers=0,
    )
    for batch in ptds:
        assert len(batch) == 3
        for i in range(2):
            assert len(batch[i]) == 4
        np.testing.assert_array_equal(batch[0], 2 * np.ones((4, 300, 300)))
        np.testing.assert_array_equal(batch[1], 0 * np.ones((4, 300, 300)))
        np.testing.assert_array_equal(batch[2], 1 * np.ones((4, 300, 300)))


def run_ddp(rank, size, ds, q, backend="gloo"):
    import torch.distributed as dist
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=backend, rank=rank, world_size=size)

    s = 0
    for x in ds.pytorch(num_workers=0):
        s += int(x["image"][0].mean())

    q.put(s)


@pytest.mark.slow
@requires_torch
@pytest.mark.slow
def test_pytorch_ddp(local_ds):
    ds = local_ds
    import multiprocessing as mp

    with ds:
        ds.create_tensor("image", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(np.array([i * np.ones((10, 10)) for i in range(255)]))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            ds.pytorch(num_workers=0)
        return

    size = 2
    processes = []
    ctx = mp.get_context("spawn")
    q = ctx.Queue()

    for rank in range(size):
        p = ctx.Process(target=run_ddp, args=(rank, size, ds, q), daemon=True)
        p.start()
        processes.append(p)

    s = 0
    for p in processes:
        p.join()
        p.terminate()
        s += q.get()

    q.close()

    assert s == sum(list(range(254)))


def identity(x):
    return x


@requires_torch
@pytest.mark.slow
@pytest.mark.parametrize("compression", [None, "jpeg"])
def test_pytorch_decode(local_ds, compressed_image_paths, compression):
    ds = local_ds
    with ds:
        ds.create_tensor("image", sample_compression=compression)
        ds.image.extend(
            np.array([i * np.ones((10, 10, 3), dtype=np.uint8) for i in range(5)])
        )
        ds.image.extend([deeplake.read(compressed_image_paths["jpeg"][0])] * 5)
    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            ds.pytorch(num_workers=0)
        return

    for i, batch in enumerate(
        ds.pytorch(decode_method={"image": "tobytes"}, num_workers=0)
    ):
        image = convert_data_according_to_torch_version(batch["image"])
        assert isinstance(image, bytes)
        if i < 5 and not compression:
            np.testing.assert_array_equal(
                np.frombuffer(image, dtype=np.uint8).reshape(10, 10, 3),
                i * np.ones((10, 10, 3), dtype=np.uint8),
            )
        elif i >= 5 and compression:
            with open(compressed_image_paths["jpeg"][0], "rb") as f:
                assert f.read() == image

    if compression:
        ptds = ds.pytorch(
            decode_method={"image": "pil"}, collate_fn=identity, num_workers=0
        )
        for i, batch in enumerate(ptds):
            image = batch[0]["image"]
            assert isinstance(image, Image.Image)
            if i < 5:
                np.testing.assert_array_equal(
                    np.array(image), i * np.ones((10, 10, 3), dtype=np.uint8)
                )
            elif i >= 5:
                with Image.open(compressed_image_paths["jpeg"][0]) as f:
                    np.testing.assert_array_equal(np.array(f), np.array(image))


@requires_torch
def test_pytorch_decode_multi_worker_shuffle(local_ds, compressed_image_paths):
    with local_ds as ds:
        ds.create_tensor("image", sample_compression="jpeg")
        ds.image.extend(
            np.array([i * np.ones((10, 10, 3), dtype=np.uint8) for i in range(5)])
        )
        ds.image.extend([deeplake.read(compressed_image_paths["jpeg"][0])] * 5)
        for i, batch in enumerate(
            ds.pytorch(
                num_workers=2,
                shuffle=True,
                decode_method={"image": "pil"},
                collate_fn=identity,
            )
        ):
            image = batch[0]["image"]
            index = batch[0]["index"][0]
            assert isinstance(image, Image.Image)
            if index < 5:
                np.testing.assert_array_equal(
                    np.array(image), index * np.ones((10, 10, 3), dtype=np.uint8)
                )
            elif index >= 5:
                with Image.open(compressed_image_paths["jpeg"][0]) as f:
                    np.testing.assert_array_equal(np.array(f), np.array(image))


@requires_torch
def test_rename(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc")
        ds.create_tensor("blue/green")
        ds.abc.append([1, 2, 3])
        ds.rename_tensor("abc", "xyz")
        ds.rename_group("blue", "red")
        ds["red/green"].append([1, 2, 3, 4])
        loader = ds.pytorch(return_index=False, num_workers=0)
        for sample in loader:
            assert set(sample.keys()) == {"xyz", "red/green"}
            np.testing.assert_array_equal(
                np.array(sample["xyz"]), np.array([[1, 2, 3]])
            )
            np.testing.assert_array_equal(
                np.array(sample["red/green"]), np.array([[1, 2, 3, 4]])
            )


@requires_torch
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_indexes(local_ds, shuffle, num_workers):
    with local_ds as ds:
        ds.create_tensor("xyz")
        for i in range(8):
            ds.xyz.append(i * np.ones((2, 2)))

    ptds = local_ds.pytorch(
        return_index=True, batch_size=4, shuffle=shuffle, num_workers=num_workers
    )

    for batch in ptds:
        assert batch.keys() == {"xyz", "index"}
        for i in range(len(batch)):
            np.testing.assert_array_equal(batch["index"][i], batch["xyz"][i][0, 0])


def index_transform(sample):
    return sample["index"], sample["xyz"]


@requires_torch
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_indexes_transform(local_ds, shuffle, num_workers):
    with local_ds as ds:
        ds.create_tensor("xyz")
        for i in range(8):
            ds.xyz.append(i * np.ones((2, 2)))

    ptds = local_ds.pytorch(
        return_index=True,
        batch_size=4,
        shuffle=shuffle,
        num_workers=num_workers,
        transform=index_transform,
    )

    for batch in ptds:
        assert len(batch) == 2
        assert len(batch[0]) == 4
        assert len(batch[1]) == 4

        for i in range(4):
            np.testing.assert_array_equal(batch[0][i], batch[1][i][0, 0])


@requires_torch
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_indexes_transform_dict(local_ds, shuffle, num_workers):
    with local_ds as ds:
        ds.create_tensor("xyz")
        for i in range(8):
            ds.xyz.append(i * np.ones((2, 2)))

    ptds = local_ds.pytorch(
        return_index=True,
        batch_size=4,
        shuffle=shuffle,
        num_workers=num_workers,
        transform={"xyz": double, "index": None},
    )

    for batch in ptds:
        assert batch.keys() == {"xyz", "index"}
        for i in range(len(batch)):
            np.testing.assert_array_equal(2 * batch["index"][i], batch["xyz"][i][0, 0])

    ptds = local_ds.pytorch(
        return_index=True,
        batch_size=4,
        shuffle=shuffle,
        num_workers=num_workers,
        transform={"xyz": double},
    )

    for batch in ptds:
        assert batch.keys() == {"xyz"}


@requires_torch
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_indexes_tensors(local_ds, shuffle, num_workers):
    with local_ds as ds:
        ds.create_tensor("xyz")
        for i in range(8):
            ds.xyz.append(i * np.ones((2, 2)))

    with pytest.raises(ValueError):
        ptds = local_ds.pytorch(
            return_index=True,
            shuffle=shuffle,
            tensors=["xyz", "index"],
            num_workers=0,
        )

    ptds = local_ds.pytorch(
        return_index=True,
        batch_size=4,
        shuffle=shuffle,
        num_workers=num_workers,
        tensors=["xyz"],
    )

    for batch in ptds:
        assert batch.keys() == {"xyz", "index"}


def test_uneven_iteration(local_ds):
    with local_ds as ds:
        ds.create_tensor("x")
        ds.create_tensor("y")
        ds.x.extend(list(range(10)))
        ds.y.extend(list(range(5)))
        ptds = ds.pytorch(num_workers=0)
        for i, batch in enumerate(ptds):
            x, y = np.array(batch["x"][0]), np.array(batch["y"][0])
            np.testing.assert_equal(x, i)
            target_y = i if i < 5 else []
            np.testing.assert_equal(y, target_y)


def json_collate_fn(batch):
    import torch

    batch = [it["a"][0]["x"] for it in batch]
    return torch.utils.data._utils.collate.default_collate(batch)


def json_transform_fn(sample):
    return sample[0]["x"]


def list_collate_fn(batch):
    import torch

    batch = [np.array([it["a"][0], it["a"][1]]) for it in batch]
    return torch.utils.data._utils.collate.default_collate(batch)


def list_transform_fn(sample):
    return np.array([sample[0], sample[1]])


def test_pytorch_json(local_ds):
    ds = local_ds
    with ds:
        ds.create_tensor("a", htype="json")
        ds.a.append({"x": 1})
        ds.a.append({"x": 2})

    ptds = ds.pytorch(transform={"a": json_transform_fn}, batch_size=2, num_workers=0)
    batch = next(iter(ptds))
    np.testing.assert_equal(batch["a"], np.array([1, 2]))

    ptds = ds.pytorch(collate_fn=json_collate_fn, batch_size=2, num_workers=0)
    batch = next(iter(ptds))
    np.testing.assert_equal(batch, np.array([1, 2]))


def test_pytorch_list(local_ds):
    ds = local_ds
    with ds:
        ds.create_tensor("a", htype="list")
        ds.a.append([1, 2])
        ds.a.append([3, 4])

    ptds = ds.pytorch(transform={"a": list_transform_fn}, batch_size=2, num_workers=0)
    batch = next(iter(ptds))
    np.testing.assert_equal(batch["a"], np.array([[1, 2], [3, 4]]))

    ptds = ds.pytorch(collate_fn=list_collate_fn, batch_size=2, num_workers=0)
    batch = next(iter(ptds))
    np.testing.assert_equal(batch, np.array([[1, 2], [3, 4]]))


def test_pytorch_data_decode(local_ds, cat_path):
    ds = local_ds
    with ds:
        ds.create_tensor("generic")
        for i in range(10):
            ds.generic.append(i)
        ds.create_tensor("text", htype="text")
        for i in range(10):
            ds.text.append(f"hello {i}")
        ds.create_tensor("json", htype="json")
        for i in range(10):
            ds.json.append({"x": i})
        ds.create_tensor("list", htype="list")
        for i in range(10):
            ds.list.append([i, i + 1])
        ds.create_tensor("class_label", htype="class_label")
        animals = [
            "cat",
            "dog",
            "bird",
            "fish",
            "horse",
            "cow",
            "pig",
            "sheep",
            "goat",
            "chicken",
        ]
        ds.class_label.extend(animals)
        ds.create_tensor("image", htype="image", sample_compression="jpeg")
        for i in range(10):
            ds.image.append(deeplake.read(cat_path))

    decode_method = {tensor: "data" for tensor in list(ds.tensors.keys())}
    ptds = ds.pytorch(
        batch_size=1,
        num_workers=0,
        decode_method=decode_method,
        transform=identity,
        collate_fn=identity,
    )
    for i, batch in enumerate(ptds):
        sample = batch[0]
        assert sample["text"]["value"] == f"hello {i}"
        assert sample["json"]["value"] == {"x": i}
        assert sample["list"]["value"].tolist() == [i, i + 1]
        assert sample["class_label"]["value"] == [i]
        assert sample["class_label"]["text"] == [animals[i]]
        assert sample["image"]["value"].shape == (900, 900, 3)
        assert sample["generic"]["value"] == i
