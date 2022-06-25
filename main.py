import os
from functools import partial
from zipfile import ZipFile

import gdown
import numpy
import tensorflow

from Models.StyleGan.StyleGan import StyleGAN
from Models.tools.plot_image import plot_images

START_RES = 4
TARGET_RES = 128


def log2(x):
    return int(numpy.log2(x))


batch_sizes = {2: 16, 3: 16, 4: 16, 5: 16, 6: 16, 7: 8, 8: 4, 9: 2, 10: 1}
# We adjust the train step accordingly
train_step_ratio = {k: batch_sizes[2] / v for k, v in batch_sizes.items()}


#os.makedirs("celeba_gan")

#url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
#output = "celeba_gan/data.zip"
#gdown.download(url, output, quiet=True)

#with ZipFile("celeba_gan/data.zip", "r") as zipobj:
#    zipobj.extractall("celeba_gan")

# Create a dataset from our folder, and rescale the images to the [0-1] range:

ds_train = tensorflow.keras.preprocessing.image_dataset_from_directory(
    "celeba_gan", label_mode=None, image_size=(64, 64), batch_size=32
)
ds_train = ds_train.map(lambda x: x / 255.0)


def resize_image(res, image):
    # only donwsampling, so use nearest neighbor that is faster to run
    image = tensorflow.image.resize(
        image, (res, res), method=tensorflow.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    image = tensorflow.cast(image, tensorflow.float32) / 127.5 - 1.0
    return image


def create_dataloader(res):
    batch_size = batch_sizes[log2(res)]
    # NOTE: we unbatch the dataset so we can `batch()` it again with the `drop_remainder=True` option
    # since the model only supports a single batch size
    dl = ds_train.map(partial(resize_image, res), num_parallel_calls=tensorflow.data.AUTOTUNE).unbatch()
    dl = dl.shuffle(200).batch(batch_size, drop_remainder=True).prefetch(1).repeat()
    return dl





style_gan = StyleGAN(start_res=START_RES, target_res=TARGET_RES)


def train(
    start_res=START_RES,
    target_res=TARGET_RES,
    steps_per_epoch=5000,
    display_images=True,
):
    opt_cfg = {"learning_rate": 1e-3, "beta_1": 0.0, "beta_2": 0.99, "epsilon": 1e-8}

    val_batch_size = 16
    val_z = tensorflow.random.normal((val_batch_size, style_gan.z_dim))
    val_noise = style_gan.generate_noise(val_batch_size)

    start_res_log2 = int(numpy.log2(start_res))
    target_res_log2 = int(numpy.log2(target_res))

    for res_log2 in range(start_res_log2, target_res_log2 + 1):
        res = 2 ** res_log2
        for phase in ["TRANSITION", "STABLE"]:
            if res == start_res and phase == "TRANSITION":
                continue

            train_dl = create_dataloader(res)

            steps = int(train_step_ratio[res_log2] * steps_per_epoch)

            style_gan.compile(
                d_optimizer=tensorflow.keras.optimizers.Adam(**opt_cfg),
                g_optimizer=tensorflow.keras.optimizers.Adam(**opt_cfg),
                loss_weights={"gradient_penalty": 10, "drift": 0.001},
                steps_per_epoch=steps,
                res=res,
                phase=phase,
                run_eagerly=False,
            )

            prefix = f"res_{res}x{res}_{style_gan.phase}"

            ckpt_cb = tensorflow.keras.callbacks.ModelCheckpoint(
                f"checkpoints/stylegan_{res}x{res}.ckpt",
                save_weights_only=True,
                verbose=0,
            )
            print(phase)
            style_gan.fit(
                train_dl, epochs=1, steps_per_epoch=steps, callbacks=[ckpt_cb]
            )

            if display_images:
                images = style_gan({"z": val_z, "noise": val_noise, "alpha": 1.0})
                plot_images(images, res_log2)



train(start_res=4, target_res=16, steps_per_epoch=20, display_images=False)