import hub

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    ds = hub.load("hub://activeloop/glas-train")
    ds_test = hub.load("hub://activeloop/glas-test")

    print("Contenido del dataset: ")
    contents = ds.tensors.keys()
    print(', '.join(contents))

    print(f"\nCantidad de imágenes:")
    print(f"\tTrain: {ds.images.shape[0]}")
    print(f"\tTest: {ds_test.images.shape[0]}")

    image0 = ds.images[0].numpy()
    print(f"\nTamaño de las imágenes: {image0.shape}")

    for i in np.random.randint(0, ds.images.shape[0], (10,)):
        plt.subplot(1, 2, 1)
        plt.imshow(ds.images[int(i)].numpy())
        plt.subplot(1, 2, 2)
        plt.imshow(ds.masks[int(i)].numpy())
        plt.show()
        plt.close()