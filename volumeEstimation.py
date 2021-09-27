import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# from keras.optimizers import Adam
# from keras.models import Model, Input
# from keras.callbacks import ModelCheckpoint
# from keras.layers import (
#     Dense, Conv1D, Conv2D, Activation, BatchNormalization,
#     Concatenate, Flatten, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (  # BatchNormalization, Concatenate,
    Dense, Conv1D, Conv2D, Activation, Flatten, Dropout)


def DenseLayer(x, units):
    x = Dense(
        units=units
        )(x)
    x = Dropout(0.7)(x)
    x = Activation("relu")(x)
    return x


def Conv2DLayer(x, filters):
    x = Conv2D(
        filters=filters,
        kernel_size=(5, 5),
        strides=2
        )(x)
    x = Activation("relu")(x)
    return x


def Conv1DLayer(x, filters):
    x = Conv1D(
        filters=filters,
        kernel_size=32,
        strides=4,
        )(x)
    x = Activation("relu")(x)
    return x


def CheckpointCallback(filepath):
    return ModelCheckpoint(
        filepath=filepath, save_weights_only=True,
        monitor='val_mae', save_best_only=True, verbose=1)


def TiNeuralNetworkModel():
    # The function prepares the neural network model
    # for volume estimation based on TI IWR6843 radar data.
    # ------------------------------------------------------------------------
    input = Input((64, 128, 1))
    layer = Conv2DLayer(input, 8)
    for filters in [16, 32, 64]:
        layer = Conv2DLayer(layer, filters)
    layer = Flatten()(layer)

    layer = DenseLayer(layer, 512)
    layer = DenseLayer(layer, 512)
    layer = Dropout(0.7)(layer)
    output = Dense(1, "linear")(layer)

    return Model(inputs=input, outputs=output)


def ANeuralNetworkModel():
    # The function prepares the neural network model
    # for volume estimation based on A111 radar data.
    # ------------------------------------------------------------------------
    input = Input((1860, 1))
    layer = Conv1DLayer(input, 8)
    for filters in [16, 32]:
        layer = Conv1DLayer(layer, filters)
    layer = Flatten()(layer)

    layer = DenseLayer(layer, 512)
    layer = DenseLayer(layer, 512)
    layer = Dropout(0.5)(layer)
    output = Dense(1, "linear")(layer)

    return Model(inputs=input, outputs=output)


def train_nn(dataSet, num_epochs, approach):
    #
    # (I) APPROACH -
    #     - CREATING VOLUME ESTIMATORS BASED ON CUMULATIVE VOLUMES
    #       OF SEQUENTIALY COLLECTED OBJECTS
    # (II) APPROACH -
    #     - CREATING VOLUME ESTIMATORS BASED ON REALSENSE DATA
    # ------------------------------------------------------------

    # Net volume
    cumulativeVolume = np.array([data["movingVolume"] for data in dataSet])

    # Gross volume
    # it's a volume from the sensors to the bin base.
    offset = 11.0 * 2.5 * 5.0
    refVolumeRealsense = offset - np.array(
        [data["realsense"]["depth"].sum()*10 * 3.4 * 5.4 / 100 / 80
            for data in dataSet])
    absoluteVolumes = refVolumeRealsense * 1000  # [cm3]

    tiHeatMaps = np.expand_dims([data["ti_heat_map"] for data in dataSet], -1)
    aData = np.expand_dims([data["a111_iq"] for data in dataSet], -1)

    # #    Preparing TI IWR6843 radar data to training process.
    # #    Neural network volume estimator training and estimation process.
    # # -------------------------------------------------------------------

    tiModel = TiNeuralNetworkModel()
    tiModel.compile(optimizer=Adam(0.000003), loss="mse", metrics=["mae"])

    if approach == 1:
        train_volume = cumulativeVolume
    else:
        train_volume = absoluteVolumes

    tiHistory = tiModel.fit(
        # The training data were collected in [0, 207) range
        # of the whole data set.
        x=tiHeatMaps[0:207], y=train_volume[0:207],
        batch_size=5,
        epochs=num_epochs,
        # The testing data were collected in [207, 248) range
        # of the whole data set.
        validation_data=(tiHeatMaps[207:248], train_volume[207:248]),
        callbacks=[CheckpointCallback(f"models/tiModelV{approach}")]
    )

    tiModel.load_weights(f"models/tiModelV{approach}")
    tiEstimatedVolumes = tiModel.predict(tiHeatMaps).reshape(-1)  # [cm3]

    # #    Preparing A1111 radar data for training process and neural network.
    # #    Neural network volume estimator training and estimation process.
    # # -----------------------------------------------------------------------

    aModel = ANeuralNetworkModel()
    aModel.compile(optimizer=Adam(0.000003), loss="mse", metrics=["mae"])

    aHistory = aModel.fit(
        # The training data were collected in [0, 207) range
        # of the whole data set.
        x=aData[0:207], y=train_volume[0:207],
        batch_size=5,
        epochs=num_epochs,
        # The testing data were collected in [207, 248) range
        # of the whole data set.
        validation_data=(aData[207:248], cumulativeVolume[207:248]),
        callbacks=[CheckpointCallback(f"models/aModelV{approach}")]
    )

    aModel.load_weights(f"models/aModelV{approach}")
    aEstimatedVolumes = aModel.predict(aData).reshape(-1)  # [cm3]

    show_train_curves(tiHistory, aHistory, approach)
    show_evaluation(dataSet, approach, tiEstimatedVolumes, aEstimatedVolumes)

    return tiHistory, tiEstimatedVolumes, aHistory, aEstimatedVolumes


def show_train_curves(tiHistory, aHistory, approach):
    # #   Training process metrics visualisation.
    # #   WARNING !!! Units of the metrics data will be recalculated
    # #   from cm and cm3 to dm and dm3.
    # # -----------------------------------------------------------------------

    plt.figure(figsize=(10, 10), dpi=300)
    plt.grid()

    plt.plot(
        np.array(tiHistory.history["mae"])/1000,
        label="Training MAE - TI IWR6843", alpha=0.5, c="blue"
    )
    plt.plot(
        np.array(tiHistory.history["val_mae"])/1000,
        label="Testing MAE - TI IWR6843", alpha=0.9, c="cyan"
    )
    plt.plot(
        np.array(aHistory.history["mae"])/1000,
        label="Training MAE - A111", alpha=0.5, c="orange"
    )
    plt.plot(
        np.array(aHistory.history["val_mae"])/1000,
        label=" Testing MAE - A111", alpha=0.9, c="red"
    )

    plt.ylabel("MAE [dm3]", fontsize=15)
    plt.xlabel("Epoch", fontsize=15)
    plt.legend(fontsize=15)
    plt.title(
        f"Neural network training curves for {approach}. approach.",
        fontsize=15)
    plt.savefig(
        f"plots/Neural network training curves for {approach}. approach.png")
    plt.ylim([0.0, 30])
    # plt.show()


def show_evaluation(dataSet, approach, tiEstimatedVolumes,
                    aEstimatedVolumes):
    # #   Preparing reference data from other sensors
    # #   and calculating the volume.
    # #   This data will be used in the second approach
    # #   of volume estimation process.
    # #   WARNINGS !!!
    # #       Units of the metrics data will be recalculated
    # #       from cm and cm3 to dm and dm3.
    # # -----------------------------------------------------------------------

    # it's a volume from the sensors to the bin base.
    # Net volume
    cumulativeVolume = np.array([data["movingVolume"] for data in dataSet])

    offset = 11.0 * 2.5 * 5.0
    refTrueVolumes = cumulativeVolume / 1000  # '/1000' cm3->dm3
    refVolumeDists = offset - np.array(
        [[data["infrared_range"], data["ultrasound_range"]]
            # distance cm->dm; '*2.5*5' the bin area [dm];
            for data in dataSet]) / 10 * 2.5 * 5  # '/10'
    refVolumeRealsense = offset - np.array(
        [data["realsense"]["depth"].sum()*10 * 3.4 * 5.4 / 100 / 80
            for data in dataSet])  # '/10' distance cm->dm; '*3.4*5.4' depth raster resolution [dm]; '/100/80' depth raster resolution [px];
    estVolumeTi = tiEstimatedVolumes / 1000  # '/1000' cm3->dm3
    estVolumeA = aEstimatedVolumes / 1000  # '/1000' cm3->dm3

    # maeVolumeTiV2 = np.abs(refTrueVolumes[248:] - estVolumeTiV1[248:]).mean()
    # maeVolumeAV2 = np.abs(refTrueVolumes[248:] - estVolumeAV1[248:]).mean()

    # #   Plots
    # # -----------------------------------------------------------------------

    def toTrigger(x):
        return np.vstack([x, x]).reshape(-1, order="F")

    for j, (minr, maxr, title) in enumerate([
        [207, 228, f"Volume estimation (1. test set - {approach}. approach)"],
        [228, 248, f"Volume estimation (2. test set - {approach}. approach)"],
        [248, 260, f"Volume estimation (1. validation set - {approach}. approach)"],
        [260, 285, f"Volume estimation (2. validation set - {approach}. approach)"],
        [285, 304, f"Volume estimation (3. validation set - {approach}. approach)"],
        [304, 317, f"Volume estimation (4. validation set - {approach}. approach)"],
        [317, 343, f"Volume estimation (5. validation set - {approach}. approach)"],
    ]):
        lenr = maxr-minr
        ticks = np.vstack([
            np.arange(0, lenr, 1),
            np.arange(1, lenr+1, 1)
        ]).reshape(-1, order="F")

        plt.figure(figsize=(10, 10), dpi=300)
        plt.plot(
            ticks,
            toTrigger(refTrueVolumes[minr:maxr]),
            c="k", label="Reference Volume - True", lw=4, alpha=0.7
        )
        plt.plot(
            ticks,
            toTrigger(refVolumeRealsense[minr:maxr]),
            c="orange", label="Reference Volume - RealSense", lw=3, alpha=0.7
        )
        plt.plot(
            ticks,
            toTrigger(refVolumeDists[minr:maxr, 0]),
            c="blue", label="Reference Volume - M18 ToF", lw=3, alpha=0.7
        )
        #
        # There was a problem with UltrasonicUK1D sensor measurements
        # when testing data was collected,
        # so we must skip UltrasinicUK1D in test process.
        # -------------------------------------------------------------------------
        if j > 1:
            plt.plot(
                ticks,
                toTrigger(refVolumeDists[minr:maxr, 1]),
                c="green", label="Reference Volume - Ultrasonic UK1D",
                lw=3, alpha=0.7
            )
        plt.plot(
            ticks,
            toTrigger(estVolumeTi[minr:maxr]),
            c="red", label="Estimated Volume - TI IWR6843", lw=3, alpha=0.7
        )
        plt.plot(
            ticks,
            toTrigger(estVolumeA[minr:maxr]),
            c="m", label="Estimated Volume - A111", lw=3, alpha=0.7
        )
        for i, label in enumerate([d["input"] for d in dataSet[minr:maxr]]):
            plt.text(i+0.2, -9, label, fontsize=15, rotation=90)
        plt.xticks(np.arange(0, lenr, 1))
        plt.plot([-10, 100], [60, 60], c="black", lw=3)
        plt.text(0.5, 63, 'Above the bin', fontsize=15, weight="bold")
        plt.text(0.5, 57, 'In the bin', fontsize=15, weight="bold")
        plt.xlabel("Trigger", fontsize=15)
        plt.ylabel("Volume [dm3]", fontsize=15)
        plt.legend(fontsize=15, loc=2)
        plt.grid(axis='x')
        plt.xlim([0, lenr])
        plt.ylim([-10, 120])
        plt.title(title, fontsize=15)
        plt.savefig(f"plots/{title}.png", dpi=300)
        # plt.show()

    plt.figure(figsize=(10, 10), dpi=300)
    plt.grid(axis="x", c="k", alpha=0.3)
    sns.distplot(
        refVolumeRealsense[248:] - refTrueVolumes[248:],
        hist=False, kde=True,
        kde_kws={'shade': True, 'linewidth': 3, "alpha": 0.2},
        color="orange", label="Reference Volume - RealSense"
    )
    sns.distplot(
        refVolumeDists[248:, 0] - refTrueVolumes[248:],
        hist=False, kde=True,
        kde_kws={'shade': True, 'linewidth': 3, "alpha": 0.2},
        color="blue", label="Reference Volume - M18 ToF"
    )
    sns.distplot(
        refVolumeDists[248:, 1] - refTrueVolumes[248:],
        hist=False, kde=True,
        kde_kws={'shade': True, 'linewidth': 3, "alpha": 0.2},
        color="green", label="Reference Volume - Ultrasonic UK1D"
    )
    sns.distplot(
        estVolumeTi[248:] - refTrueVolumes[248:],
        hist=False, kde=True,
        kde_kws={'shade': True, 'linewidth': 3, "alpha": 0.2}, color="red",
        label="Estimated Volume - TI"
    )
    sns.distplot(
        estVolumeA[248:] - refTrueVolumes[248:],
        hist=False, kde=True,
        kde_kws={'shade': True, 'linewidth': 3, "alpha": 0.2}, color="m",
        label="Estimated Volume - A111")
    plt.plot([0, 0], [0, 0.045], ls="--", lw=2, c="k")
    plt.xticks(np.arange(-100, 100, 10))
    plt.title(f"Error probability distribution ({approach}. approach, validation set)", fontsize=15)
    plt.ylabel("Density", fontsize=15)
    plt.xlabel("Volume error [dm3]", fontsize=15)
    plt.ylim([0.0, 0.040])
    plt.xlim([-60, 60])
    plt.legend(fontsize=15, loc=2)
    plt.savefig(f"plots/Error probability distribution ({approach}. approach, validation set.png", dpi=300)
    # plt.show()


if __name__ == "__main__":
    NUM_EPOCHS = 7000
    # # Loading the data set.
    # #    WARNING! Metrics data were collected using cm and cm3 units !!
    # # -----------------------------------------------------------------
    dataSet = []
    for i in range(1, 6):
        dataSet += np.load(f"data/dataset_pt{i}.npy", allow_pickle=True).tolist()
    print(len(dataSet))

    # (I) approach (net volume in training)
    train_nn(dataSet, NUM_EPOCHS, approach=1)

    # (II) approach (gross volume in training)
    train_nn(dataSet, NUM_EPOCHS, approach=2)

    # plt.show()  # show performance plots (saved to .png anyway)
