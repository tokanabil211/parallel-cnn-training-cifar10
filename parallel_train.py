from mpi4py import MPI
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# === Load and normalize data (only once) ===
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# === Split training data among processes ===
x_split = np.array_split(x_train, size)
y_split = np.array_split(y_train, size)
x_local = x_split[rank]
y_local = y_split[rank]

# === Broadcast test data to all processes ===
x_test = comm.bcast(x_test, root=0)
y_test = comm.bcast(y_test, root=0)

# === Define model ===
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()

# === Train model ===
start_time = MPI.Wtime()
model.fit(x_local, y_local, epochs=10, batch_size=64, verbose=0)
end_time = MPI.Wtime()
train_time = end_time - start_time

# === Evaluate on local train and full test set ===
train_loss, train_acc = model.evaluate(x_local, y_local, verbose=0)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

# === Gather metrics at rank 0 ===
all_times = comm.gather(train_time, root=0)
all_train_acc = comm.gather(train_acc, root=0)
all_test_acc = comm.gather(test_acc, root=0)

if rank == 0:
    print("\n⏱️ Parallel Training Times:")
    for i, t in enumerate(all_times):
        print(f"  Rank {i}: {t:.2f} sec")

    print("\n📈 Train Accuracy per Process:")
    for i, acc in enumerate(all_train_acc):
        print(f"  Rank {i}: {acc:.4f}")

    print("\n📊 Test Accuracy per Process:")
    for i, acc in enumerate(all_test_acc):
        print(f"  Rank {i}: {acc:.4f}")

    print(f"\n✅ Avg Test Accuracy: {np.mean(all_test_acc):.4f}")
