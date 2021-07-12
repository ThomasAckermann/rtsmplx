import numpy

if __name__ == "__main__":
    path = "../models/smplx/SMPLX_NEUTRAL.npz"
    data = numpy.load(path, allow_pickle=True)
    for k in data.files:
        print(k)
    # print(data[0])

