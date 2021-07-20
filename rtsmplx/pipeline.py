import smplx
import smplx.utils


if __name__ == "__main__":
    model = utils.create_model("../models")
    utils.plot_model(model)
    print(model)
