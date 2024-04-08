import typer

from xrkit.enhancement import Enhancer


def main(
    techniques: str = typer.Option(
        "bilateral,clahe,dual,he,lhe,lime,tv",
        help="List of techniques to be applied. If not provided, use all available techniques.",
    ),
    n_samples: int = typer.Option(
        1000, help="Number of samples to be processed. If 0, processes all available samples."
    ),
    save_images: bool = typer.Option(False, help="Indicates whether the images should be saved or not."),
    generate_report: bool = typer.Option(
        False, help="Indicates whether the report should be generated or not."
    ),
):
    enhancer = Enhancer(n_samples=n_samples, save_images=save_images, generate_report=generate_report)

    for technique in techniques.split(","):
        enhancer.run(technique=technique)


if __name__ == "__main__":
    typer.run(main)
