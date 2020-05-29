import click
import tensorflow as tf


@click.group()
def cli():
    pass


@click.command()
@click.option('-i', '--input', help='Input tensorflow model', required=True)
@click.option('-o', '--output', help='Output path', default='.')
def convert(input, output):
    """Converts a tensorflow model to tensorflow lite file"""
    converter = tf.lite.TFLiteConverter.from_saved_model(input)
    tflite_model = converter.convert()
    open(output + "converted_model.tflite", "wb").write(tflite_model)


cli.add_command(convert)


if __name__ == "__main__":
    cli()
