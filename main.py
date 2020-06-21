import click
import rasp

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


@click.command()
@click.option('-i', '--input', help='Input tensorflow lite model', required=True)
@click.option('-c', '--count', help='The number of inferences to perform')
def rasprun(input, count):
    """To be used on a raspberry
    Performs a single random inference
    on a given model"""
    rasp.latency_test(input, count)

    
cli.add_command(convert)
cli.add_command(rasprun)


if __name__ == "__main__":
    cli()
