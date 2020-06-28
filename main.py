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
    """Tests given model's latency"""
    rasp.latency_test(input, count)


@click.command()
@click.option('-i', '--input', help='Input tensorflow lite model', required=True)
@click.option('-c', '--count', help='The number of inferences to perform for each class', required=True)
@click.option('-d', '--dataset', help="Classes to test",
              type=click.Choice(['leopards', 'lions', 'honeybee'], case_sensitive=False),
              multiple=True,
              required=True)
@click.option('-v', '--verbose', is_flag=True)
def raspacc(input, count, dataset, verbose):
    """Tests given model's accuracy"""
    rasp.accuracy_test(input, count, dataset, verbose)

    
cli.add_command(convert)
cli.add_command(rasprun)
cli.add_command(raspacc)


if __name__ == "__main__":
    cli()
