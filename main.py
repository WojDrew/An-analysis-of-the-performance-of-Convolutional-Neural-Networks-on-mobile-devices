import click

@click.command()
@click.option('--testArg', help='Sample CLI option')
def cli(name):
    print(name)
    return


if __name__ == "__main__":
    cli()
