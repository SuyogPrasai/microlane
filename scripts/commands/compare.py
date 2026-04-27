import click

@click.command()
@click.argument('file')
def compare() -> None:
    print("Comparing results...")