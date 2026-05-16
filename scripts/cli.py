import click

from scripts.commands.evaluate import evaluate

@click.group()
def main():
    pass


main.add_command(evaluate)