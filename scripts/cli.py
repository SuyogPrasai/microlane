import click

from scripts.commands.evaluate import evaluate
from scripts.commands.summarize import summarize

@click.group()
def main():
    pass


main.add_command(evaluate)
main.add_command(summarize)