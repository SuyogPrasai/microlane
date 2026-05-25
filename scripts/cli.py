import click

from scripts.commands.evaluate import evaluate
from scripts.commands.summarize import summarize
from scripts.commands.comp import compare

@click.group()
def main():
    pass


main.add_command(evaluate)
main.add_command(summarize)
main.add_command(compare)