import click
import scripts.commands.compare as compare
import scripts.commands.evaluate as evaluate

@click.group()
def main() -> None:
    print("Hello, Microlane!")

main.add_command(compare.compare)
main.add_command(evaluate.evaluate)