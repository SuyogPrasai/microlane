import click
import yaml

import scripts.commands.compare as compare
import scripts.commands.evaluate as evaluate
import scripts.commands.index as index

@click.group()
@click.pass_context
def main(ctx: click.Context) -> None:
    
    config_path = "configs/config.yaml"
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    ctx.obj = {"config": config}
    
main.add_command(compare.compare)
main.add_command(evaluate.evaluate)
main.add_command(index.index)