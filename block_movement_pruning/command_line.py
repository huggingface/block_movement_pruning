import click

@click.group()
@click.pass_context
def cli(ctx):
    ctx.obj = {}

@cli.command()
@click.pass_context
@click.argument('path', default = None, type=click.Path(exists = True, resolve_path = True))
@click.argument('output', default = None, type=click.Path(resolve_path = True))
@click.option('--arg', '-a', is_flag = True)
def command1(ctx, path, output, arg):
    click.echo(click.style("Running !", fg="red"))
    #print(path + ":" + main(path, output))


def train_command():
    pass