import sys
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Console for logs (stdout)
console = Console()
# Console for progress bar (stderr)
progress_console = Console(file=sys.stderr)

def make_progress():
    return Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("{task.description}: {task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=progress_console
    )
