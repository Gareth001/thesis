import logging
import coloredlogs


def configure_logging(verbose=True):
    level = logging.DEBUG if verbose else logging.INFO

    fmt = "[%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s" if verbose else "%(message)s"
    handlers = [logging.StreamHandler()]
    handlers[0].setLevel(level)
    logging.basicConfig(level=level, format=fmt, handlers=handlers)

    coloredlogs.install(
        level=level,
        fmt=fmt,
        reconfigure=True,
        level_styles=coloredlogs.parse_encoded_styles("debug=8;notice=green;warning=yellow;error=red,bold;critical=red,inverse"),
    )
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
