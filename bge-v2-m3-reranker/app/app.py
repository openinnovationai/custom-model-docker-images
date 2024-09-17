from model import App


def init_app() -> App:
    return App.bind()


app = init_app()
