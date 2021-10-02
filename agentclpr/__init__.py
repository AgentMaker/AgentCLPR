from .infer import CLPSystem


def command():
    import argparse
    parser = argparse.ArgumentParser()

    # Command mode
    parser.add_argument(dest="mode", type=str)

    # Server config
    parser.add_argument("--host", type=str, default='127.0.0.1')
    parser.add_argument("--port", type=int, default=5000)

    args = parser.parse_known_args()[0]

    if args.mode == 'server':
        from gevent.pywsgi import WSGIServer
        from .infer.server import clp_server
        server = WSGIServer((args.host, args.port), clp_server)
        server.serve_forever()
    else:
        raise ValueError('Please check the mode.')
