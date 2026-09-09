import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Odin Vision: servidor de memoria visual")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", choices=["demo", "models"], default=os.getenv("ODIN_BACKEND", "demo"))
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not os.getenv("ODIN_TOKEN"):
        parser.error("Configure ODIN_TOKEN antes de escuchar en una interfaz de red")
    os.environ["ODIN_BACKEND"] = args.backend
    import uvicorn
    uvicorn.run("odin_vision.server:create_app", factory=True, host=args.host, port=args.port,
                ws_max_size=2_000_000, ws_max_queue=1)


if __name__ == "__main__":
    main()
