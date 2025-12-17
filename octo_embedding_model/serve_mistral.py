import argparse
import sglang as sgl
from sglang.utils import wait_for_server

def main():
    parser = argparse.ArgumentParser(description="Serve Mistral model using sglang")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the Mistral model")
    parser.add_argument("--port", type=int, default=30000, help="Port to serve on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to serve on")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
    
    args = parser.parse_args()
    
    # Launch the server
    # We use python -m sglang.launch_server usually, but we can also use the Runtime API if preferred.
    # For simplicity and standard usage, we'll wrap the command line launch or use the Runtime.
    # However, sglang recommends using the command line for the server. 
    # Let's provide a wrapper script that uses subprocess to launch it, 
    # OR we can use the python API if we want to integrate it deeper.
    # Given the request is "serve mistral model using sglang", the Python Runtime is a good choice for programmatic control.
    
    print(f"Starting sglang server for model {args.model_path} on {args.host}:{args.port} with TP={args.tp}...")
    
    try:
        # Using sglang.Runtime to launch programmatically
        runtime = sgl.Runtime(
            model_path=args.model_path,
            port=args.port,
            host=args.host,
            tp_size=args.tp,
            # Add other sglang arguments as needed, e.g. schedule heuristic depending on use case
        )
        print(f"Server started at http://{args.host}:{args.port}")
        
        # Keep the main thread alive/wait for termination
        runtime.url # access to ensure it's up?
        
        # In a real script we might want to just block here. 
        # The runtime might run in background threads.
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Shutting down...")
        if 'runtime' in locals():
            runtime.shutdown()

if __name__ == "__main__":
    main()
