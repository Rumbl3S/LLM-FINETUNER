import time
from sshtunnel import SSHTunnelForwarder
tunnel = None
def start_tunnel(
    remote_server,
    ssh_username,
    ssh_pkey,
    remote_port,
    local_port,
):
    global tunnel
    tunnel = SSHTunnelForwarder(
        (remote_server, 22),
        ssh_username=ssh_username,
        ssh_pkey=ssh_pkey,
        remote_bind_address=("localhost", remote_port),
        local_bind_address=("localhost", local_port),
    )
    tunnel.start()
    print(f"Tunnel opened at localhost:{tunnel.local_bind_port}")
def stop_tunnel():
    """
    Stops the SSH tunnel
    """
    global tunnel
    if tunnel:
        tunnel.stop()
        print("SSH tunnel closed")