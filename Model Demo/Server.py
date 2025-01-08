import socket
import os

# Define the pipeline function
def pipeline(audio_file_path):
    """
    Simulates running the audio file through a pipeline.
    For now, it returns "Real" if the word 'real' exists in the file name, otherwise "Fake".
    """
    print(f"Running pipeline on: {audio_file_path}")
    return "Real" if "real" in audio_file_path.lower() else "Fake"

def start_server(host='127.0.0.1', port=55449):
    """
    Starts the server to listen for incoming connections from the client.
    The server receives a file name and its content, processes the file through a pipeline,
    and sends the result back to the client.
    """
    # Create a socket for TCP communication
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))  # Bind the socket to the specified host and port
    server_socket.listen(1)  # Listen for incoming connections (only 1 at a time)
    print(f"Server listening on {host}:{port}...")

    while True:
        conn, addr = server_socket.accept()
        print(f"Connection from {addr} has been established.")

        try:
            # Receive the length of the file name (4 bytes)
            filename_length_data = conn.recv(4)
            if not filename_length_data:
                print("Failed to receive filename length.")
                continue

            # Convert the filename length to an integer
            filename_length = int.from_bytes(filename_length_data, 'big')
            print(f"Filename length: {filename_length}")

            # Receive the actual file name
            filename = conn.recv(filename_length).decode()
            print(f"Receiving file: '{filename}'")

            # Receive the file content and write it to a temporary file
            with conn:
                print(f"Connected by {addr}")

                # Receive the audio file
                received_data = b""
                while True:
                    chunk = conn.recv(1024)
                    if chunk == b"END":  # Check for the "END" signal
                        print("Received 'END' signal. File transfer complete.")
                        break
                    received_data += chunk

            # Save the received audio file
            with open("received_audio.wav", "wb") as f:
                f.write(received_data)
            print("Audio file received. Processing...")
            print(f"File '{filename}' received successfully.")

            # Run the received file through the pipeline
            result = pipeline(filename)

            # Send the result of the pipeline back to the client
            conn.sendall(result.encode())
            print(f"Result '{result}' sent to the client.")

        except Exception as e:
            # Handle any exceptions that occur during processing
            print(f"Error occurred: {e}")
        finally:
            # Ensure any temporary files are removed
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Temporary file '{filename}' deleted.")
            conn.close()
            print("Connection closed.")

if __name__ == "__main__":
    # Start the server when the script is run
    start_server()
