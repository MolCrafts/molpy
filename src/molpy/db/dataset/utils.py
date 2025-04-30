import io
import requests

def download(url:str, save_dir:str|None=None) -> io.BytesIO:
    """
    Download a file from a URL and save it to a specified directory.

    Args:
        url (str): The URL of the file to download.
        save_dir (str): The directory where the file will be saved.

    Returns:
        None
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    byte_content = response.content
    if save_dir is not None:
        with open(save_dir, 'wb') as f:
            f.write(byte_content)
    return io.BytesIO(byte_content)