import gdown
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Download dataset')
    parser.add_argument('-l', "--link", type=str, help="Link to google drive file")
    args = parser.parse_args()
    return args

def parse_link(link: str):
    base_url = "https://drive.google.com/uc?id="
    file_id = link.split("/")[5]
    return base_url + file_id

if __name__ == "__main__":
    link = parse_args().link
    parsed_link = parse_link(link)
    gdown.download(parsed_link)