from data import DL17Lands

if __name__ == "__main__":
    print("#" * 50)
    print("Test MKM")
    print("=" * 10)
    dataloader = DL17Lands("MKM")

    print("#" * 50)
    print("Test OTJ")
    print("=" * 10)
    dataloader = DL17Lands()

    print("#" * 50)
