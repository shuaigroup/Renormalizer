
def test_ppp():
    with open("pppdump.txt", "w") as f:
        for iline in range(4):
            f.write("ppp")
        # hubbard
        for iorb in range(orbs):
            f.write(f"{}")
            

