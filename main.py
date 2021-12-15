from sodistec.core.detection import DetectPerson

def main():
    #  p1 = DetectPerson(".\\videos\\test.mp4")
    p1 = DetectPerson(0)
    p1.run()


if __name__ == '__main__':
    main()
