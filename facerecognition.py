import dlib, json, os, random, string, datetime, glob, cv2, time, threading, numpy
import multiprocessing as mp
from skimage import io
from scipy.spatial import distance

"""@Face recognition project
    Welcome to tiny dlib and opencv project done by Alex Kovrigin

    To analyze file:
        After line 'Waiting for input', type 'file' and image path.
        For example: 'file C:\somefolder\somefile.jpeg
    To analyze folder:
        After line 'Waiting for input', type 'dir ' and after that
        directory path with '\*.(file extension)'. For example:
        'dir C:\somefolder\*.jpg'
    To analyze Camera footage in real time:
        Just type 'cctv'
        To exit press ESC
    To end the program type '0' and the program will save footage and close
    Contact me a.kovrigin0@gmail.com, t.me/alex_kovrigin or alex.unaux.com
"""


def cctv(q=None, fps=25, calibration=False, delay=0):
    """
    Function is creating a window for user to watch and sends frames to q param, which is accessed in the maincode
    Pressing M makes you see the mirrored image, press M again to put everything back
    When destructing it saves video to folder

    :param q: Queue for returning frames
    :param fps: Fps setting, currently not in use
    :param calibration: Calibration, is used when data.json file is being created to determine the delay
    :param delay: Delay in camera recording it's different for every PC, because speed of processor is very different
    :return: In calibration returning delay, else returning void
    """
    vc = cv2.VideoCapture(0)
    mirrored = False

    if calibration:
        count = 0
        maxcount = fps * 5
        cv2.namedWindow("Calibration...")
        if vc.isOpened():  # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False
        now = int(round(time.time() * 1000))
        while rval and count < maxcount:
            cv2.imshow("Calibration...", frame)
            rval, frame = vc.read()
            key = cv2.waitKey(1000 // fps)
            count += 1
        cv2.destroyWindow("Calibration...")
        return ((int(round(time.time() * 1000)) - now) - 5000) // (fps * 5)
    cv2.namedWindow("Camera")
    if not os.path.exists('videos/'):
        os.makedirs('videos/')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    random_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])
    out = cv2.VideoWriter('videos/' + current_time() + '_' + random_name + '.avi', fourcc, float(fps), (640, 480))

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        if mirrored:
            show_frame = cv2.flip(frame, 1)
        else:
            show_frame = frame
        cv2.imshow("Camera", show_frame)
        q.put(frame)
        rval, frame = vc.read()
        out.write(frame)
        key = cv2.waitKey(1000 // fps - delay)
        if key == 27:  # exit on ESC
            break
        elif key == 77 or key == 109:
            mirrored = not mirrored
    cv2.destroyWindow("Camera")
    q.put('CCTV Closed')
    vc.release()
    out.release()


def current_time():
    """
    :return: Current time function returns string of current time
    """
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d %H-%M-%S')


def detect_frame(img):
    """
    Detect frame function, using dlib, recognizes humans and saving cropped images of their faces
    :param img: Frame which is being analyzed
    :return: In case of error returns
    """
    global database, sp, facerec
    original_img = img
    detector = dlib.get_frontal_face_detector()
    try:
        dets_webcam = detector(img, 1)
    except RuntimeError:
        return
    # dets_webcam = detector(img, 1)
    for k, d in enumerate(dets_webcam):
        print("Detection : Left: {} Top: {} Right: {} Bottom: {}".format(
            d.left(), d.top(), d.right(), d.bottom()))
        # dlib.hit_enter_to_continue()
        shape = sp(img, d)
        fb = facerec.compute_face_descriptor(img, shape)
        tr = (-1, 1000000)
        for q in range(len(database)):
            o = distance.euclidean(database[q]['image'], fb)
            if o < tr[1]:
                tr = (q, o)
        if (not tr[0] == -1) and tr[1] <= 0.6:
            print('Person recognised:', database[tr[0]]['name'],
                  ' (detection score:', str(tr[1]) + ', id=' + str(tr[0]) + ')')
            num = tr[0]
        else:
            win2 = dlib.image_window()
            win2.clear_overlay()
            try:
                win2.set_image(original_img)
            except RuntimeError:
                return
            win2.clear_overlay()
            win2.add_overlay(d)
            win2.add_overlay(shape)
            print('Hm... I have never seen this human. What is his name? ')
            name = str(input())
            print(name + '? Ok, now I know this human.')
            num = len(database)
            database.append(dict(image=list(fb), name=name, id=len(database)))
        images.append((original_img[d.top():d.bottom(), d.left():d.right()], num, current_time()))


def cycle_detect_frame():
    """
    Just a cycle for detect_frame() function which is on when cctv() is online
    :return: Returns nothing
    """
    global database, sp, facerec, cctv_closed, current_frame
    print('Recognition started')
    while not cctv_closed:
        time.sleep(0.01)
        detect_frame(current_frame)
    print('Recognition terminated')


def detect(path):
    """
    Extracts image from file and processes it through detect_frame() function
    :param path: Path of the analyzed file
    :return: Returns -1 if is incorrect path
    """
    if not os.path.isfile(path):
        print('Incorrect path!')
        return -1
    img = io.imread(path)
    return detect_frame(img)


# If it's just a thread miss main function
if __name__ == '__main__':
    print('Loading started...')

    # Reading data from file such as: dlib profiles for every person and paths to sp and facerec
    filename = 'data.json'
    if not os.path.isfile(filename):
        f = open(filename, 'w+')
        f.close()
    raw = open(filename)
    correctJson = True
    database = list()
    data = str()
    delay = 0

    # If data is valid json
    try:
        data = json.load(raw)
    except ValueError:
        correctJson = False
    if correctJson:
        # If valid json Load everything from there
        path_sp = data["sp"]
        path_facerec = data["facerec"]
        database = data["db"]
        delay = data["delay"]
    elif not os.path.getsize(filename):
        # If file is empty set manually
        path_sp = 'dats/shape_predictor_68_face_landmarks.dat'
        path_facerec = 'dats/dlib_face_recognition_resnet_model_v1.dat'
        delay = cctv(calibration=True)
        print('Video delay determined. ' + str(delay) + ' ms.')
    else:
        # If couldn't access file raise error
        print('Incorrect JSON syntax, see', filename)
        exit(1)

    # Load models from paths
    sp = dlib.shape_predictor(path_sp)
    facerec = dlib.face_recognition_model_v1(path_facerec)
    images = list()

    print('Loading complete!')

    # The input cycle
    while 1:
        print('Waiting for input...')

        # The command
        inp = str(input())

        # Stop command
        if inp == '0' or inp == 'break':
            break

        # Folder open command
        if inp[:6] == 'folder' or inp[:3] == 'dir':
            h = glob.glob(inp[inp.index(' ') + 1:])
            print('Files:', *h)
            for p in h:
                detect(p)

        # Camera use command
        elif inp[:4] == 'cctv' or inp[:6] == 'camera':
            current_frame = 'init'
            cctv_closed = False
            q = mp.Queue()
            camera = mp.Process(target=cctv, args=(q, delay,))
            camera.start()
            recognition = threading.Thread(target=cycle_detect_frame)
            recognition.start()
            while not cctv_closed:
                t = q.get()
                if (not isinstance(t, numpy.ndarray)) and t == 'CCTV Closed':
                    cctv_closed = True
                else:
                    current_frame = t

        # Help command
        elif inp[:4] == 'help':
            print(
                "To analyze file:",
                "   After line 'Waiting for input', type 'file' and image path.",
                "   For example: 'file C:\somefolder\somefile.jpeg"
                "To analyze folder:",
                "   After line 'Waiting for input', type 'dir ' and after that",
                "   directory path with '\*.(file extension)'. For example:",
                "   'dir C:\somefolder\*.jpg'",
                "To analyze Camera footage in real time:",
                "   Just type 'cctv'",
                "   To exit press ESC",
                "To end the program type '0' and the program will save footage and close",
                "Contact me a.kovrigin0@gmail.com, t.me/alex_kovrigin or alex.unaux.com", sep='\n')

        # File open command
        elif inp[:4] == 'file':
            detect(inp[5:])

        # If couldn't understand
        else:
            print("We couldn't understand what you meant. Type 'help' for help.")
    # Saving part
    print('Saving...')
    # Start creating json
    data = dict()
    data['db'] = database
    data['sp'] = path_sp
    data['facerec'] = path_facerec
    data['delay'] = delay
    # Write json data
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print('Profile data saved')

    # Save images
    # Create folder
    if not os.path.exists('images'):
        os.makedirs('images')
    for q in images:
        # For every image create a filename and try to save it in case of failure print failure statement
        name = database[q[1]]['name']
        dirname = name + '_' + str(q[1])
        if not os.path.exists('images/' + dirname):
            os.makedirs('images/' + dirname)
        wfilename = 'images/' + dirname + '/' + q[2] + '_' + ''.join(
            [random.choice(string.ascii_letters + string.digits) for n in range(10)])
        try:
            io.imsave(wfilename + '.jpeg', q[0])
        except:
            print('Failed to save', wfilename, 'Photo of', name)
    print('Image data saved')
    print('Saving complete!')
