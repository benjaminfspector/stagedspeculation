import time

def printheader(msg):
    # format message to start with " ----- " and end with " ----- "
    # the output will be at least 80 characters wide
    # which should be accomplished by padding the message on both side with spaces
    # odd number of characters
    pad = ' ' * ((80 - len(msg)) // 2)
    paddedmsg = pad + msg + pad
    if(len(paddedmsg) % 2 == 1): paddedmsg += ' '
    print('\n ----- ' + paddedmsg + ' -----\n')

def timefn(fn, header, nwarmup=3, niter=5, *args, **kwargs):
    printheader('WARMING UP '+header)
    for _ in range(nwarmup):
        fn(*args, **kwargs)
    printheader('STARTING '+header)
    t0 = time.time()
    for _ in range(niter):
        result = fn(*args, **kwargs)
    t1 = time.time()
    printheader('OUTPUTTING '+header)
    return result, t1-t0