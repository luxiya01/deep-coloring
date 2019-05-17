import cProfile, pstats, StringIO
import functools
import atexit

def profile(func):
    pr = cProfile.Profile()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        return result

    def _exit():
        s = StringIO.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats(30)
        print(s.getvalue())
    atexit.register(_exit)
    return wrapper
