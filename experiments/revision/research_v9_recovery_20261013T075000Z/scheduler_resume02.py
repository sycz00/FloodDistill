from pathlib import Path
import scheduler_extended as extended
import common_v9 as c
scheduler=extended.scheduler
scheduler.__file__=__file__
old_read,old_sha=c.read,c.sha
def read(path,default=None):
    if Path(path)==c.ROOT/'manifest.json':path=c.ROOT/'manifest_resume02.json'
    return old_read(path,default)
def sha(path):
    if Path(path)==c.ROOT/'manifest.json':path=c.ROOT/'manifest_resume02.json'
    return old_sha(path)
c.read,c.sha=read,sha
if __name__=='__main__':scheduler.main()
