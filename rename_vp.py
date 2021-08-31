import glob
import os

if __name__=="__main__":
    for vid in glob.glob(f'{os.path.expanduser("~")}/Downloads/*.webm'):
        if any([f'arms{i}' in vid for i in range(3)]):
            # print(vid)
            # os.rename()
            [vp,date,user_id,arms,scenario] = vid.split('.')[0].split('_')
            # print(date,user_id,arms,scenario)

            # print(f"{vid} -> vp_{scenario}_{arms}_{user_id}.webm")
            # os.rename(vid, f"{vp}_{date}_{scenario}_{arms}_{user_id}.webm")