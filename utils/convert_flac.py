from tqdm import tqdm
import subprocess
import os
from os.path import exists, join
from utils.common_utils import get_list_of_files
from multiprocessing import Pool, freeze_support, RLock
import glob


def convert_flac_worker(pid, file_list, dst_dir):
    tqdm_text = "#" + "{}".format(pid).zfill(3)
    print(len(file_list))
    with tqdm(total=len(file_list), desc=tqdm_text, position=pid+1) as pbar:
        for fname in file_list:
            output_fname = join(dst_dir, "/".join(fname.split("/")[-3:])).replace(".flac", ".wav")
            output_dir = "/".join(output_fname.split("/")[:-1])
            if not exists(output_dir):
                os.makedirs(output_dir)
            out = subprocess.call('ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null' %
                                  (fname, output_fname), shell=True)
            if out != 0:
                raise ValueError('Conversion failed %s.' % fname)
            pbar.update(1)


def convert_flac_multi_process(src_dir, dst_dir, n_process=12):
    print("Get list of files...")
    file_list = glob.glob(join(src_dir, "*/*/*/*.flac"))
    print("Source path: %s" % src_dir)
    print("Destination: %s" % dst_dir)
    print("Total file: %d" % len(file_list))
    l = 1 + len(file_list) // n_process
    argument_list = [file_list[i * l: (i + 1) * l] for i in range(n_process)]

    pool = Pool(processes=n_process,
                initargs=(RLock(),),
                initializer=tqdm.set_lock)

    jobs = [pool.apply_async(convert_flac_worker, args=(i, n, dst_dir))
            for i, n in enumerate(argument_list)]
    pool.close()
    result_list = [job.get() for job in jobs]
    # Important to print these blanks
    print("\n" * (len(argument_list) + 1))


if __name__ == "__main__":
    convert_flac_multi_process("/home/messier/PycharmProjects/data/librispeech/flac",
                               "/home/messier/PycharmProjects/data/librispeech/wav")
