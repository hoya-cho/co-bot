import os
import json


__all__ = ['FileManager']


class FileManager:
    def __init__(self, path):
        self.path = path
        self._get_path_info(path=path)

    def _get_path_info(self, path):
        path_without_ext, ext = os.path.splitext(path)
        basename_without_ext = os.path.basename(path_without_ext)
        self.dir_path = os.path.dirname(path)
        self.basename_without_ext = basename_without_ext
        self.ext = ext
        self.fname = basename_without_ext + ext
        
    def get_parent_path(self, base_dir):
        base_dir = base_dir[:-1] if base_dir.endswith(os.sep) else base_dir
        base_dir_depth = len(base_dir.split(os.sep))
        parent_path = self.dir_path.split(os.sep)[base_dir_depth:]
        parent_path = os.path.join(*parent_path) if parent_path else ''
        parent_path = os.path.join(os.sep, parent_path)
        return parent_path

    @staticmethod
    def get_all_fpaths(src_dir, exts=['.png', '.jpg', '.jpeg']):
        fpaths = list()
        for current_path, dirnames, fnames in os.walk(src_dir, followlinks=True):
            if not fnames:
                continue
                
            for fname in fnames:
                fname_ext = FileManager(path=fname).ext
                if not fname_ext in exts:
                    continue
                fpath = os.path.join(current_path, fname)
                fpaths.append(fpath)
        return fpaths

    @staticmethod
    def json_dump(obj, dst):
        with open(dst, 'wt', encoding='utf-8') as j:
            json.dump(obj, j, indent=4, ensure_ascii=False)