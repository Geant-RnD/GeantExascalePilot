#!/usr/bin/env python

import os, sys, glob, shutil, argparse, subprocess as sp
from enum import Enum
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
FILE_ROOT = os.path.dirname(os.path.realpath(__file__))
CLANG_FORMAT_EXE = shutil.which('clang-format')
DEFAULT_PATHS = [ os.path.join('source', 'Geant') ]
FORMAT_MODE_OPTIONS = ['local', 'geant', 'custom', 'global']
CUSTOM_FORMAT_FILE = 'local-clang-format'
GLOBAL_FORMAT_FILE = 'geant-clang-format'
VERBOSE = 0
EXTENSIONS = ['.hh', '.cc', '.icc', '.hpp', '.cpp', '.tcc']


class FormatMode(Enum):
    GLOBAL = 0
    CUSTOM = 1


def get_format_mode(mode):
    """
    Determines the format mode
    """
    if mode is None:
        return None
    mode = ("{}".format(mode)).lower()
    if mode == 'local' or mode == 'custom':
        return FormatMode.CUSTOM
    elif mode == 'global' or mode == 'geant':
        return FormatMode.GLOBAL
    else:
        raise KeyError('Unknown format mode: {}. Valid options: {}', mode, FORMAT_MODE_OPTIONS)


def get_default_paths():
    """
    Provides the default path of the repository
    """
    paths = []
    for i in DEFAULT_PATHS:
        paths.append(os.path.join(REPO_ROOT, i))
    return paths


def get_worker_format_files(paths=[]):
    """
    Finds all .clang-format files in provided paths
    """
    if len(paths) == 0:
        paths += get_default_paths()
    files = []
    for i in paths:
        p = Path(os.path.join(REPO_ROOT, i))
        files.extend(list(p.glob('**/*.clang-format')))
    return files


def get_cxx_files(paths, extensions):
    """
    Globs for C++ files
    """
    if len(paths) == 0:
        paths += get_default_paths()
    files = []
    for i in paths:
        if not os.path.isabs(i):
            i = os.path.join(REPO_ROOT, i)
        p = Path(i)
        for ext in extensions:
            reg = '**/*{}'.format(ext)
            files.extend(list(p.glob(reg)))
    return files


def get_custom_format():
    """
    Get path to local clang-format
    """
    fname = os.path.join(FILE_ROOT, CUSTOM_FORMAT_FILE)
    if os.path.exists(fname):
        return fname
    else:
        msg = 'no custom format ("{}") file exists'.format(fname)
        raise FileNotFoundError(msg)


def get_global_format():
    """
    Get path to global custom format
    """
    fname = os.path.join(FILE_ROOT, GLOBAL_FORMAT_FILE)
    if os.path.exists(fname):
        return fname
    else:
        msg = 'no global format ("{}") file exists'.format(fname)
        raise FileNotFoundError(msg)


def copy_format(paths, format_mode):
    """
    Copy clang-format file to path based on mode
    """
    format_file = get_global_format(
    ) if format_mode == FormatMode.GLOBAL else get_custom_format()
    if len(paths) == 0:
        paths += get_default_paths()
    paths = list(set(paths))
    for p in paths:
        worker_format_file = os.path.join(p, '.clang-format')
        if VERBOSE > 0:
            print('Copying "{}" to "{}"...'.format(format_file, worker_format_file))
        shutil.copy2(format_file, worker_format_file)


def remove_format(paths):
    """
    Remove worker format paths
    """
    if len(paths) == 0:
        paths += get_worker_format_files()
    paths = list(set(paths))
    for p in paths:
        if VERBOSE > 0:
            print('Removing format file: "{}"...'.format(p))
        os.remove(p)


def get_clang_format_exe(path=None):
    return shutil.which('clang-format', path=path)


def apply_clang_format(target_file, inplace=False):
    """
    Apply clang-format to a file
    """
    err = ""
    out = ""
    try:
        cmd = [CLANG_FORMAT_EXE]
        if inplace:
            cmd += ["-i"]
        cmd += [os.fspath(target_file)]
        proc = sp.Popen(cmd, stdout=sp.PIPE)
        out, err = proc.communicate()
    except Exception as e:
        print('Error running clang-format on "{}"...\nProcess Error Message:{}\nException:{}'.format(target_file, err.decode("utf-8"), e))
    return out.decode("utf-8")


def verify(files, paths, extensions):
    """
    This function is used to validate that a local '.clang-format' that is introduced will not cause formatting changes
    """
    global_format = {}
    custom_format = {}
    revert_format = {}
    good_format = {}
    bad_format = {}

    cxx_files = []
    cxx_files += files
    if len(paths) > 0 or len(files) == 0:
        cxx_files += get_cxx_files(paths, extensions)

    tmpfiles = []
    for fname in cxx_files:
        if os.path.isdir(fname):
            continue
        fsrc = '{}'.format(fname)
        ftmp = fsrc.replace(REPO_ROOT, '/tmp')
        fdir = os.path.dirname(ftmp)
        if not os.path.exists(fdir):
            print('Making directory: {}'.format(fdir))
            os.makedirs(fdir, exist_ok=True)
        shutil.copy2(fname, ftmp)
        tmpfiles += [ftmp]
    cxx_files = tmpfiles

    format_dict = {}
    # create a dictionary of folder and their files
    for fname in cxx_files:
        dname = os.path.dirname(fname)
        if not dname in format_dict.keys():
            format_dict[dname] = [fname]
        else:
            format_dict[dname] += [fname]

    for dict_dir, dict_files in format_dict.items():
        # avoid multiple operations
        dict_files = list(set(dict_files))

        # notify which directory being worked on
        msg = 'Verifying "{}" '.format(dict_dir)
        sys.stdout.write(msg)
        sys.stdout.flush()
        counter = len(msg)

        # remove custom format in directory
        remove_format(get_worker_format_files([dict_dir]))

        # format with global
        #   for reference against reverted format after custom is applied
        copy_format([dict_dir], FormatMode.GLOBAL)
        for f in dict_files:
            # apply inplace
            apply_clang_format(f, inplace=True)
            global_format[f] = apply_clang_format(f)

        # format with custom
        copy_format([dict_dir], FormatMode.CUSTOM)
        for f in dict_files:
            custom_format[f] = apply_clang_format(f)
            apply_clang_format(f, inplace=True)

        # format with global
        #   see if un-applying custom format made persistent changes
        copy_format([dict_dir], FormatMode.GLOBAL)
        for f in dict_files:
            revert_format[f] = apply_clang_format(f)
            apply_clang_format(f, inplace=True)

        # check to see if the formatting was successfully reverted
        msg = ''
        for f in dict_files:
            counter += 1
            if revert_format[f] != global_format[f]:
                sys.stdout.write("!") # warn there was an error
                msg += '  - Warning! File: "{}" had formatting changes after applying custom format and then global format\n'.format(f.replace('/tmp', REPO_ROOT))
                good_format[f] = global_format[f]
                bad_format[f] = revert_format[f]
            else:
                sys.stdout.write(".") # no warning

            # flush the output
            sys.stdout.flush()

        # end the line
        sys.stdout.write("\n")

        # flush the output
        sys.stdout.flush()

        # print msg
        if msg:
            print("\n{}".format(msg))

        # remove custom format in directory
        remove_format(get_worker_format_files([dict_dir]))


    # Uh-oh
    if len(bad_format) > 0:
        forig = open(os.path.join(REPO_ROOT, 'format-original.txt'), 'w')
        for key, out in good_format.items():
            forig.write("\n\n{0}\n{1}\n{0}\n\n{2}\n".format('#'*80, key, "{}".format(out)))
        forig.close()

        fmod = open(os.path.join(REPO_ROOT, 'format-modified.txt'), 'w')
        for key, out in bad_format.items():
            fmod.write("\n\n{0}\n{1}\n{0}\n\n{2}\n".format('#'*80, key, "{}".format(out)))
        fmod.close()

        raise RuntimeError('Verification of custom format failed')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help="Enable clang-formatting to reflect Geant format (geant-clang-format) or custom format (custom-clang-format)",
                        choices=FORMAT_MODE_OPTIONS, default='global', type=str)
    parser.add_argument('--no-copy', help="Disable copy of clang-format file for the specified mode to provided paths",
                        action='store_true')
    parser.add_argument('-d', '--delete', help="Remove clang-format files not in repo root",
                        action='store_true')
    parser.add_argument('-v', '--verify', help="Verify custom/local format does not cause persistent changes",
                        action='store_true')
    parser.add_argument('-e', '--format-exe', help="Path to clang-format",
                        type=str, default=CLANG_FORMAT_EXE)
    parser.add_argument('-i', '--inplace', help="When applying clang-format, write formatting changes to file",
                        action='store_true')
    parser.add_argument('-g', '--glob', help="Glob folder paths for files ending in {}".format(EXTENSIONS),
                        action='store_true')
    parser.add_argument('--extensions', type=str, nargs='*', default=EXTENSIONS,
                        help="Override default extension for glob operations")
    parser.add_argument('paths', metavar='PATH', type=str, nargs='*',
                        help='Folder and/or file paths to operate on')
    parser.add_argument('-l', '--local-file', help="Path to local/custom clang-format file",
                        type=str, default=None)
    parser.add_argument('-V', '--verbose', help="Verbosity", type=int, default=VERBOSE)

    args = parser.parse_args()

    VERBOSE = args.verbose
    if args.local_file is not None:
        CUSTOM_FORMAT_FILE = args.local_file
    format_mode = get_format_mode(args.mode)

    if args.format_exe is None:
        raise FileNotFoundError('Error! No "clang-format" executable was found or specified!')

    CLANG_FORMAT_EXE = os.path.realpath(args.format_exe)
    print("Using '{}'...".format(CLANG_FORMAT_EXE))

    files = []
    dpaths = []
    fpaths = []
    for var in args.paths:
        # full path
        if not os.path.isabs(var):
            var = os.path.join(REPO_ROOT, var)
        # realpath
        var = os.path.realpath(var)

        if os.path.isfile(var):
            # if file, add to file list and append to path
            files.append(var)
            fpaths.append(os.path.dirname(var))
        elif os.path.isdir(var):
            # if does not exist, skip
            if not os.path.exists(var):
                print('Path: "{}" does not exist'.format(var))
                continue
            else:
                dpaths.append(var)
        else:
            print('Path "{}" is neither file nor directory'.format(var))

    # ensure that we aren't repeating work
    files = list(set(files))
    dpaths = list(set(dpaths))
    fpaths = list(set(fpaths))

    print('Files: {}')
    print('Paths: {}'.format(dpaths))
    print('')

    ret = 0
    try:
        # remove first
        if args.delete:
            remove_format(get_worker_format_files(dpaths + fpaths))

        # copy
        if not args.no_copy:
            copy_format(dpaths + fpaths, format_mode)

        # do verification
        if args.verify:
            verify(files, dpaths, args.extensions)
        else: # apply clang-formatting
            def apply_clang_format_to_file(f, inplace):
                msg = apply_clang_format(f, inplace)
                print('##### {} #####{}{}'.format(f, '\n' if msg else '', msg))

            if args.glob:
                for dpath in dpaths:
                    p = Path(dpath)
                    for ext in args.extensions:
                        files.extend(list(p.glob('**/*{}'.format(ext))))

            for f in files:
                apply_clang_format_to_file(f, args.inplace)

        # remove first
        if args.delete:
            remove_format(get_worker_format_files(dpaths + fpaths))

    except Exception as e:
        ret = 1
        print('Exception occurred: {}'.format(e))
        import traceback
        print("Exception in user code:")
        print('-'*60)
        traceback.print_exc(file=sys.stdout)
        print('-'*60)

    sys.exit(ret)
