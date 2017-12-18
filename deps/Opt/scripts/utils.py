# utils.py
#
# Morgan's Python Utility Functions
#
# Ripped from icompile, modified to only use VS2013
#
from __future__ import print_function

import sys, string, os, os.path, fileinput, tempfile, shutil, re
import subprocess, pickle, time
try:
  from subprocess import getoutput
except ImportError:
  from commands import getoutput
try:
  import configparser
except ImportError:
  import ConfigParser as configparser

#############################################################################
# Verbosity levels
# Print only errors and prompts
QUIET                     = 10

NORMAL                    = 11

# Print complete commands
VERBOSE                   = 12

# Print additional debugging information
TRACE                     = 13

SUPERTRACE                = 14

verbosity                 = NORMAL

##############################################################################
#                              Color Printing                                #
##############################################################################

# Used by colorPrint
ansiColor = {
    'black'            :   "0",
    'red'              :   "1",
    'green'            :   "2",
    'brown'            :   "3",
    'blue'             :   "4",
    'purple'           :   "5",
    'cyan'             :   "6",
    'white'            :   "7",
    'defaultnounderscore' : "8",
    'default'          :   "9",}

ansiStyle = {
    'bold'             :   "1",
    'dim'              :   "2",  # Unsupported by most terminals
    'italic'           :   "3",  # Unsupported by most terminals
    'underline'        :   "4",
    'blink'            :   "5",  # Unsupported by most terminals
    'fastblink'        :   "6",  # Unsupported by most terminals
    'reverse'          :   "7",
    'hidden'           :   "8",  # Unsupported by most terminals
    'strikethrough'    :   "9"}  # Unsupported by most terminals

""" Used by colorPrint """
useColor = 'Unknown'

""" If the terminal supports color, prints in the specified color.  
    Otherwise, prints using normal color. The color argument
    must have the form:

    [bold|underline|reverse|italic|blink|fastblink|hidden|strikethrough] [FGCOLOR] [on BGCOLOR]

    COLOR = {default, black, red, green, brown, blue, purple, cyan, white}

    Yellow = light brown, Pink = light red, etc.

"""
def colorPrint(text, color = 'default'):
    print(colorize(text, color))
    sys.stdout.flush()
    
def colorize(text, color = 'default'):
    global useColor

    if useColor == 'Unknown':
        # Figure out if this device supports color
        useColor = (os.name != 'nt' and
                   ('TERM' in os.environ and
                    ((os.environ['TERM'] == 'xterm') or 
                     (os.environ['TERM'] == 'xterm-color'))))
   
    if not useColor:

        return text
        
    else:

        # Parse the color

        # First divide up into lower-case words
        tokens = color.lower().split(' ')
        if len(tokens) == 0:
            # Give up
            print(('Warning: illegal icompile color specified ("' +
                   color + '")\n\n'))
            useColor = False
            return text

        styleString     = ''
        foreColorString = ''
        backColorString = ''        

        if tokens[0] in ansiStyle:
            styleString = tokens[0]
            tokens = tokens[1:]

        if (len(tokens) > 0) and (tokens[0] != 'on'):
            # Foreground color
            foreColorString = tokens[0]
            tokens = tokens[1:]

        if len(tokens) > 0:
            # Background color, must start with 'on' keyword
            if (tokens[0] != 'on') or (len(tokens) < 2):
                # Give up
                useColor = False
                print(('Warning: illegal icompile background color' +
                       ' specified ("' + color + '")\n\n'))
                return text
            backColorString = tokens[1]

        foreDigit = '3'
        backDigit = '4'

        style     = ''
        foreColor = ''
        backColor = ''
    
        if styleString != '':
            style     = ansiStyle[styleString]

        if (foreColorString != '') and foreColorString in ansiColor:
            foreColor = foreDigit + ansiColor[foreColorString]

        if (backColorString != '') and backColorString in ansiColor:
            backColor = backDigit + ansiColor[backColorString]

        featureString = ''
        for s in [style, foreColor, backColor]:
            if (s != ''):
                if (featureString != ''):
                    featureString = featureString + ';' + s
                else:
                    featureString = s

        openCol = '\033['
        closeCol = 'm'
        stop = openCol + '0' + closeCol
        start = openCol + featureString + closeCol
        return start + text + stop


WARNING_COLOR = 'bold red'
ERROR_COLOR = 'bold red'
SECTION_COLOR = 'bold'
COMMAND_COLOR = 'green'

def printBar():
    print("_______________________________________________________________________\n")

def beep():
    print('\a')

##############################################################################
#                                 getch                                      #
##############################################################################

class _Getch:
    """Gets a single character from standard input.  Does not echo to the
    screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()
            
    def __call__(self): return self.impl()
            
            
class _GetchUnix:
    def __init__(self):
        import tty, sys
        
    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
            
            
class _GetchWindows:
    def __init__(self):
        import msvcrt
        
    def __call__(self):
        import msvcrt
        return msvcrt.getch()
        
        
getch = _Getch()


##############################################################################
#                               Shell Helpers                                #
##############################################################################

"""Create a directory if it does not exist."""
def mkdir(path, echo = True):
    if path[-1] == '/':
        path = path[:-1]

    if not os.path.exists(path) and (path != '.'):
        if echo: colorPrint('mkdir ' + path, COMMAND_COLOR)
        # TODO: set group and permissions from parent directory
        try:
            os.makedirs(path)
        except OSError:
            # There can be a race condition when two compile jobs try to
            # create the same directory at the same time.  If the directory
            # exists here, we can ignore the error
            if not os.path.exists(path):
                raise

##############################################################################

""" Turns a list of filenames into a list of at most four directories. """
def shortlist(L):
    num = len(L)
    if (num > 4):
        return L[0] + ', ' + L[1] + ', ' + L[2] + ', ' + L[3] + ', ...'
    if (num > 3):
        return L[0] + ', ' + L[1] + ', ' + L[2] + ', ...'
    if (num > 2):
        return L[0] + ', ' + L[1] + ', ...'
    elif (num > 1):
        return L[0] + ', ' + L[1]
    elif (num > 0):
        return L[0]
    else:
        return ''
    
##############################################################################
        
"""Recursively remove a directory tree if it exists."""
def rmdir(path, echo = True):
    if echo: colorPrint('rm -rf ' + path, COMMAND_COLOR)
    if (os.path.exists(path)):
        shutil.rmtree(path, 1)

""" Remove a single file, if it exists. """
def rm(file, echo = True):
    if (os.path.exists(file)):
        if echo: colorPrint('rm ' + file, COMMAND_COLOR)
        os.remove(file)

##############################################################################

""" Runs a program and returns a string of its output. """
def shell(cmd, printCmd = True):
    if printCmd: colorPrint(cmd, COMMAND_COLOR)
    
    if os.name == 'nt':
        # commands.getoutput is not supported on Win32, so we
        # must simulate it
        pipe = os.popen(cmd)
        result = pipe.read()
        pipe.close()
        return result
    else:
        return getoutput(cmd)

##############################################################################
        
"""Finds an executable on Unix.
Note: This is used only to better report errors, since spawn automatically
searches the PATH."""
def _findUnixBinary(program, path=None):
    if path == None:
        path = os.getenv('PATH', '').split(':')
    for directory in path:
        filename = pathConcat(directory, program)
        if os.path.isfile(filename):
            return filename
    return None

"""Finds an executable on Windows."""
def _findWindowsBinary(program):     
    # Paths that may contain the program
       
    PROGRAMFILES = os.getenv('PROGRAMFILES', '')
    PROGRAMFILESX86 = os.getenv('PROGRAMFILES(x86)', '')
    SYSTEMDRIVE = os.getenv('SystemDrive', '')

    PATH = [''] + os.getenv('PATH', '').split(';') + \
           ['.',\
           '../bin',\
           PROGRAMFILESX86 + '/MSBuild/12.0/Bin/',\
		   'C:/Windows/Microsoft.NET/Framework64/v4.0.30319',\
		   'C:/Windows/Microsoft.NET/Framework64/v3.5',\
           PROGRAMFILESX86 + '/Microsoft Visual Studio 12.0/Common7/IDE',\
           PROGRAMFILES + '/Microsoft Visual Studio 12.0/Common7/IDE',\
           PROGRAMFILESX86 + '/Microsoft Visual Studio 11.0/Common7/IDE',\
           PROGRAMFILES + '/Microsoft Visual Studio 11.0/Common7/IDE',\
           PROGRAMFILES + '/Java/jdk1.5.0_06/bin',\
           SYSTEMDRIVE + '/python',\
           SYSTEMDRIVE + '/doxygen/bin',\
           PROGRAMFILES + '/doxygen/bin',\
           PROGRAMFILES + '/PKZIP',\
           'bin']

    for path in PATH:
        filename = pathConcat(path, program)
        if os.path.exists(filename):
            return filename
            break

        filename = pathConcat(path, program) + '.exe'
        if os.path.exists(filename):
            return filename
            break

        filename = pathConcat(path, program) + '.com'
        if os.path.exists(filename):
            return filename
            break

    return None


"""Convert path separators to local style from Unix style.
   s is a string that contains a path name."""
def toLocalPath(s):
    return s.replace('/', os.sep)


#############################################################################

""" If the program name contains spaces, we
   add quotes around it. """
def safeWindowsName(p):
  if (' ' in p) and not ('"' in p):
    return '"' + p + '"'
  else:
    return p

#############################################################################

"""Run a program with command line arguments.

args must be a list of arguments (argv).  Spaces in arguments are *not* the same as having
separate list elements; these will not be re-parsed when they become the argv 
strings.

args must be a list.
Switches the slashes from unix to dos style in program.
Blocks until shell returns, then returns the exit code of the program.
"""
def run(program, args = [], echo = True, env = {}):
    windows = os.name == 'nt' or os.name == 'vista'
    
    # Windows doesn't support spawnvp, so we have to locate the binary
    if windows:
        original = program
        program = _findWindowsBinary(program)
        if not program: raise Exception('Cannot find "' + str(original) + '"')

    program = toLocalPath(program)
    argProgram = program

    if windows:
      argProgram = safeWindowsName(argProgram)
                    
    # spawn requires specification of argv[0]
    # Because the program name may contain spaces, we
    # add quotes around it.
    newArgs = [argProgram] + args

    newEnv = {}
    newEnv.update(os.environ)
    newEnv.update(env)

    if echo: colorPrint(' '.join(newArgs), COMMAND_COLOR)

    if windows:
        # Windows doesn't support spawnvpe
        exitcode = os.spawnve(os.P_WAIT, program, newArgs, newEnv)
    else:
        exitcode = os.spawnvpe(os.P_WAIT, program, newArgs, newEnv)
        if exitcode == 127:
            if _findUnixBinary(program) == None:
                raise Exception("Cannot find program " + program)
            else:
                colorPrint("Warning: possible error while executing program " +
                           program + ". spawn returned 127.", WARNING_COLOR)

    return exitcode

###########################################################################

"""Runs MSDEV (VC6) on the given dsw filename and builds the 
specified configs.  configs is a list of strings
"""
def msdev(filename, configs):
    binary = 'msdev'

    logfile = tempfile.mktemp()
    args = [filename]

    for config in configs:
        args.append('/MAKE')
        args.append('"' + config + '"')

    args.append('/OUT')
    args.append(logfile)

    x = run(binary, args)
  
    # Print the output to standard out
    for line in fileinput.input(logfile):
        print(line.rstrip('\n'))
 
    return x

###############################################################################

"""Runs DEVENV (visual studio IDE) on the given sld filename and builds the 
specified configs.  configs is a list of strings
"""
def devenv(filename, configs):
    binary = 'devenv'

    for config in configs:
        for target in ['debug', 'release']:
            logfile = tempfile.mktemp()
            args = [filename]

            args.append('/build')
            args.append(target)

            i = config.rfind('\\')
            if i >= 0: config = config[(i+1):]
            args.append('/project "' + config + '"')

            args.append('/out')
            args.append(logfile)

            x = run(binary, args)
  
            # Print the output to standard out
            for line in fileinput.input(logfile):
                print(line.rstrip('\n'))

            if x != 0:
                # Abort-- a build failed
                return x;
 
    return 0

##############################################
"""Runs VCExpress (VC10) on the given sln filename and builds the 
specified configs.  configs is a list of strings
"""
def VCExpress(filename, configs):
    binary = 'VCExpress'

    for config in configs:
        for target in ['debug', 'release']:
            logfile = tempfile.mktemp()
            args = [toLocalPath(filename)]

            args.append('/build')
            args.append(target)

			# Strip any leading path from config because VC12 doesn't use the
			# same syntax as msbuild
            i = config.rfind('\\')
            if i >= 0: config = config[(i+1):]

            args.append('/project "' + config + '"')

            args.append('/out')
            args.append(logfile)

            x = run(binary, args)
  
            # Print the output to standard out
            for line in fileinput.input(logfile):
                print(line.rstrip('\n'))

            if x != 0:
                # Abort-- a build failed
                return x;
 
    return 0


def MSBuild(filename, configs):
    binary = 'MSBuild'
    for config in configs:
        for target in ['Debug', 'Release']:
            args = [toLocalPath(filename)]

            args.append('/p:Configuration=' + target)
            # MSBuild converts periods in project name to underscores
            args.append('/t:"' + config.replace('.', '_') + '"')

            args.append('/p:Platform="x64"')

            args.append('/fl') # Log to msbuild.log

            x = run(binary, args)

            if x != 0:
                # Abort-- a build failed
                return x;
 
    return 0

###############################################################################
""" 
 Visual Studio dispatcher
"""
def VisualStudio(filename, configs):
     # Determine which version of MSVC is available

     if _findWindowsBinary('MSBuild'):
         return MSBuild(filename, configs)
      
     elif _findWindowsBinary('devenv'):
         # found Visual C++ Standard/Pro
         return devenv(filename, configs)
     
     elif _findWindowsBinary('VCExpress'):
         # found Visual C++ Express
         return VCExpress(filename, configs)
         
     else:
         print("Failed to find Visual Studio. Could not continue.")
         return -1

###########################################################################

"""Run a program with command line arguments.

args must be a list of arguments (argv).  Spaces in arguments are *not* the same as having
separate list elements; these will not be re-parsed when they become the argv 
strings.

args must be a list.
Switches the slashes from unix to dos style in program.
Blocks until shell returns, then returns the (exit code, stdout text, stdin text) of the program.
Environment defaults to the current one if not specified.
"""
def runWithOutput(prog, args = [], echo = True, env = None):
    windows = os.name == 'nt' or os.name == 'vista'
    
    program = toLocalPath(prog)

    # Windows doesn't support spawnvp, so we have to locate the binary
    if windows:
        program = _findWindowsBinary(program)
        if not program: raise Exception('Cannot find "' + program + '"')

    newArgs = [program] + args

    messages = ''
    if echo: messages += colorize(' '.join(newArgs), COMMAND_COLOR) + '\n'

    newEnv = {}
    newEnv.update(os.environ)
    if env is not None:
      newEnv.update(env)

    outPipe = subprocess.PIPE
    inPipe = None
    errPipe = subprocess.PIPE
    preexec = None
    proc = subprocess.Popen(newArgs, 0, None, inPipe, outPipe, errPipe, preexec,
                       False, False, None, newEnv, True)

    (out, err) = proc.communicate()
    return (proc.returncode, messages + out, err)


""" Returns the current processor count.
   From processing <http://cheeseshop.python.org/pypi/processing/0.34>
"""
def cpuCount():
    num = 1
    
    if sys.platform == 'win32':
        try:
            num = int(os.environ['NUMBER_OF_PROCESSORS'])
        except (ValueError, KeyError):
            pass
    elif sys.platform == 'darwin':
        try:
            num = int(os.popen('sysctl -n hw.ncpu').read())
        except ValueError:
            pass
    else:
        try:
            num = os.sysconf('SC_NPROCESSORS_ONLN')
        except (ValueError, OSError, AttributeError):
            pass

    if num >= 1:
        return num
    else:
        return 1

##############################################################################

"""Returns 0 if the file does not exist, otherwise returns the modification
   time of the file in the same form as time.time()."""
def getTimeStamp(file):
   try:
       t = os.path.getmtime(file)
       if t > time.time():
           colorPrint('Warning: ' + file +
                      ' time stamp is in the future (' + time.ctime(t) + ')', WARNING_COLOR)
       return t
   except OSError:
       return 0


""" Like getTimeStamp, but uses the specified cache. """
def getTimeStampCached(file, cache):
    if file not in cache:
        cache[file] = getTimeStamp(file)
    return cache[file]

##############################################################################

"""Determine if a target is out of date.

Returns nonzero if file1 is newer than file2.
Throws an error if file1 does not exist, returns
nonzero if file2 does not exist."""
def newer(file1, file2):
   time1 = os.path.getmtime(file1)
   time2 = 0
   try:
       time2 = os.path.getmtime(file2)
   except OSError:
       time2 = 0
       
   return time1 >= time2


""" Removes quotation marks from the outside of a string. """
def removeQuotes(s):
    if (s[1] == '"'):
        s = s[2:]
    if (s[(len(s)-2):] == '"'):
        s = s[:len(s)-2]
    return s

###############################################################################

"""
  verInfo: A string containing (somewhere) a version number.  Typically, the
  output of commands.getoutput().  Returns the version as a list of version
  numbers.
"""
def findVersionInString(verInfo):

    # Look for a number followed by a period.
    for i in range(1, len(verInfo) - 1):
        if (verInfo[i] == '.' and 
           (verInfo[i - 1]).isdigit() and 
           (verInfo[i + 1]).isdigit()):

            # We've found a version number.  Walk back to the
            # beginning.
            i -= 2
            while (i > 0) and verInfo[i].isdigit():
                i -= 1
            i += 1

            version = []
            while (i < len(verInfo)) and verInfo[i].isdigit():
                d = ''

                # Now walk forward
                while (i < len(verInfo)) and verInfo[i].isdigit():
                    d += verInfo[i]
                    i += 1

                version.append(int(d))

                # Skip the non-digit
                i += 1
           
            return version     

    return [0]

###############################################################################

""" Takes a version list and converts it to a string."""
def versionToString(v):
    s = ''
    for i in v:
        s += str(i) + '.'
    return s[:-1]

##############################################################################

def removeTrailingSlash(s):
    if (s[-1] == '/'):
        s = s[:-1]
    elif (s[-1] == '\\'):
        s = s[:-1]
    return s

def addTrailingSlash(s):
    if s.endswith('\\') or s.endswith('/'):
        return s
    elif s.endswith(':'):
        # Win32 drive spec
        return s + '\\'
    elif s == '':
        # Empty dir
        return './'
    else:
        return s + '/'


"""
Strips the path from the front of a filename.

os.path.basename strips one extra character from the beginning.
This restores it.
"""
def betterbasename(filename):
    # Find the index of the last slash
    i = max(filename.rfind('/'), filename.rfind('\\'))

    # Copy from there on (whole string if no slashes)
    return filename[(i + 1):]


""" Returns the part of a full filename after the path and before the last ext"""
def rawfilename(filename):
    f = betterbasename(filename)
    period = f.rfind('.')

    if period > 0:
        return f[0:period]
    else:
        return f


""" Returns the extensions from a full filename."""
def extname(filename):

    f = betterbasename(filename)
    period = f.rfind('.')

    if period > 0:
        return f[(period + 1):]
    else:
        return ''

""" Given a library filename, returns the name that should be passed to a linker, 
    e.g., /usr/lib/libfoo-1.1.so -> foo-1.1"""
def rawLibraryFilename(filename):
    n = rawfilename(filename)
    if n.startswith('lib'):
        n = n[3:]
    return n


# Concatenates unless b is already absolute
def maybePathConcat(a, b):
    if b.startswith('/') or ((len(b) > 3) and 
                             (b[2:3] == ':\\') or b.startswith('\\')):
        return b
    else:
        return pathConcat(a, b)


def _pathConcat2(a, b):
    if len(b) == 0:
        return a
 
    if len(a) == 0:
        return b

    # remove any leading slash from b
    if ((b[0] == '/') or
        (b[0] == '\\')):
       b = b[1:]

    if ((a != '') and
        (a[-1] != '/') and
        (a[-1] != '\\')):
        return a + '/' + b
    else:
        return a + b

"""
 Concatenates a file or path onto a path with a '/' if the first
 is non-empty and lacks a '/'
"""
def pathConcat(*args):
    if len(args) == 2:
        return _pathConcat2(args[0], args[1])
    else:
        current = ''
        for x in args:
            current = _pathConcat2(current, x)
        return current
    

""" Returns the version number of a file as a list.  Note that under
comparison, 1.10 != 1.1 and 1.01 == 1.1, which is usually what you
want.
"""
def getVersion(filename):
    cmd = filename

    base = betterbasename(filename)

    # We check only the beginning of a filename because it may have
    # a version number as part of the name.
    if base == 'cl':
        cmd = filename
    elif base.startswith('g++'):
        cmd = filename + ' --version'
    elif base.startswith('clang++'):
        cmd = filename + ' --version'
    elif base.startswith('clang'):
        cmd = filename + ' --version'
    elif base.startswith('python'):
        cmd = filename + ' -V'
    elif base.startswith('cl'):
        # MSVC++ compiler
        cmd = filename
    elif base.startswith('doxygen'):
        cmd = filename + ' --version'
    elif base.startswith('ar'):
        cmd = filename + ' --version'
    elif base.startswith('make'):
        cmd = filename + ' -v'
    elif base.startswith('ld'):
        cmd = filename + ' --version'
    else:
        # Unsupported
        return [0, 0, 0]
    
    return findVersionInString(getoutput(cmd))

def maybeColorPrint(text, color = 'default'):
    if verbosity >= VERBOSE:
        colorPrint(text, color)

""" Prints a line if quiet is False. """
def maybePrintBar():
    if verbosity >= VERBOSE:
        printBar()

""" Returns a list of all directories (without '..') that are
    next to this directory.
"""
def getSiblingDirs(howFarBack = 1):
    siblings = []
    me = betterbasename(os.getcwd())

    prefix = '..'

    i = 0
    while i < howFarBack:
        for node in os.listdir(prefix):
            fullname = prefix + '/' + node
            # See if the node is a directory (and not *this* directory!)
            if ((i > 0) or (node != me)) and os.path.isdir(fullname):
                siblings.append(fullname)
        i += 1
        prefix += '/..'

    return siblings

""" Returns the index of x in list L, starting at start.  Returns -1 if not found."""
def find(L, x, start = 0):
    i = start
    while i < len(L):
        if L[i] == x:
            return i
        i += 1
    return -1   

##############################################################################
#                              Locate Compiler                               #
##############################################################################

_preferredCompiler = None

"""  Used by newestCompiler"""
def _newestCompilerVisitor(best, dirname, files):
    for file in files:
        if (_preferredCompiler == file[:len(_preferredCompiler)]):
            # Form of file is g++-VERSION or g++VERSION
            try:
                ff = dirname + "/" + file        
                v = getVersion(ff)
                
                if v > best[1]:
                    best[0] = ff
                    best[1] = v

            except ValueError:
                pass

_newestCompilerFilename = None
_newestCompilerVersion  = None
 
"""AI for locating the latest version of g++ or clang++
   Returns the full path to the program (including the program name),
   the version as a list, and the common name (e.g., vc8) for the compiler. 
"""
def newestCompiler():
    global _newestCompilerFilename, _newestCompilerVersion, _preferredCompiler
 
    if _newestCompilerFilename == None:
        # Filename has not been cached; compute it for the first time

        if os.name == 'nt':
            # Windows
 
            vsDir = 'C:/Program Files/Microsoft Visual Studio 8'
            _newestCompilerFilename = pathConcat(vsDir, 'VC/bin/cl.exe')
              
            if not os.path.exists(_newestCompilerFilename):
                # TODO: look at the PATH variable
                print('Error: could not find Visual Studio 8 Compiler at at \'' + _newestCompilerFilename + '\'')
                sys.exit(-1)

            if ('VSINSTALLDIR' not in os.environ or
                (os.path.normpath(os.environ['VSINSTALLDIR']) != os.path.normpath(vsDir))):
                print('Error: you must run vsvars32.bat before iCompile')
                sys.exit(-1)

            _newestCompilerVersion  = getVersion(_newestCompilerFilename)


        else:
            # Unix-like system, use clang/llvm
            #_preferredCompiler = 'g++'
            _preferredCompiler = 'clang++'

            bin = getoutput('which ' + _preferredCompiler)
    
            # Turn binLoc into just the directory, not the path to the file g++
            binLocs = [bin[0:bin.rfind('/')]]
            
            # best will keep track of our current newest g++/clang++ found
            best = [bin, getVersion(bin)]

            # Search for all compiler binaries
            for path in binLocs:
                for dirpath, dirnames, filenames in os.walk(path):
                      _newestCompilerVisitor(best, dirpath, filenames)

            _newestCompilerFilename = best[0]
            _newestCompilerVersion  = best[1]

    return (_newestCompilerFilename, _newestCompilerVersion)

#############################################################

def getCompilerNickname(compilerFilename):
    base = betterbasename(compilerFilename)

    if (os.name == 'nt') and base.startswith('cl'):

       # Windows Visual Studio
       verString = shell('"' + compilerFilename.replace('/', '\\') + '"', False)

       if verString.startswith('Microsoft (R) 32-bit C/C++ Optimizing ' +
                               'Compiler Version 16.'):
           return 'vc10.0'

       if verString.startswith('Microsoft (R) 32-bit C/C++ Optimizing ' +
                               'Compiler Version 15.'):
           return 'vc9.0'
 
       elif verString.startswith('Microsoft (R) 32-bit C/C++ Optimizing ' +
                               'Compiler Version 14.'):
           return 'vc8.0'
 
       elif verString.startswith('Microsoft (R) 32-bit C/C++ Optimizing ' + 
                                 'Compiler Version 13.'):
           return 'vc7.1'
  
       elif verString.startswith('Microsoft (R) 32-bit C/C++ Optimizing ' +
                                 'Compiler Version 12.'):
 
           return 'vc6.0'

       else:
           # Not a recognized compiler!
           return 'unknown'
    else:

       # Unix

       v = getVersion(compilerFilename)
       if len(v) > 2:
           v = v[0:2]
       if len(v) < 2:
           # Version number was short; add a 0 minor number
           v = v + [0]
       version = '.'.join(list(map(str, v)))

       if base.startswith('g++') or base.startswith('gcc'):
           base = base[0:3]

       name = base

       return name + version
         
#############################################################

""" List all directories in a directory """
def listDirs(_dir = ''):
    if (_dir == ''):
        dir = './'
    else:
        dir = _dir

    all = os.listdir(dir)
    dirs = []
    for d in all:
        if os.path.isdir(d):
            dirs.append(_dir + d)

    return dirs

########################################################
""" Turns a string with paths separated by ; (or : on Linux) into
    a list of paths each ending in /."""
def makePathList(paths):
    if os.name == 'posix':
        # Allow ':' as a separator between paths
        paths = paths.replace(':', ';')
        
    return cleanPathList(paths.split(';'))


""" Ensures that every string in a list ends with a trailing slash,
    is non-empty, and appears exactly once.

    Preserves the order of the input list.
    """
def cleanPathList(paths):
    out = []

    for path in paths:
        # Strip surrounding quotes
        if path.startswith('\"') and path.endswith('\"'):
            path = path[1:-1]
        elif path.startswith('\'') and path.endswith('\''):
            path = path[1:-1]
            
        if path == '':
            # do nothing
            0
        else:
            # Append trailing slash
            if path[-1] != '/':
                path += '/'

            # Only add paths not already in the list
            if not path in out:
                out.append(path)

    return out

##################################################################

"""
"""
def shortname(prefix, cfile):
    if cfile.startswith(prefix):
        # Don't bother printing the root directory name
        # when it appears
        return cfile[len(prefix):]
    else:
        return cfile

#########################################################################

""" Returns true if this is a cpp source filename. """
def isCFile(file):
    ext = extname(file).lower()

    isOSX = (os.name != 'nt') and (os.uname()[0] == 'Darwin')

    return ((ext == 'cpp') or
           (ext == 'c') or
           (ext == 'c++') or
           (ext == 'cc') or
           (ext == 'cxx') or
           (ext == 'i') or
           (ext == 'ii') or
	   (isOSX and
            ((ext == 'mm') or
             (ext == 'm') or
             (ext == 'mi') or
             (ext == 'mii'))))

""" Returns true if this is a cpp header filename (or doxygen file). """
def isCHeader(file):
    ext = extname(file).lower()
    return (ext == 'h') or (ext == 'hpp') or (ext == 'h++') or (ext == 'dox')

def isObjectiveC(file):
  return extname(file).lower() in ['mm', 'm', 'mi', 'mii']


#########################################################################

"""
A regular expression matching files that should be excluded from compilation
"""
excludeFromCompilation = None
_includeHeaders = False

def _listCFilesVisitor(result, dirname, files, subdirectories):
    dir = dirname

    # Strip any unnecessary "./"
    if (dirname[:2] == "./"):
        dir = dir[2:]

    if ((excludeFromCompilation != None) and
        ((excludeFromCompilation.search(dir) != None) or
         (excludeFromCompilation.search(betterbasename(dir))))):
        # Don't recurse into subdirectories of excluded directories either
        del subdirectories[:]
        return

    # We can't modify files while iterating through it, so
    # we must make a list of all files that are to be removed before the
    # next iteration of the visitor.   
    removelist = [];
    for f in files:
         if ((excludeFromCompilation != None) and
             (excludeFromCompilation.search(f) != None)):
            if verbosity >= SUPERTRACE: print("  Ignoring '" + f + "'")
            removelist.append(f)
            
         elif isCFile(f) or (_includeHeaders and isCHeader(f)):

             if ((excludeFromCompilation == None) or
                 (excludeFromCompilation.search(f) == None)):
                 # Ensure the path ends in a slash (when needed)
                 filename = pathConcat(dir, f)
                 result.append(filename)

    # Remove any subdir in 'files' that is itself excluded so as to prevent
    # later recursion into it
    for f in removelist:
        files.remove(f)


"""Returns all files with gcc-recognized C/C++ endings for the given directory
   and all subdirectories.
   
   Filenames must be relative to the "rootDir" directory.  dir will be
   a subdirectory of rootDir.

   exclude must be a regular expression for files to exclude.
   """
def listCFiles(dir = '', exclude = None, includeHeaders = False):
    global excludeFromCompilation
    global _includeHeaders
    _includeHeaders = includeHeaders
    if (dir == ''): dir = './'

    excludeFromCompilation = exclude
    result = []

    for dirpath, subdirectories, filenames in os.walk(dir):
          _listCFilesVisitor(result, dirpath, filenames, subdirectories)
    return result

####################################################################

""" Reads an entire text file from disk """
def readFile(filename):
    f = open(filename, 'rt')
    try:
      s = f.read()
    except UnicodeDecodeError:
      f.close()
      f = open(filename, 'rb')
      s = f.read().decode('iso8859-1')
    f.close()
    return s

####################################################################

""" Writes an entire text file to disk """
def writeFile(filename, contents):
    f = open(filename, 'wt')
    f.write(contents)
    f.close()
    
####################################################################

""" Appends new contents to existing file (creating it if it does not exist) """
def appendFile(filename, contents):
    f = open(filename, 'at')
    f.write(contents)
    f.close()

####################################################################

# Returns a tuple of the number of lines of non-doxygen comments and
# doxygen comments.
def countComments(str):
    
    # Count C++ comments
    comments = str.count('//')
    doxygen = str.count('///')
    
    # Count C-style comments
    start = str.find('/*')
    while start != -1:
        end   = str.find('*/', start + 1)

        if end != -1:
            numLines = str.count('\n', start, end) + 1
            if (str[start + 2] == '*'):
                # This is a doxygen comment
                doxygen += numLines
            else:
                # This is a regular C comment
                comments += numLines
            start = str.find('/*', end + 1)
        else:
            start = -1

    return (comments, doxygen)


###############################################################

""" Adds the Unix mode mod (e.g., S_IROTH) to the specified path """
def addFilePermission(path, mod):
    os.chmod(os.stat().ST_MOD | mod)

###############################################################

def findMatchingParen(str, start):
    count = 0
    i = start
    open = str[start]
    close = {'(':')', '[':']', '{':'}', '<':'>'}[open]
    while i < len(str):
      if str[i] == open:
         count += 1
      elif str[i] == close:
         count -= 1
         if (count == 0): return i
      i += 1
    return -1

###############################################################

""" Expands environment variables of the form $(var) and $(shell ...)
    in a string. Also allows arbitrary python expressions of
    the form $eval(...)
    Compare to os.path.expandvars, which only handles
    $var and therefore requires a separation character after
    variables."""
def expandvars(str, evalevt):
    while '$' in str:
        value = ''
        i = str.find('$')
        j = i

        if str[i: i + 2] == '$(':
          # Variable processing
          j = str.find(')', i)
          if (i > len(str) - 3) or (str[i + 1] != '(') or (j < 0):
            raise Exception('Environment variables must have the form $(var), $shell(cmds), or $eval(expr)')

          varexpr = str[i:(j + 1)]
          varname = str[(i + 2):j]
          value = os.getenv(varname)

        elif str[i : i+7] == '$shell(':
          # Shell processing
          j = findMatchingParen(str, i + 1)
          cmd = str[(i + 7) : j]
          value = shell(cmd, verbosity)
        elif  str[i : i+6] == '$eval(':
          # Python expression processing
          j = findMatchingParen(str, i + 5)
          expr = str[(i + 6) : j]
          value = eval(expr, evalevt)
        elif str[i : i + 2] == '$$':
            value = '\0'
            j = i + 1
        else:
            raise Exception('Environment variables must have the form $(var), $shell(cmds), or $eval(expr). Encountered ' + str)
        if (value == None):
          value = ''

        if verbosity > NORMAL: print('Replaced ' + str[i : j + 1] + ' with ' + value)
        str = str[:i] + value + str[j + 1:]
    str = str.replace('\0', "$")
    return str
