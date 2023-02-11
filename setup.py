from distutils.core import setup, Extension
import sysconfig

def main():
	CFLAGS = ['-g', '-Wall', '-std=c99', '-fopenmp', '-mavx', '-mfma', '-pthread', '-O3']
	LDFLAGS = ['-fopenmp']
	#numc
	faster = Extension('numc',sources = ['numc.c', 'matrix.c'], extra_compile_args = CFLAGS, extra_link_args= LDFLAGS)

	setup (name = 'numc',
	version = '1.0',
	description = 'This is the faster package',
	ext_modules = [faster])

if __name__ == "__main__":
    main()
