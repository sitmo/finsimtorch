@REM Minimal makefile for Sphinx documentation
@REM

@REM You can set these variables from the command line.
SET SPHINXOPTS=-W
SET SPHINXBUILD=sphinx-build
SET SOURCEDIR=.
SET BUILDDIR=_build

@REM Put it first so that "make" without argument is like "make help".
help:
	%SPHINXBUILD% -M help "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%

.PHONY: help Makefile

@REM Catch-all target: route all unknown targets to Sphinx using the "new" style
@REM where the target is the name of the output file.
%: Makefile
	%SPHINXBUILD% -M %* "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
