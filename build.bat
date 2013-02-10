
:: don't print every command
@ECHO OFF

setlocal enableextensions enabledelayedexpansion

:: remove old filenames
del /q concatenated.js 2> NUL
del /q phet-scene-min.js 2> NUL
del /q phet-scene-min.js.map 2> NUL

:: clean new filenames
del /q scenery.js 2> NUL
del /q scenery-min.js 2> NUL
del /q scenery-min.js.map 2> NUL

:: rebuild scenery.js from the file list by appending
echo Building scenery.js
for /f %%k in (build/file-list.txt) DO (
	:: call out to the inside of our loop. it's a separate file because of how the for loop is expanded. see http://www.robvanderwoude.com/for.php (section 7)
	call build\sub-build.bat %%k
)

:: build scenery-min.js
echo Building scenery-min.js
java -jar bin/closure-compiler.jar --compilation_level SIMPLE_OPTIMIZATIONS --js scenery.js --js_output_file scenery-min.js --create_source_map ./scenery-min.js.map --source_map_format=V3 --define=phetDebug=false --language_in ECMASCRIPT5_STRICT
type build\source-map-appendix.js >> scenery-min.js
