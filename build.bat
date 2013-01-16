
:: don't print every command
@ECHO OFF

setlocal enableextensions enabledelayedexpansion

del /q concatenated.js 2> NUL
del /q phet-scene.js 2> NUL
del /q phet-scene-min.js 2> NUL
del /q phet-scene-min.js.map 2> NUL

:: rebuild concatenated.js from the file list by appending
echo Building concatenated.js
for /f %%k in (build/file-list.txt) DO (
	:: call out to the inside of our loop. it's a separate file because of how the for loop is expanded. see http://www.robvanderwoude.com/for.php (section 7)
	call build\sub-build.bat %%k
)

:: build phet-scene.js
:: echo Building phet-scene.js
:: java -jar bin/closure-compiler.jar --compilation_level WHITESPACE_ONLY --formatting PRETTY_PRINT --js concatenated.js --js_output_file phet-scene.js

:: build phet-scene-min.js
echo Building phet-scene-min.js
java -jar bin/closure-compiler.jar --compilation_level SIMPLE_OPTIMIZATIONS --js concatenated.js --js_output_file phet-scene-min.js --create_source_map ./phet-scene-min.js.map --source_map_format=V3 --define=phetDebug=false
type build\source-map-appendix.js >> phet-scene-min.js
