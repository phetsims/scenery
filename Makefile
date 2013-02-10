# requires GNU Make. for Windows, use build.bat instead

all: scenery-min.js

# depends on having GNU Make, according to http://stackoverflow.com/questions/6767413/create-a-variable-in-a-makefile-by-reading-contents-of-another-file
JS_FILES := $(shell cat build/file-list.txt | xargs)

scenery-min.js: scenery.js
	java -jar bin/closure-compiler.jar --compilation_level SIMPLE_OPTIMIZATIONS --js scenery.js --js_output_file scenery-min.js \
		--create_source_map ./scenery-min.js.map --source_map_format=V3 --define=phetDebug=false --language_in ECMASCRIPT5_STRICT
	cat build/source-map-appendix.js >> scenery-min.js

scenery.js: $(JS_FILES)
	cat $(JS_FILES) > scenery.js

clean:
	rm -f concatenated.js phet-scene-min.js phet-scene-min.js.map scenery-min.js scenery-min.js.map scenery.js

