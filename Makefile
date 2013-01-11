# requires GNU Make

all: phet-scene.js phet-scene-min.js

# depends on having GNU Make, according to http://stackoverflow.com/questions/6767413/create-a-variable-in-a-makefile-by-reading-contents-of-another-file
JS_FILES := $(shell cat build/file-list.txt | xargs)

phet-scene.js: concatenated.js
	java -jar bin/closure-compiler.jar --compilation_level WHITESPACE_ONLY --formatting PRETTY_PRINT --js concatenated.js --js_output_file phet-scene.js

phet-scene-min.js: concatenated.js
	java -jar bin/closure-compiler.jar --compilation_level SIMPLE_OPTIMIZATIONS --js concatenated.js --js_output_file phet-scene-min.js

concatenated.js: $(JS_FILES)
	cat $(JS_FILES) > concatenated.js

clean:
	rm -f phet-scene.js phet-scene-min.js concatenated.js

