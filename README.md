
Scenery
=======

Prototype HTML5 Scene Graph
---------------------------

This is under active prototyping, so please expect any sort of API to change. Comments at this stage are very welcome.

[Grunt](http://gruntjs.com/) is used to build the source ("npm install -g grunt-cli", "npm install" and "grunt" at the top level
should build into dist/). [Node.js](http://nodejs.org/) is required for this process.

Currently, you can also grab the unminified [scenery.js](http://phet.colorado.edu/files/scenery/dist/standalone/scenery.js) or
minified version [scenery-min.js](http://phet.colorado.edu/files/scenery/dist/standalone/scenery.min.js).
They are currently not versioned due to the accelerated development speed, but will be more stable soon. A development version
will be available soon that has assertions enabled.

Documentation of Scenery is available at:
* [Main Documentation](http://phet.colorado.edu/files/scenery/doc/) (up-to-date version is checked in at scenery/doc/index.html)

Examples:
* [Hello world](http://phet.colorado.edu/files/scenery/examples/hello-world.html)
* [Node types](http://phet.colorado.edu/files/scenery/examples/nodes.html)
* [Multi-touch and Drag-by-touchover](http://phet.colorado.edu/files/scenery/examples/multi-touch.html)
* [Cursors](http://phet.colorado.edu/files/scenery/examples/cursors.html)
* [Devious Dragging (handling of corner cases)](http://phet.colorado.edu/files/scenery/examples/devious-drag.html)

For testing purposes, the following are currently being worked on:
* [Unit Tests / Linter](http://phet.colorado.edu/files/scenery/tests/qunit/compiled-unit-tests.html)
* [Renderer Comparison](http://phet.colorado.edu/files/scenery/tests/renderer-comparison.html)
* [Scene Graph Comparisons and Experiments](http://phet.colorado.edu/files/scenery/tests/easel-performance/easel-tests.html)
* [Performance Improvement/Regression](http://phet.colorado.edu/files/scenery/tests/benchmarks/performance-tests.html)
* [Canvas Browser Differences](http://jonathan-olson.com/canvas-diff/canvas-diff.html)
