
Scenery
=======

Prototype HTML5 Scene Graph
---------------------------

This is under active prototyping, so please expect any sort of API to change. Comments at this stage are very welcome.
A [tentative website](http://phetsims.github.com/scenery/) is up for browsing documentation and examples in a more user-friendly way.

[Grunt](http://gruntjs.com/) is used to build the source ("npm install -g grunt-cli", "npm install" and "grunt" at the top level
should build into dist/). [Node.js](http://nodejs.org/) is required for this process.

Currently, you can also grab the unminified [scenery.js](http://phetsims.github.com/scenery/dist/development/scenery.js) or
minified version [scenery-min.js](http://phetsims.github.com/scenery/dist/standalone/scenery.min.js).
They are currently not versioned due to the accelerated development speed, but will be more stable soon. A development version
will be available soon that has assertions enabled.

Documentation of Scenery is available at:
* [Main Documentation](http://phetsims.github.com/scenery/doc/) (up-to-date version is checked in at scenery/doc/index.html)
* [A Tour of Scenery](http://phetsims.github.com/scenery/doc/a-tour-of-scenery.html) (up-to-date version is checked in at scenery/doc/index.html)

Examples:
* [Hello world](http://phetsims.github.com/scenery/examples/hello-world.html)
* [Node types](http://phetsims.github.com/scenery/examples/nodes.html)
* [Multi-touch and Drag-by-touchover](http://phetsims.github.com/scenery/examples/multi-touch.html)
* [Cursors](http://phetsims.github.com/scenery/examples/cursors.html)
* [Devious Dragging (handling of corner cases)](http://phetsims.github.com/scenery/examples/devious-drag.html)

For testing purposes, the following are currently being worked on:
* [Unit Tests / Linter](http://phetsims.github.com/scenery/tests/qunit/compiled-unit-tests.html)
* [Renderer Comparison](http://phetsims.github.com/scenery/tests/renderer-comparison.html)
* [Scene Graph Comparisons and Experiments](http://phetsims.github.com/scenery/tests/easel-performance/easel-tests.html)
* [Performance Improvement/Regression](http://phetsims.github.com/scenery/tests/benchmarks/performance-tests.html)
* [Canvas Browser Differences](http://jonathan-olson.com/canvas-diff/canvas-diff.html)
