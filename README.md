
Scenery
=======

Prototype HTML5 Scene Graph
---------------------------

This is under active prototyping, so please expect any sort of API to change. Comments at this stage are very welcome.

Since the concatenated / minified JS is under .gitignore, please either use the
associated Makefile (with 'make') or the Windows batch script 'build.bat'.

You can also grab the development version [scenery.js](http://phet.colorado.edu/files/scenery/scenery.js) or production version
[scenery-min.js](http://phet.colorado.edu/files/scenery/scenery-min.js). They are currently not versioned due to the accelerated
development speed, but will be more stable soon.

Documentation of Scenery is available at:
* [Main Documentation](http://phet.colorado.edu/files/scenery/doc/) (up-to-date version is checked in at scenery/doc/index.html)

Examples:
* [Hello world](http://phet.colorado.edu/files/scenery/examples/hello-world.html)
* [Node types](http://phet.colorado.edu/files/scenery/examples/nodes.html)
* [Multi-touch and Drag-by-touchover](http://phet.colorado.edu/files/scenery/examples/multi-touch.html)
* [Cursors](http://phet.colorado.edu/files/scenery/examples/cursors.html)
* [Devious Dragging (handling of corner cases)](http://phet.colorado.edu/files/scenery/examples/devious-drag.html)

For testing purposes, the following are currently being worked on:
* [Scene Graph Comparisons and Experiments](http://phet.colorado.edu/files/scenery/tests/easel-performance/easel-tests.html)
* [Unit Tests / Linter](http://phet.colorado.edu/files/scenery/tests/unit-tests/unit-tests.html)
* [Performance Improvement/Regression](http://phet.colorado.edu/files/scenery/tests/benchmarks/performance-tests.html)
* [Benchmarks](http://phet.colorado.edu/files/scenery/tests/benchmarks/benchmarks.html)
* [WebGL Prototype](http://phet.colorado.edu/files/scenery/tests/webgl-test/webgl-test.html)
* [Canvas Browser Test Suite](http://phet.colorado.edu/files/scenery/tests/browsers/canvas-test-suite.html)

jsPerf tests for various purposes:
* [http://jsperf.com/overloading-options](http://jsperf.com/overloading-options)
* [http://jsperf.com/small-operation-testing](http://jsperf.com/small-operation-testing)
* [http://jsperf.com/small-operation-testing-2](http://jsperf.com/small-operation-testing-2)
