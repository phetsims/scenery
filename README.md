scenery
=======

Scenery is an HTML5 scene graph.

By PhET Interactive Simulations
http://phet.colorado.edu/

### Documentation

This is under active prototyping, so please expect any sort of API to change. Comments at this stage are very welcome.
A [tentative website](http://phetsims.github.io/scenery/) is up for browsing documentation and examples in a more user-friendly way.

[Grunt](http://gruntjs.com/) is used to build the source ("npm update -g grunt-cli", "npm update" and "grunt" at the top level
should build into build/). [Node.js](http://nodejs.org/) is required for this process.

Currently, you can also grab the unminified [scenery.js](http://phetsims.github.io/scenery/build/scenery.min.js) or
minified version [scenery-min.js](http://phetsims.github.io/scenery/build/standalone/scenery.min.js).
They are currently not versioned due to the accelerated development speed, but will be more stable soon. A development version
will be available soon that has assertions enabled.

Documentation of Scenery is available at:
* [Main Documentation](http://phetsims.github.io/scenery/doc/) (up-to-date version is checked in at scenery/doc/index.html)
* [A Tour of Scenery](http://phetsims.github.io/scenery/doc/a-tour-of-scenery.html) (up-to-date version is checked in at scenery/doc/index.html)

Examples:
* [Hello world](http://phetsims.github.io/scenery/examples/hello-world.html)
* [Node types](http://phetsims.github.io/scenery/examples/nodes.html)
* [Multi-touch and Drag-by-touchover](http://phetsims.github.io/scenery/examples/multi-touch.html)
* [Cursors](http://phetsims.github.io/scenery/examples/cursors.html)
* [Devious Dragging (handling of corner cases)](http://phetsims.github.io/scenery/examples/devious-drag.html)

For testing purposes, the following are currently being worked on:
* [Unit Tests / Linter](http://phetsims.github.io/scenery/scenery-tests.html)
* [Renderer Comparison](http://phetsims.github.io/scenery/tests/renderer-comparison.html)
* [Canvas Browser Differences](http://jonathan-olson.com/canvas-diff/canvas-diff.html)

The [PhET Development Overview](http://bit.ly/phet-html5-development-overview) is the most complete guide to PhET Simulation Development. This guide includes how
to obtain simulation code and its dependencies, notes about architecture & design, how to test and build the sims, as well as other important information.

### License
See the [license](LICENSE)