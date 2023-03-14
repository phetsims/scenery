scenery
=======

Scenery is a library for building interactive visual experiences in HTML5, which can be displayed in a
combination of ways (WebGL, SVG, Canvas, etc.)

By PhET Interactive Simulations
https://phet.colorado.edu/

### Documentation

The majority of the documentation exists within the code itself, but our
[main documentation](http://phetsims.github.io/scenery/doc/)
and [a tour of features](http://phetsims.github.io/scenery/doc/a-tour-of-scenery.html)
is available online, along with other resources under [the dedicated website](http://phetsims.github.io/scenery/)

Currently, you can grab the unminified [scenery.debug.js](http://phetsims.github.io/scenery/dist/scenery.debug.js)
or
minified version [scenery.min.js](http://phetsims.github.io/scenery/dist/scenery.min.js).

We have a [list of examples](https://phetsims.github.io/scenery/examples/) to get started from

The [PhET Development Overview](https://github.com/phetsims/phet-info/blob/master/doc/phet-development-overview.md) is
the most complete guide to PhET Simulation Development. This guide includes how
to obtain simulation code and its dependencies, notes about architecture & design, how to test and build the sims, as
well as other important information.

### To check out and build the code

Our processes depend on [Node.js](http://nodejs.org/) and [Grunt](http://gruntjs.com/). It's highly recommended to install
Node.js and then grunt with `npm install -g grunt-cli`.

(1) Clone the simulation and its dependencies:
```
git clone https://github.com/phetsims/assert.git
git clone https://github.com/phetsims/axon.git
git clone https://github.com/phetsims/chipper.git
git clone https://github.com/phetsims/dot.git
git clone https://github.com/phetsims/kite.git
git clone https://github.com/phetsims/perennial.git perennial-alias
git clone https://github.com/phetsims/phet-core.git
git clone https://github.com/phetsims/phetcommon.git
git clone https://github.com/phetsims/scenery.git
git clone https://github.com/phetsims/sherpa.git
git clone https://github.com/phetsims/tandem.git
git clone https://github.com/phetsims/utterance-queue.git
```

(2) Install dev dependencies:
```
cd chipper
npm install
cd ../perennial-alias
npm install
cd ../scenery
npm install
```

(3) Build scenery

Ensure you're in the kite directory and run `grunt --lint=false --report-media=false`. This will output files under the `build/` directory

### License

MIT license, see [LICENSE](LICENSE)

### Contributing
If you would like to contribute to this repo, please read our [contributing guidelines](https://github.com/phetsims/community/blob/master/CONTRIBUTING.md).
