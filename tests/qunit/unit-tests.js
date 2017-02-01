// Copyright 2002-2014, University of Colorado Boulder

// path to test base, without a slash, e.g. '../../../scenery/tests/qunit'
function runSceneryTests( pathToTestBase ) { // eslint-disable-line no-unused-vars
  'use strict';

  function loadTestFile( src ) {
    var script = document.createElement( 'script' );
    script.type = 'text/javascript';
    script.async = false;

    // make sure things aren't cached, just in case
    script.src = pathToTestBase + '/' + src + '?random=' + Math.random().toFixed( 10 );

    document.getElementsByTagName( 'head' )[ 0 ].appendChild( script );
  }

  loadTestFile( 'js/test-utils.js' );
  loadTestFile( 'js/scene.js' );
  loadTestFile( 'js/shapes.js' );
  loadTestFile( 'js/alignment.js' );
  loadTestFile( 'js/color.js' );
  loadTestFile( 'js/input.js' );
  loadTestFile( 'js/transforms.js' );
  loadTestFile( 'js/miscellaneous.js' );
  loadTestFile( 'js/display.js' );
  loadTestFile( 'js/focus.js' );
  loadTestFile( 'js/from-fuzzer.js' );
  loadTestFile( 'js/pixel-comparison.js' );
  loadTestFile( 'js/accessibility.js' );
}
