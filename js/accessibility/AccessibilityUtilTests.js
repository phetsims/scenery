// Copyright 2017, University of Colorado Boulder

/**
 * Accessibility tests
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var AccessibilityUtil = require( 'SCENERY/accessibility/AccessibilityUtil' );

  QUnit.module( 'AccessibilityUtilTests' );


  QUnit.test( 'insertElements', function( assert ) {

    var div1 = document.createElement( 'div1' );
    var div2 = document.createElement( 'div2' );
    var div3 = document.createElement( 'div3' );
    var div4 = document.createElement( 'div4' );

    AccessibilityUtil.insertElements( div1, [ div2, div3, div4 ] );

    assert.ok( div1.childNodes.length === 3, 'inserted number of elements');
    assert.ok( div1.childNodes[0] === div2, 'inserted div2 order of elements');
    assert.ok( div1.childNodes[1] === div3, 'inserted div3 order of elements');
    assert.ok( div1.childNodes[2] === div4, 'inserted div4 order of elements');


    var div5 = document.createElement( 'div5' );
    var div6 = document.createElement( 'div6' );
    var div7 = document.createElement( 'div7' );

    AccessibilityUtil.insertElements( div1, [div5,div6,div7], div3);
    assert.ok( div1.childNodes[0] === div2, 'inserted div2 order of elements');
    assert.ok( div1.childNodes[1] === div5, 'inserted div5 order of elements');
    assert.ok( div1.childNodes[2] === div6, 'inserted div6 order of elements');
    assert.ok( div1.childNodes[3] === div7, 'inserted div7 order of elements');
    assert.ok( div1.childNodes[4] === div3, 'inserted div3 order of elements');
    assert.ok( div1.childNodes[5] === div4, 'inserted div4 order of elements');
  } );

} );