// Copyright 2017, University of Colorado Boulder

/**
 * Accessibility tests
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */
define( require => {
  'use strict';

  // modules
  var Display = require( 'SCENERY/display/Display' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Rectangle = require( 'SCENERY/nodes/Rectangle' );

  QUnit.module( 'AccessibilityEvents' );

  QUnit.test( 'focusin/focusout', assert => {


    var rootNode = new Node( { tagName: 'div' } );
    var display = new Display( rootNode ); // eslint-disable-line
    display.initializeEvents();
    document.body.appendChild( display.domElement );

    var a = new Rectangle( 0, 0, 20, 20, { tagName: 'button' } );

    var gotFocus = false;

    rootNode.addChild( a );

    a.addInputListener( {
      focus() {
        gotFocus = true;
      }
    } );

    a.focus();
    assert.ok( gotFocus, 'a should have been focused' );
  } );

} );