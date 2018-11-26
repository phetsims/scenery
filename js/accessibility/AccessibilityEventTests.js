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

    var aGotFocus = false;
    var aLostFocus = false;
    var bGotFocus = false;

    rootNode.addChild( a );

    a.addInputListener( {
      focus() {
        console.log( 'afocus' );

        aGotFocus = true;
      },
      blur() {
        console.log( 'ablur' );

        aLostFocus = true;
      }
    } );

    a.focus();
    assert.ok( aGotFocus, 'a should have been focused' );
    assert.ok( !aLostFocus, 'a should not blur' );

    var b = new Rectangle( 0, 0, 20, 20, { tagName: 'button' } );

    // TODO: what if b was child of a, make sure these events don't bubble!
    rootNode.addChild( b );

    b.addInputListener( {
      focus() {
        console.log( 'bfocus' );

        bGotFocus = true;
      }
    } );

    b.focus();

    assert.ok( bGotFocus, 'b should have been focused' );
    assert.ok( aLostFocus, 'a should have lost focused' );
  } );

} );