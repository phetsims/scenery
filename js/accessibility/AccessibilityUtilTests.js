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

  // tests
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

  QUnit.test( 'getNextPreviousFocusable', function( assert ) {
    var parent = AccessibilityUtil.createElement( 'div', false );

    var button = AccessibilityUtil.createElement( 'button', true ); // focusable
    var div = AccessibilityUtil.createElement( 'div', true ); // focusable
    var p = AccessibilityUtil.createElement( 'p', false ); // not focusable

    // elements must be in DOM to be focusable
    document.body.appendChild( parent );
    parent.appendChild( button );
    parent.appendChild( div );
    parent.appendChild( p );

    var firstFocusable = AccessibilityUtil.getNextFocusable( parent );
    assert.ok( firstFocusable === button, 'first focusable found' );
    firstFocusable.focus();

    var secondFocusable = AccessibilityUtil.getNextFocusable( parent );
    assert.ok( secondFocusable === div, 'second focusable found' );
    secondFocusable.focus();

    // should still return the div because the p isn't focusable
    var thirdFocusable = AccessibilityUtil.getNextFocusable( parent );
    assert.ok( thirdFocusable === div, 'no more focusables after div' );

    // remove the DOM nodes so they don't clutter the tests
    document.body.removeChild( parent );
  } );

  QUnit.test( 'overrideFocusWithTabIndex', function ( assert ) {

    // tab index should only be set on elements where we are overriding what is being done natively in the browser
    var defaultButton = AccessibilityUtil.createElement( 'button', true ); // focusable
    var defaultParagraph = AccessibilityUtil.createElement( 'p', false ); // not focusable
    var defaultDiv = AccessibilityUtil.createElement( 'div', false ); // not focusable

    // use getAttribute because tabIndex DOM property is always provided by default
    assert.ok( defaultButton.getAttribute( 'tabindex' ) === null, 'default button has no tab index' );
    assert.ok( defaultParagraph.getAttribute( 'tabindex' ) === null, 'default paragraph has no tab index' );
    assert.ok( defaultDiv.getAttribute( 'tabindex' ) === null, 'default div has no tab index' );

    // custom focusability should all have tab indices, even those that are being removed from the document
    var customButton = AccessibilityUtil.createElement( 'button', false ); // not focusable
    var customParagraph = AccessibilityUtil.createElement( 'p', true ); // focusable
    var customDiv = AccessibilityUtil.createElement( 'div', true ); // focusable

    assert.ok( customButton.getAttribute( 'tabindex' ) === '-1', 'custom button removed from focus' );
    assert.ok( customParagraph.getAttribute( 'tabindex' ) === '0', 'custom paragraph added to focus' );
    assert.ok( customDiv.getAttribute( 'tabindex' ) === '0', 'custom button removed from focus' );
  } );
} );