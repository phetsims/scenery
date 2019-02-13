// Copyright 2017, University of Colorado Boulder

/**
 * Tests for styling accessibility siblings. Siblings should be positioned on top of other elements in the PDOM.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  // const AccessibilityUtil = require( 'SCENERY/accessibility/AccessibilityUtil' );
  const Bounds2 = require( 'DOT/Bounds2' );
  const Display = require( 'SCENERY/display/Display' );
  const Node = require( 'SCENERY/nodes/Node' );
  const Rectangle = require( 'SCENERY/nodes/Rectangle' );

  // constants
  const PIXEL_PADDING = 3;

  QUnit.module( 'AccessibleSiblingTests' );

  /**
   * Returns true if the node's first accessible peer has a primary sibling with bounds thar are correctly positioned
   * in the viewport. Some padding is required for the HTML elements to have defined bounds, so we allow a few pixels
   * of error.
   * @param  {Node} node
   * @returns {boolean}
   */
  const siblingBoundsCorrect = node => {
    const nodeBounds = node.globalBounds;

    const siblingRect = node.accessibleInstances[ 0 ].peer.primarySibling.getBoundingClientRect();
    const siblingBounds = new Bounds2( siblingRect.x, siblingRect.y, siblingRect.x + siblingRect.width, siblingRect.y + siblingRect.height );

    return nodeBounds.equalsEpsilon( siblingBounds, PIXEL_PADDING );
  };

  // tests
  QUnit.test( 'sibling positioning', function( assert ) {

    const rootNode = new Node( { tagName: 'div' } );
    const display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    const buttonElement = new Rectangle( 5, 5, 5, 5, { tagName: 'button' } );
    const divElement = new Rectangle( 0, 0, 20, 20, { tagName: 'div', focusable: true } );
    const inputElement = new Rectangle( 10, 3, 25, 5, { tagName: 'input', inputType: 'range' } );

    rootNode.addChild( buttonElement );
    rootNode.addChild( divElement );
    rootNode.addChild( inputElement );

    // udpdate so the display to position elements
    display.updateDisplay();

    assert.ok( siblingBoundsCorrect( buttonElement ), 'button element child of root correctly positioned' );
    assert.ok( siblingBoundsCorrect( divElement ), 'div element child of root correctly positioned' );
    assert.ok( siblingBoundsCorrect( inputElement ), 'input element child of root correctly positioned' );
  } );
} );