// Copyright 2019-2022, University of Colorado Boulder

/**
 * Tests for styling accessibility siblings. Siblings should be positioned on top of other elements in the PDOM.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import Bounds2 from '../../../../dot/js/Bounds2.js';
import Vector2 from '../../../../dot/js/Vector2.js';
import Display from '../../display/Display.js';
import Node from '../../nodes/Node.js';
import Rectangle from '../../nodes/Rectangle.js';
import PDOMPeer from './PDOMPeer.js';

// constants
const PIXEL_PADDING = 3;

QUnit.module( 'PDOMSiblingTests' );

/**
 * Gets the bounds of the DOMElement in the viewport in global coordinates.
 * @param {Node} node
 * @returns {Bounds2}
 */
const getSiblingBounds = node => {
  const siblingRect = node.pdomInstances[ 0 ].peer.primarySibling.getBoundingClientRect();
  return new Bounds2( siblingRect.x, siblingRect.y, siblingRect.x + siblingRect.width, siblingRect.y + siblingRect.height );
};

/**
 * Returns true if the node's first accessible peer has a primary sibling with bounds thar are correctly positioned
 * in the viewport. Some padding is required for the HTML elements to have defined bounds, so we allow a few pixels
 * of error.
 * @param  {Node} node
 * @returns {boolean}
 */
const siblingBoundsCorrect = node => {

  // if a pdomTransformSourceNode is specified, the sibling should overlap this node's bounds instead of its own
  // bounds
  const transformSourceNode = node.pdomTransformSourceNode;
  const sourceNodeBounds = transformSourceNode ? transformSourceNode.globalBounds : node.globalBounds;

  const siblingBounds = getSiblingBounds( node );
  const comparedBounds = node.positionInPDOM ? sourceNodeBounds : PDOMPeer.OFFSCREEN_SIBLING_BOUNDS;
  return siblingBounds.equalsEpsilon( comparedBounds, PIXEL_PADDING );
};

// tests
QUnit.test( 'sibling positioning', assert => {

  const rootNode = new Node( { tagName: 'div' } );
  const display = new Display( rootNode );
  document.body.appendChild( display.domElement );

  // test bounds are set for basic input elements
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

  // test that bounds are set correctly once we have a hierarchy and add transformations
  rootNode.removeChild( buttonElement );
  rootNode.removeChild( divElement );
  rootNode.removeChild( inputElement );

  rootNode.addChild( divElement );
  divElement.addChild( buttonElement );
  buttonElement.addChild( inputElement );

  // arbitrary transformations down the tree (should be propagated to input element)
  divElement.setCenter( new Vector2( 50, 50 ) );
  buttonElement.setScaleMagnitude( 0.89 );
  inputElement.setRotation( Math.PI / 4 );

  // udpdate so the display to position elements
  display.updateDisplay();
  assert.ok( siblingBoundsCorrect( buttonElement ), 'button element descendant correctly positioned' );
  assert.ok( siblingBoundsCorrect( inputElement ), 'input element descendant correctly positioned' );

  // when inner content of an element changes, its client bounds change - make sure that the element still matches
  // the Node
  buttonElement.innerHTML = 'Some Test';
  display.updateDisplay();
  assert.ok( siblingBoundsCorrect( buttonElement ), 'button element descendant correclty positioned after inner content changed' );

  // remove the display element so it doesn't interfere with qunit API
  document.body.removeChild( display.domElement );
  display.dispose();

} );

QUnit.test( 'PDOM transform source Node', assert => {
  const rootNode = new Node( { tagName: 'div' } );
  const display = new Display( rootNode );
  document.body.appendChild( display.domElement );

  const buttonNode = new Rectangle( 5, 5, 5, 5, { tagName: 'button', positionInPDOM: true } );
  const transformSourceNode = new Rectangle( 0, 0, 25, 25 );

  rootNode.addChild( buttonNode );
  rootNode.addChild( transformSourceNode );

  // update the display to position elements
  display.updateDisplay();
  assert.ok( siblingBoundsCorrect( buttonNode ), 'button element child of root correctly positioned' );

  const siblingBoundsBefore = getSiblingBounds( buttonNode );

  // test setting transform source Node
  buttonNode.pdomTransformSourceNode = transformSourceNode;

  // update the display to position elements
  display.updateDisplay();

  assert.ok( siblingBoundsCorrect( buttonNode ), 'button element transformed with transform source Node' );
  assert.ok( !siblingBoundsBefore.equals( getSiblingBounds( buttonNode ) ), 'sibling bounds should have changed after setting transform source' );

  // reposition the buttonNode - pdom sibling should NOT reposition
  const siblingBoundsBeforeNodeReposition = getSiblingBounds( buttonNode );
  buttonNode.setX( 100 );
  buttonNode.setY( 100 );
  display.updateDisplay();
  assert.ok( siblingBoundsCorrect( buttonNode ), 'sibling bounds correct after node repositioned' );
  assert.ok( siblingBoundsBeforeNodeReposition.equals( getSiblingBounds( buttonNode ) ), 'transform source didnt change, primary sibling should not reposition' );

  // reposition the transform source - pdom sibling SHOULD reposition
  const siblingBoundsBeforeSourceReposition = getSiblingBounds( buttonNode );
  transformSourceNode.setX( 50 );
  transformSourceNode.setX( 50 );
  display.updateDisplay();
  assert.ok( siblingBoundsCorrect( buttonNode ), 'sibling bounds correct after source node repositioned' );
  assert.ok( !siblingBoundsBeforeSourceReposition.equals( getSiblingBounds( buttonNode ) ), 'transform source didnt change, primary sibling should not reposition' );

  // remove the display element so it doesn't interfere with qunit API
  document.body.removeChild( display.domElement );
  display.dispose();

} );

QUnit.test( 'setPositionElements test', assert => {
  const rootNode = new Node( { tagName: 'div' } );
  const display = new Display( rootNode );
  document.body.appendChild( display.domElement );

  const buttonNode = new Rectangle( 5, 5, 5, 5, { tagName: 'button', positionInPDOM: true } );
  rootNode.addChild( buttonNode );

  display.updateDisplay();
  assert.ok( siblingBoundsCorrect( buttonNode ), 'sibling bounds initially correct' );

  buttonNode.setPositionInPDOM( false );
  display.updateDisplay();
  assert.ok( siblingBoundsCorrect( buttonNode ), 'sibling bounds correct after positionInPDOM false' );

  buttonNode.setPositionInPDOM( true );
  display.updateDisplay();
  assert.ok( siblingBoundsCorrect( buttonNode ), 'sibling bounds repositioned after positionInPDOM true' );

  // remove the display element so it doesn't interfere with qunit API
  document.body.removeChild( display.domElement );
  display.dispose();
} );