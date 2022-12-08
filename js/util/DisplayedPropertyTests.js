// Copyright 2021-2022, University of Colorado Boulder

/**
 * DisplayedProperty Tests
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import Display from '../display/Display.js';
import Node from '../nodes/Node.js';
import DisplayedProperty from './DisplayedProperty.js';

QUnit.module( 'DisplayedProperty' );

QUnit.test( 'basics', assert => {

  const rootNode = new Node();
  var display = new Display( rootNode ); // eslint-disable-line no-var
  display.initializeEvents();
  document.body.appendChild( display.domElement );

  const aNode = new Node();
  const aDisplayedProperty = new DisplayedProperty( aNode );
  assert.ok( aDisplayedProperty.value === false, 'not connected, not displayed' );

  rootNode.addChild( aNode );
  display.updateDisplay();

  assert.ok( aDisplayedProperty.value === true, 'connected, displayed' );

  rootNode.visible = false;
  display.updateDisplay();

  assert.ok( aDisplayedProperty.value === false, 'connected, parent not visible, not displayed' );

  aNode.visible = false;
  rootNode.visible = true;
  display.updateDisplay();

  assert.ok( aDisplayedProperty.value === false, 'connected, not visible, not displayed' );

  aNode.visible = true;
  display.updateDisplay();

  assert.ok( aDisplayedProperty.value === true, 'back to visible, displayed' );

  display.dispose();
  display.domElement.parentElement.removeChild( display.domElement );
} );


QUnit.test( 'pdom visibility', assert => {

  const rootNode = new Node();
  var display = new Display( rootNode ); // eslint-disable-line no-var
  display.initializeEvents();
  document.body.appendChild( display.domElement );

  const aNode = new Node( { tagName: 'p' } );
  const aDisplayedProperty = new DisplayedProperty( aNode );

  const aParent = new Node( { children: [ aNode ] } );
  rootNode.addChild( aParent );

  const pdomParentForA = new Node( { tagName: 'div' } );
  rootNode.addChild( pdomParentForA );

  pdomParentForA.pdomOrder = [ aNode ];
  display.updateDisplay();

  assert.ok( aDisplayedProperty.value === true, 'visible even with pdomOrder, displayed' );

  aParent.visible = false;
  display.updateDisplay();


  // This test fails and @zepumph doesn't think it should! // TODO support pdom visibility, https://github.com/phetsims/scenery/issues/1167
  // assert.ok( aDisplayedProperty.value === true, 'pdomOrder makes it visible, displayed' );


  // Some more tests to run:

  // toggle instance visibility in the PDOM trail

  // swap pdom order and make sure DisplayedProperty updates

  display.dispose();
  display.domElement.parentElement.removeChild( display.domElement );
} );