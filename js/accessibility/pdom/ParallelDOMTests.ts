// Copyright 2017-2025, University of Colorado Boulder

/**
 * ParallelDOM tests
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import StringProperty from '../../../../axon/js/StringProperty.js';
import Display from '../../display/Display.js';
import Circle from '../../nodes/Circle.js';
import Node from '../../nodes/Node.js';
import Rectangle from '../../nodes/Rectangle.js';
import { AccessibleNameBehaviorFunction, ParallelDOMOptions } from './ParallelDOM.js';
import PDOMFuzzer from './PDOMFuzzer.js';
import PDOMPeer from './PDOMPeer.js';
import PDOMUtils from './PDOMUtils.js';

// constants
const TEST_INNER_CONTENT = 'Test Inner Content Here please^&*. Thanks you so very mucho.';
const TEST_LABEL = 'Test label';
const TEST_LABEL_2 = 'Test label 2';
const TEST_DESCRIPTION = 'Test description';
const TEST_LABEL_HTML = '<strong>I ROCK as a LABEL</strong>';
const TEST_LABEL_HTML_2 = '<strong>I ROCK as a LABEL 2</strong>';
const TEST_DESCRIPTION_HTML = '<strong>I ROCK as a DESCRIPTION</strong>';
const TEST_DESCRIPTION_HTML_2 = '<strong>I ROCK as a DESCRIPTION 2</strong>';
const TEST_CLASS_ONE = 'test-class-one';
const TEST_CLASS_TWO = 'test-class-two';

// These should manually match the defaults in the ParallelDOM.js trait
const DEFAULT_LABEL_TAG_NAME = PDOMUtils.DEFAULT_LABEL_TAG_NAME;
const DEFAULT_DESCRIPTION_TAG_NAME = PDOMUtils.DEFAULT_DESCRIPTION_TAG_NAME;

// given the parent container element for a node, this value is the index of the label sibling in the
// parent's array of children HTMLElements.
const DEFAULT_LABEL_SIBLING_INDEX = 0;
const DEFAULT_DESCRIPTION_SIBLING_INDEX = 1;
const APPENDED_DESCRIPTION_SIBLING_INDEX = 2;

// a focus highlight for testing, since dummy nodes tend to have no bounds
const TEST_HIGHLIGHT = new Circle( 5 );

// a custom focus highlight (since dummy node's have no bounds)
const focusHighlight = new Rectangle( 0, 0, 10, 10 );

let canRunTests = true;

QUnit.module( 'ParallelDOM', {
  beforeEach: () => {

    // A test can only be run when the document has focus because tests require focus/blur events. Browsers
    // do not emit these events when the window is not active (especially true for pupetteer
    canRunTests = document.hasFocus();

    if ( !canRunTests ) {
      console.warn( 'Unable to run focus tests because the document does not have focus' );
    }
  }
} );

/**
 * Get a unique PDOMPeer from a node with accessible content. Will error if the node has multiple instances
 * or if the node hasn't been attached to a display (and therefore has no accessible content).
 */
function getPDOMPeerByNode( node: Node ): PDOMPeer {
  if ( node.pdomInstances.length === 0 ) {
    throw new Error( 'No pdomInstances. Was your node added to the scene graph?' );
  }

  else if ( node.pdomInstances.length > 1 ) {
    throw new Error( 'There should one and only one accessible instance for the node' );
  }
  else if ( !node.pdomInstances[ 0 ].peer ) {
    throw new Error( 'pdomInstance\'s peer should exist.' );
  }

  return node.pdomInstances[ 0 ].peer;
}

/**
 * Get an HTML Element for a Node from a particular PDOMPeer element name.
 * Assertions are thrown if none is found.
 */
function getDOMElementByName( node: Node, name: string ): HTMLElement {
  const peer = getPDOMPeerByNode( node );
  return peer.getElementByName( name );
}

/**
 * Get the id of a dom element representing a node in the DOM.  The accessible content must exist and be unique,
 * there should only be one accessible instance and one dom element for the node.
 *
 * NOTE: Be careful about getting references to dom Elements, the reference will be stale each time
 * the view (PDOMPeer) is redrawn, which is quite often when setting options.
 */
function getPrimarySiblingElementByNode( node: Node ): HTMLElement {
  const uniquePeer = getPDOMPeerByNode( node );
  return document.getElementById( uniquePeer.primarySibling!.id )!;
}

/**
 * Audit the root node for accessible content within a test, to make sure that content is accessible as we expect,
 * and so that our pdomAudit function may catch things that have gone wrong.
 * @param rootNode - the root Node attached to the Display being tested
 */
function pdomAuditRootNode( rootNode: Node ): void {
  rootNode.pdomAudit();
}

QUnit.test( 'tagName/innerContent options', assert => {

  const rootNode = new Node( { tagName: 'div' } );
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  // create some nodes for testing
  const a = new Node( { tagName: 'button', innerContent: TEST_LABEL } );

  rootNode.addChild( a );

  const aElement = getPrimarySiblingElementByNode( a );
  assert.ok( a.pdomInstances.length === 1, 'only 1 instance' );
  assert.ok( aElement.parentElement!.childNodes.length === 1, 'parent contains one primary siblings' );
  assert.ok( aElement.tagName === 'BUTTON', 'default label tagName' );
  assert.ok( aElement.textContent === TEST_LABEL, 'no html should use textContent' );

  a.innerContent = TEST_LABEL_HTML;
  assert.ok( aElement.innerHTML === TEST_LABEL_HTML, 'html label should use innerHTML' );

  a.innerContent = TEST_LABEL_HTML_2;
  assert.ok( aElement.innerHTML === TEST_LABEL_HTML_2, 'html label should use innerHTML, overwrite from html' );

  a.innerContent = null;
  assert.ok( aElement.innerHTML === '', 'innerHTML should be empty after clearing innerContent' );

  a.tagName = null;
  assert.ok( a.pdomInstances.length === 0, 'set to null should clear accessible instances' );

  // make sure that no errors when setting innerContent with tagName null.
  a.innerContent = 'hello';

  a.tagName = 'button';
  a.innerContent = TEST_LABEL_HTML_2;
  assert.ok( getPrimarySiblingElementByNode( a ).innerHTML === TEST_LABEL_HTML_2, 'innerContent not cleared when tagName set to null.' );

  // verify that setting inner content on an input is not allowed
  const b = new Node( { tagName: 'input', inputType: 'range' } );
  rootNode.addChild( b );
  window.assert && assert.throws( () => {
    b.innerContent = 'this should fail';
  }, /.*/, 'cannot set inner content on input' );

  // now that it is a div, innerContent is allowed
  b.tagName = 'div';
  assert.ok( b.tagName === 'div', 'expect tagName setter to work.' );
  b.innerContent = TEST_LABEL;
  assert.ok( b.innerContent === TEST_LABEL, 'inner content allowed' );

  // revert tag name to input, should throw an error
  window.assert && assert.throws( () => {
    b.tagName = 'input';
  }, /.*/, 'error thrown after setting tagName to input on Node with innerContent.' );

  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );
} );


QUnit.test( 'containerTagName option', assert => {

  const rootNode = new Node( { tagName: 'div' } );
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  // create some nodes for testing
  const a = new Node( { tagName: 'button' } );

  rootNode.addChild( a );
  assert.ok( a.pdomInstances.length === 1, 'only 1 instance' );
  assert.ok( a.pdomInstances[ 0 ].peer!.containerParent === null, 'no container parent for just button' );
  assert.ok( rootNode[ '_pdomInstances' ][ 0 ].peer!.primarySibling!.children[ 0 ] === a[ '_pdomInstances' ][ 0 ].peer!.primarySibling!,
    'rootNode peer should hold node a\'s peer in the PDOM' );

  a.containerTagName = 'div';

  assert.ok( a.pdomInstances[ 0 ].peer!.containerParent!.id.includes( 'container' ), 'container parent is div if specified' );
  assert.ok( rootNode[ '_pdomInstances' ][ 0 ].peer!.primarySibling!.children[ 0 ] === a[ '_pdomInstances' ][ 0 ].peer!.containerParent!,
    'container parent is div if specified' );

  a.containerTagName = null;

  assert.ok( !a.pdomInstances[ 0 ].peer!.containerParent!, 'container parent is cleared if specified' );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );
} );

QUnit.test( 'labelTagName/labelContent option', assert => {

  const rootNode = new Node( { tagName: 'div' } );
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  // create some nodes for testing
  const a = new Node( { tagName: 'button', labelContent: TEST_LABEL } );

  rootNode.addChild( a );

  const aElement = getPrimarySiblingElementByNode( a );
  const labelSibling = aElement.parentElement!.childNodes[ 0 ] as HTMLElement;
  assert.ok( a.pdomInstances.length === 1, 'only 1 instance' );
  assert.ok( aElement.parentElement!.childNodes.length === 2, 'parent contains two siblings' );
  assert.ok( labelSibling.tagName === DEFAULT_LABEL_TAG_NAME, 'default label tagName' );
  assert.ok( labelSibling.textContent === TEST_LABEL, 'no html should use textContent' );

  a.labelContent = TEST_LABEL_HTML;
  assert.ok( labelSibling.innerHTML === TEST_LABEL_HTML, 'html label should use innerHTML' );

  a.labelContent = null;
  assert.ok( labelSibling.innerHTML === '', 'label content should be empty after setting to null' );

  a.labelContent = TEST_LABEL_HTML_2;
  assert.ok( labelSibling.innerHTML === TEST_LABEL_HTML_2, 'html label should use innerHTML, overwrite from html' );

  a.tagName = 'div';

  const newAElement = getPrimarySiblingElementByNode( a );
  const newLabelSibling = newAElement.parentElement!.childNodes[ 0 ] as HTMLElement;

  assert.ok( newLabelSibling.innerHTML === TEST_LABEL_HTML_2, 'tagName independent of: html label should use innerHTML, overwrite from html' );

  a.labelTagName = null;

  // make sure label was cleared from PDOM
  assert.ok( getPrimarySiblingElementByNode( a ).parentElement!.childNodes.length === 1,
    'Only one element after clearing label' );

  assert.ok( a.labelContent === TEST_LABEL_HTML_2, 'clearing labelTagName should not change content, even  though it is not displayed' );

  a.labelTagName = 'p';
  assert.ok( a.labelTagName === 'p', 'expect labelTagName setter to work.' );

  const b = new Node( { tagName: 'p', labelContent: 'I am groot' } );
  rootNode.addChild( b );
  let bLabelElement = document.getElementById( b.pdomInstances[ 0 ].peer!.labelSibling!.id )!;
  assert.ok( !bLabelElement.getAttribute( 'for' ), 'for attribute should not be on non label label sibling.' );
  b.labelTagName = 'label';
  bLabelElement = document.getElementById( b.pdomInstances[ 0 ].peer!.labelSibling!.id )!;
  assert.ok( bLabelElement.getAttribute( 'for' ) !== null, 'for attribute should be on "label" tag for label sibling.' );

  const c = new Node( { tagName: 'p' } );
  rootNode.addChild( c );
  c.labelTagName = 'label';
  c.labelContent = TEST_LABEL;
  const cLabelElement = document.getElementById( c.pdomInstances[ 0 ].peer!.labelSibling!.id )!;
  assert.ok( cLabelElement.getAttribute( 'for' ) !== null, 'order should not matter' );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );

} );

QUnit.test( 'container element not needed for multiple siblings', assert => {

  const rootNode = new Node( { tagName: 'div' } );
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  // test containerTag is not needed
  const b = new Node( {
    tagName: 'div',
    labelContent: 'hello'
  } );
  const c = new Node( {
    tagName: 'section',
    labelContent: 'hi'
  } );
  const d = new Node( {
    tagName: 'p',
    innerContent: 'PPPP',
    containerTagName: 'div'
  } );
  rootNode.addChild( b );
  b.addChild( c );
  b.addChild( d );
  let bElement = getPrimarySiblingElementByNode( b );
  let cPeer = c.pdomInstances[ 0 ].peer!;
  let dPeer = d.pdomInstances[ 0 ].peer!;
  assert.ok( bElement.children.length === 3, 'c.p, c.section, d.div should all be on the same level' );
  const confirmOriginalOrder = () => {
    assert.ok( bElement.children[ 0 ].tagName === 'P', 'p first' );
    assert.ok( bElement.children[ 1 ].tagName === 'SECTION', 'section 2nd' );
    assert.ok( bElement.children[ 2 ].tagName === 'DIV', 'div 3rd' );
    assert.ok( bElement.children[ 0 ] === cPeer.labelSibling, 'c label first' );
    assert.ok( bElement.children[ 1 ] === cPeer.primarySibling, 'c primary 2nd' );
    assert.ok( bElement.children[ 2 ] === dPeer.containerParent, 'd container 3rd' );
  };
  confirmOriginalOrder();

  // add a few more
  const e = new Node( {
    tagName: 'span',
    descriptionContent: '<br>sweet and cool things</br>'
  } );
  b.addChild( e );
  bElement = getPrimarySiblingElementByNode( b ); // refresh the DOM Elements
  cPeer = c.pdomInstances[ 0 ].peer!; // refresh the DOM Elements
  dPeer = d.pdomInstances[ 0 ].peer!; // refresh the DOM Elements
  let ePeer = e.pdomInstances[ 0 ].peer!;
  assert.ok( bElement.children.length === 5, 'e children should be added to the same PDOM level.' );
  confirmOriginalOrder();

  const confirmOriginalWithE = () => {
    assert.ok( bElement.children[ 3 ].tagName === 'P', 'P 4rd' );
    assert.ok( bElement.children[ 4 ].tagName === 'SPAN', 'SPAN 3rd' );
    assert.ok( bElement.children[ 3 ] === ePeer.descriptionSibling, 'e description 4th' );
    assert.ok( bElement.children[ 4 ] === ePeer.primarySibling, 'e primary 5th' );
  };

  // dynamically adding parent
  e.containerTagName = 'article';
  bElement = getPrimarySiblingElementByNode( b ); // refresh the DOM Elements
  cPeer = c.pdomInstances[ 0 ].peer!; // refresh the DOM Elements
  dPeer = d.pdomInstances[ 0 ].peer!; // refresh the DOM Elements
  ePeer = e.pdomInstances[ 0 ].peer!;
  assert.ok( bElement.children.length === 4, 'e children should now be under e\'s container.' );
  confirmOriginalOrder();
  assert.ok( bElement.children[ 3 ].tagName === 'ARTICLE', 'SPAN 3rd' );
  assert.ok( bElement.children[ 3 ] === ePeer.containerParent, 'e parent 3rd' );

  // clear container
  e.containerTagName = null;
  bElement = getPrimarySiblingElementByNode( b ); // refresh the DOM Elements
  cPeer = c.pdomInstances[ 0 ].peer!; // refresh the DOM Elements
  dPeer = d.pdomInstances[ 0 ].peer!; // refresh the DOM Elements
  ePeer = e.pdomInstances[ 0 ].peer!;
  assert.ok( bElement.children.length === 5, 'e children should be added to the same PDOM level again.' );
  confirmOriginalOrder();
  confirmOriginalWithE();

  // proper disposal
  e.dispose();
  bElement = getPrimarySiblingElementByNode( b );
  assert.ok( bElement.children.length === 3, 'e children should have been removed' );
  assert.ok( e.pdomInstances.length === 0, 'e is disposed' );
  confirmOriginalOrder();

  // reorder d correctly when c removed
  b.removeChild( c );
  assert.ok( bElement.children.length === 1, 'c children should have been removed, only d container' );
  bElement = getPrimarySiblingElementByNode( b );
  dPeer = d.pdomInstances[ 0 ].peer!;
  assert.ok( bElement.children[ 0 ].tagName === 'DIV', 'DIV first' );
  assert.ok( bElement.children[ 0 ] === dPeer.containerParent, 'd container first' );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );

} );

QUnit.test( 'pdomOrder tests', assert => {

  const rootNode = new Node( { tagName: 'div' } );
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  const firstChild = new Node( { tagName: 'div' } );
  const a = new Node( { tagName: 'button' } );
  const b = new Node( { tagName: 'button' } );
  const c = new Node( { tagName: 'button' } );
  const d = new Node( { tagName: 'button' } );

  const arraysEqual = ( a: ( Node | null )[], b: ( Node | null )[] ) => {
    return a.length === b.length && a.every( ( value, index ) => value === b[ index ] );
  };

  rootNode.addChild( firstChild );
  firstChild.children = [ a, b, c, d ];

  //------------------------------------------------------------------
  // Basic setter/getter tests
  //------------------------------------------------------------------

  // pdomOrder is initially null
  assert.ok( a.pdomOrder === null, 'pdomOrder is initially null' );

  const firstOrder = [ c, a, d, b ];

  // set the pdomOrder
  firstChild.pdomOrder = firstOrder;
  assert.ok( arraysEqual( firstChild.pdomOrder, firstOrder ), 'pdomOrder is set correctly' );

  // New order should be applied
  const secondOrder = [ a, b, c, d ];
  firstChild.pdomOrder = secondOrder;
  assert.ok( arraysEqual( firstChild.pdomOrder, secondOrder ), 'pdomOrder is set correctly' );

  // Try providing the same instance of an array to pdomOrder
  const thirdOrder = firstChild.pdomOrder;
  const e = new Node( { tagName: 'button' } );
  thirdOrder.splice( 1, 0, e );
  firstChild.pdomOrder = thirdOrder;
  assert.ok( arraysEqual( firstChild.pdomOrder, thirdOrder ), 'pdomOrder is set correctly' );

  // Try removing an element from the pdomOrder
  const fourthOrder = firstChild.pdomOrder.splice( 1, 1 );
  firstChild.pdomOrder = fourthOrder;
  assert.ok( arraysEqual( firstChild.pdomOrder, fourthOrder ), 'pdomOrder is set correctly' );

  // Clear the pdomOrder
  firstChild.pdomOrder = null;
  assert.ok( firstChild.pdomOrder === null, 'pdomOrder is cleared' );

  //------------------------------------------------------------------
  // disposing a Node should remove it from any pdomOrder
  //------------------------------------------------------------------
  // These Nodes are named based on their place in pdomOrder - all will be children of firstChild
  const parent = new Node( { tagName: 'div', labelContent: 'parent' } );
  const child1 = new Node( { tagName: 'div', labelContent: 'child1' } );
  const child2 = new Node( { tagName: 'div', labelContent: 'child2' } );
  const grandchild1 = new Node( { tagName: 'div', labelContent: 'grandchild1' } );
  const grandchild2 = new Node( { tagName: 'div', labelContent: 'grandchild2' } );
  firstChild.children = [ parent, child1, child2, grandchild1, grandchild2 ];

  // Setup nested pdomOrders
  parent.pdomOrder = [ child1, child2 ];
  child1.pdomOrder = [ grandchild1 ];
  child2.pdomOrder = [ grandchild2 ];

  // Verify initial pdomOrder setup
  assert.ok( arraysEqual( parent.pdomOrder, [ child1, child2 ] ), 'parent pdomOrder is set correctly' );
  assert.ok( arraysEqual( child1.pdomOrder, [ grandchild1 ] ), 'child1 pdomOrder is set correctly' );
  assert.ok( arraysEqual( child2.pdomOrder, [ grandchild2 ] ), 'child2 pdomOrder is set correctly' );

  // Dispose a grandchild and verify changes propagate correctly
  grandchild1.dispose();
  assert.ok( arraysEqual( child1.pdomOrder, [] ), 'child1 pdomOrder is updated when grandchild1 is disposed' );
  assert.ok( arraysEqual( parent.pdomOrder, [ child1, child2 ] ), 'parent pdomOrder remains unchanged when grandchild1 is disposed' );

  // Dispose a child and verify changes propagate to parent
  child2.dispose();
  assert.ok( arraysEqual( parent.pdomOrder, [ child1 ] ), 'parent pdomOrder is updated when child2 is disposed' );
  const parentPeer = getPDOMPeerByNode( parent );
  assert.ok( parentPeer.primarySibling!.children.length === 2, 'child2 was disposed, so only child1 label/primary are left' );

  // Dispose the remaining child to check the parent pdomOrder
  child1.dispose();
  assert.ok( arraysEqual( parent.pdomOrder, [] ), 'parent pdomOrder is empty when both children are disposed' );

  //------------------------------------------------------------------
  // Done with this tests, clean up
  //------------------------------------------------------------------
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );
} );

QUnit.test( 'descriptionTagName/descriptionContent option', assert => {

  const rootNode = new Node( { tagName: 'div' } );
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  // create some nodes for testing
  const a = new Node( { tagName: 'button', descriptionContent: TEST_DESCRIPTION } );

  rootNode.addChild( a );

  const aElement = getPrimarySiblingElementByNode( a );
  const descriptionSibling = aElement.parentElement!.childNodes[ 0 ] as HTMLElement;
  assert.ok( a.pdomInstances.length === 1, 'only 1 instance' );
  assert.ok( aElement.parentElement!.childNodes.length === 2, 'parent contains two siblings' );
  assert.ok( descriptionSibling.tagName === DEFAULT_DESCRIPTION_TAG_NAME, 'default label tagName' );
  assert.ok( descriptionSibling.textContent === TEST_DESCRIPTION, 'no html should use textContent' );

  a.descriptionContent = TEST_DESCRIPTION_HTML;
  assert.ok( descriptionSibling.innerHTML === TEST_DESCRIPTION_HTML, 'html label should use innerHTML' );

  a.descriptionContent = null;
  assert.ok( descriptionSibling.innerHTML === '', 'description content should be cleared' );

  a.descriptionContent = TEST_DESCRIPTION_HTML_2;
  assert.ok( descriptionSibling.innerHTML === TEST_DESCRIPTION_HTML_2, 'html label should use innerHTML, overwrite from html' );

  a.descriptionTagName = null;

  // make sure description was cleared from PDOM
  assert.ok( getPrimarySiblingElementByNode( a ).parentElement!.childNodes.length === 1,
    'Only one element after clearing description' );

  assert.ok( a.descriptionContent === TEST_DESCRIPTION_HTML_2, 'clearing descriptionTagName should not change content, even  though it is not displayed' );

  assert.ok( a.descriptionTagName === null, 'expect descriptionTagName setter to work.' );

  a.descriptionTagName = 'p';
  assert.ok( a.descriptionTagName === 'p', 'expect descriptionTagName setter to work.' );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );

} );

QUnit.test( 'ParallelDOM options', assert => {

  const rootNode = new Node();
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  // test setting of accessible content through options
  const buttonNode = new Node( {
    focusHighlight: new Circle( 5 ),
    containerTagName: 'div', // contained in parent element 'div'
    tagName: 'input', // dom element with tag name 'input'
    inputType: 'button', // input type 'button'
    labelTagName: 'label', // label with tagname 'label'
    labelContent: TEST_LABEL, // label text content
    descriptionContent: TEST_DESCRIPTION, // description text content
    focusable: false, // remove from focus order
    ariaRole: 'button' // uses the ARIA button role
  } );
  rootNode.addChild( buttonNode );

  const divNode = new Node( {
    tagName: 'div',
    ariaLabel: TEST_LABEL, // use ARIA label attribute
    pdomVisible: false, // hidden from screen readers (and browser)
    descriptionContent: TEST_DESCRIPTION, // default to a <p> tag
    containerTagName: 'div'
  } );
  rootNode.addChild( divNode );

  // verify that setters and getters worked correctly
  assert.ok( buttonNode.labelTagName === 'label', 'Label tag name' );
  assert.ok( buttonNode.containerTagName === 'div', 'container tag name' );
  assert.ok( buttonNode.labelContent === TEST_LABEL, 'Accessible label' );
  assert.ok( buttonNode.descriptionTagName!.toUpperCase() === DEFAULT_DESCRIPTION_TAG_NAME, 'Description tag name' );
  assert.equal( buttonNode.focusable, false, 'Focusable' );
  assert.ok( buttonNode.ariaRole === 'button', 'Aria role' );
  assert.ok( buttonNode.descriptionContent === TEST_DESCRIPTION, 'Accessible Description' );
  assert.ok( buttonNode.focusHighlight instanceof Circle, 'Focus highlight' );
  assert.ok( buttonNode.tagName === 'input', 'Tag name' );
  assert.ok( buttonNode.inputType === 'button', 'Input type' );

  assert.ok( divNode.tagName === 'div', 'Tag name' );
  assert.ok( divNode.ariaLabel === TEST_LABEL, 'Use aria label' );
  assert.equal( divNode.pdomVisible, false, 'pdom visible' );
  assert.ok( divNode.labelTagName === null, 'Label tag name with aria label is independent' );
  assert.ok( divNode.descriptionTagName!.toUpperCase() === DEFAULT_DESCRIPTION_TAG_NAME, 'Description tag name' );

  // verify DOM structure - options above should create something like:
  // <div id="display-root">
  //  <div id="parent-container-id">
  //    <label for="id">Test Label</label>
  //    <p>Description>Test Description</p>
  //    <input type='button' role='button' tabindex="-1" id=id>
  //  </div>
  //
  //  <div aria-label="Test Label" hidden aria-labelledBy="button-node-id" aria-describedby='button-node-id'>
  //    <p>Test Description</p>
  //  </div>
  // </div>
  pdomAuditRootNode( rootNode );
  let buttonElement = getPrimarySiblingElementByNode( buttonNode );

  const buttonParent = buttonElement.parentNode! as HTMLElement;
  const buttonPeers = buttonParent.childNodes as unknown as HTMLElement[];
  const buttonLabel = buttonPeers[ 0 ];
  const buttonDescription = buttonPeers[ 1 ];
  const divElement = getPrimarySiblingElementByNode( divNode );
  const pDescription = divElement.parentElement!.childNodes[ 0 ]; // description before primary div

  assert.ok( buttonParent.tagName === 'DIV', 'parent container' );
  assert.ok( buttonLabel.tagName === 'LABEL', 'Label first' );
  assert.ok( buttonLabel.getAttribute( 'for' ) === buttonElement.id, 'label for attribute' );
  assert.ok( buttonLabel.textContent === TEST_LABEL, 'label content' );
  assert.ok( buttonDescription.tagName === DEFAULT_DESCRIPTION_TAG_NAME, 'description second' );
  assert.equal( buttonDescription.textContent, TEST_DESCRIPTION, 'description content' );
  assert.ok( buttonPeers[ 2 ] === buttonElement, 'Button third' );
  assert.ok( buttonElement.getAttribute( 'type' ) === 'button', 'input type set' );
  assert.ok( buttonElement.getAttribute( 'role' ) === 'button', 'button role set' );
  assert.ok( buttonElement.tabIndex === -1, 'not focusable' );

  assert.ok( divElement.getAttribute( 'aria-label' ) === TEST_LABEL, 'aria label set' );
  assert.ok( divElement.parentElement!.hidden, 'hidden set should act on parent' );
  assert.ok( pDescription.textContent === TEST_DESCRIPTION, 'description content' );
  assert.ok( pDescription.parentElement === divElement.parentElement, 'description is sibling to primary' );
  assert.ok( divElement.parentElement!.childNodes.length === 2, 'no label element for aria-label, just description and primary siblings' );

  // clear values
  buttonNode.inputType = null;
  buttonElement = getPrimarySiblingElementByNode( buttonNode );
  assert.ok( buttonElement.getAttribute( 'type' ) === null, 'input type cleared' );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );

} );

// tests for aria-labelledby and aria-describedby should be the same, since both support the same feature set
function testAssociationAttribute( assert: Assert, attribute: string ): void {

  // use a different setter depending on if testing labelledby or describedby
  const addAssociationFunction = attribute === 'aria-labelledby' ? 'addAriaLabelledbyAssociation' :
                                 attribute === 'aria-describedby' ? 'addAriaDescribedbyAssociation' :
                                 attribute === 'aria-activedescendant' ? 'addActiveDescendantAssociation' :
                                 null;

  if ( !addAssociationFunction ) {
    throw new Error( 'incorrect attribute name while in testAssociationAttribute' );
  }

  const rootNode = new Node();
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  // two new nodes that will be related with the aria-labelledby and aria-describedby associations
  const a = new Node( { tagName: 'button', labelTagName: 'p', descriptionTagName: 'p' } );
  const b = new Node( { tagName: 'p', innerContent: TEST_LABEL_2 } );
  rootNode.children = [ a, b ];

  window.assert && assert.throws( () => {
    a.setPDOMAttribute( attribute, 'hello' );
  }, /.*/, 'cannot set association attributes with setPDOMAttribute' );

  a[ addAssociationFunction ]( {
    otherNode: b,
    thisElementName: PDOMPeer.PRIMARY_SIBLING,
    otherElementName: PDOMPeer.PRIMARY_SIBLING
  } );

  let aElement = getPrimarySiblingElementByNode( a );
  let bElement = getPrimarySiblingElementByNode( b );
  assert.ok( aElement.getAttribute( attribute )!.includes( bElement.id ), `${attribute} for one node.` );

  const c = new Node( { tagName: 'div', innerContent: TEST_LABEL } );
  rootNode.addChild( c );

  a[ addAssociationFunction ]( {
    otherNode: c,
    thisElementName: PDOMPeer.PRIMARY_SIBLING,
    otherElementName: PDOMPeer.PRIMARY_SIBLING
  } );

  aElement = getPrimarySiblingElementByNode( a );
  bElement = getPrimarySiblingElementByNode( b );
  let cElement = getPrimarySiblingElementByNode( c );
  const expectedValue = [ bElement.id, cElement.id ].join( ' ' );
  assert.ok( aElement.getAttribute( attribute ) === expectedValue, `${attribute} two nodes` );

  // Make c invalidate
  rootNode.removeChild( c );
  rootNode.addChild( new Node( { children: [ c ] } ) );

  const oldValue = expectedValue;

  aElement = getPrimarySiblingElementByNode( a );
  cElement = getPrimarySiblingElementByNode( c );

  assert.ok( aElement.getAttribute( attribute ) !== oldValue, 'should have invalidated on tree change' );
  assert.ok( aElement.getAttribute( attribute ) === [ bElement.id, cElement.id ].join( ' ' ),
    'should have invalidated on tree change' );

  const d = new Node( { tagName: 'div', descriptionTagName: 'p', innerContent: TEST_LABEL, containerTagName: 'div' } );
  rootNode.addChild( d );

  b[ addAssociationFunction ]( {
    otherNode: d,
    thisElementName: PDOMPeer.CONTAINER_PARENT,
    otherElementName: PDOMPeer.DESCRIPTION_SIBLING
  } );
  b.containerTagName = 'div';

  const bParentContainer = getPrimarySiblingElementByNode( b ).parentElement!;
  const dDescriptionElement = getPrimarySiblingElementByNode( d ).parentElement!.childNodes[ 0 ] as HTMLElement;
  assert.ok( bParentContainer.getAttribute( attribute ) !== oldValue, 'should have invalidated on tree change' );
  assert.ok( bParentContainer.getAttribute( attribute ) === dDescriptionElement.id,
    `b parent container element is ${attribute} d description sibling` );

  // say we have a scene graph that looks like:
  //    e
  //     \
  //      f
  //       \
  //        g
  //         \
  //          h
  // we want to make sure
  const e = new Node( { tagName: 'div' } );
  const f = new Node( { tagName: 'div' } );
  const g = new Node( { tagName: 'div' } );
  const h = new Node( { tagName: 'div' } );
  e.addChild( f );
  f.addChild( g );
  g.addChild( h );
  rootNode.addChild( e );

  e[ addAssociationFunction ]( {
    otherNode: f,
    thisElementName: PDOMPeer.PRIMARY_SIBLING,
    otherElementName: PDOMPeer.PRIMARY_SIBLING
  } );

  f[ addAssociationFunction ]( {
    otherNode: g,
    thisElementName: PDOMPeer.PRIMARY_SIBLING,
    otherElementName: PDOMPeer.PRIMARY_SIBLING
  } );

  g[ addAssociationFunction ]( {
    otherNode: h,
    thisElementName: PDOMPeer.PRIMARY_SIBLING,
    otherElementName: PDOMPeer.PRIMARY_SIBLING
  } );

  let eElement = getPrimarySiblingElementByNode( e );
  let fElement = getPrimarySiblingElementByNode( f );
  let gElement = getPrimarySiblingElementByNode( g );
  let hElement = getPrimarySiblingElementByNode( h );
  assert.ok( eElement.getAttribute( attribute ) === fElement.id, `eElement should be ${attribute} fElement` );
  assert.ok( fElement.getAttribute( attribute ) === gElement.id, `fElement should be ${attribute} gElement` );
  assert.ok( gElement.getAttribute( attribute ) === hElement.id, `gElement should be ${attribute} hElement` );

  // re-arrange the scene graph and make sure that the attribute ids remain up to date
  //    e
  //     \
  //      h
  //       \
  //        g
  //         \
  //          f
  e.removeChild( f );
  f.removeChild( g );
  g.removeChild( h );

  e.addChild( h );
  h.addChild( g );
  g.addChild( f );
  eElement = getPrimarySiblingElementByNode( e );
  fElement = getPrimarySiblingElementByNode( f );
  gElement = getPrimarySiblingElementByNode( g );
  hElement = getPrimarySiblingElementByNode( h );
  assert.ok( eElement.getAttribute( attribute ) === fElement.id, `eElement should still be ${attribute} fElement` );
  assert.ok( fElement.getAttribute( attribute ) === gElement.id, `fElement should still be ${attribute} gElement` );
  assert.ok( gElement.getAttribute( attribute ) === hElement.id, `gElement should still be ${attribute} hElement` );

  // test aria labelled by your self, but a different peer Element, multiple attribute ids included in the test.
  const containerTagName = 'div';
  const j = new Node( {
    tagName: 'button',
    labelTagName: 'label',
    descriptionTagName: 'p',
    containerTagName: containerTagName
  } );
  rootNode.children = [ j ];

  j[ addAssociationFunction ]( {
    otherNode: j,
    thisElementName: PDOMPeer.PRIMARY_SIBLING,
    otherElementName: PDOMPeer.LABEL_SIBLING
  } );

  j[ addAssociationFunction ]( {
    otherNode: j,
    thisElementName: PDOMPeer.CONTAINER_PARENT,
    otherElementName: PDOMPeer.DESCRIPTION_SIBLING
  } );

  j[ addAssociationFunction ]( {
    otherNode: j,
    thisElementName: PDOMPeer.CONTAINER_PARENT,
    otherElementName: PDOMPeer.LABEL_SIBLING
  } );

  const checkOnYourOwnAssociations = ( node: Node ) => {

    const instance = node[ '_pdomInstances' ][ 0 ];
    const nodePrimaryElement = instance.peer!.primarySibling!;
    const nodeParent = nodePrimaryElement.parentElement!;

    const getUniqueIdStringForSibling = ( siblingString: string ): string => {
      return instance.peer!.getElementId( siblingString, instance.getPDOMInstanceUniqueId() );
    };

    assert.ok( nodePrimaryElement.getAttribute( attribute )!.includes( getUniqueIdStringForSibling( 'label' ) ), `${attribute} your own label element.` );
    assert.ok( nodeParent.getAttribute( attribute )!.includes( getUniqueIdStringForSibling( 'description' ) ), `parent ${attribute} your own description element.` );

    assert.ok( nodeParent.getAttribute( attribute )!.includes( getUniqueIdStringForSibling( 'label' ) ), `parent ${attribute} your own label element.` );

  };

  // add k into the mix
  const k = new Node( { tagName: 'div' } );
  k[ addAssociationFunction ]( {
    otherNode: j,
    thisElementName: PDOMPeer.PRIMARY_SIBLING,
    otherElementName: PDOMPeer.LABEL_SIBLING
  } );
  rootNode.addChild( k );
  const testK = () => {
    const kValue = k[ '_pdomInstances' ][ 0 ].peer!.primarySibling!.getAttribute( attribute );
    const jID = j[ '_pdomInstances' ][ 0 ].peer!.labelSibling!.getAttribute( 'id' );
    assert.ok( jID === kValue, 'k pointing to j' );
  };

  // audit the content we have created
  pdomAuditRootNode( rootNode );

  // Check basic associations within single node
  checkOnYourOwnAssociations( j );
  testK();

  // Moving this node around the scene graph should not change it's aria labelled by associations.
  rootNode.addChild( new Node( { children: [ j ] } ) );
  checkOnYourOwnAssociations( j );
  testK();

  // check remove child
  rootNode.removeChild( j );
  checkOnYourOwnAssociations( j );
  testK();

  // check dispose
  const jParent = new Node( { children: [ j ] } );
  rootNode.children = [];
  rootNode.addChild( jParent );
  checkOnYourOwnAssociations( j );
  rootNode.addChild( j );
  checkOnYourOwnAssociations( j );
  rootNode.addChild( k );
  checkOnYourOwnAssociations( j );
  testK();
  jParent.dispose();
  checkOnYourOwnAssociations( j );
  testK();

  // check removeChild with dag
  const jParent2 = new Node( { children: [ j ] } );
  rootNode.insertChild( 0, jParent2 );
  checkOnYourOwnAssociations( j );
  testK();
  rootNode.removeChild( jParent2 );
  checkOnYourOwnAssociations( j );
  testK();

  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );
}

type AssociationAttribute = 'aria-labelledby' | 'aria-describedby' | 'aria-activedescendant';

function testAssociationAttributeBySetters( assert: Assert, attribute: AssociationAttribute ): void {

  const rootNode = new Node();
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  type OptionNames = 'ariaLabelledbyAssociations' | 'ariaDescribedbyAssociations' | 'activeDescendantAssociations';
  // use a different setter depending on if testing labelledby or describedby
  const associationsArrayName: OptionNames = attribute === 'aria-labelledby' ? 'ariaLabelledbyAssociations' :
                                             attribute === 'aria-describedby' ? 'ariaDescribedbyAssociations' :
                                             'activeDescendantAssociations';

  type RemovalFunctionNames = 'removeAriaLabelledbyAssociation' | 'removeAriaDescribedbyAssociation' | 'removeActiveDescendantAssociation';

  // use a different setter depending on if testing labelledby or describedby
  const associationRemovalFunction: RemovalFunctionNames = attribute === 'aria-labelledby' ? 'removeAriaLabelledbyAssociation' :
                                                           attribute === 'aria-describedby' ? 'removeAriaDescribedbyAssociation' :
                                                           'removeActiveDescendantAssociation';

  const options: ParallelDOMOptions = {
    tagName: 'p',
    labelContent: 'hi',
    descriptionContent: 'hello',
    containerTagName: 'div'
  };
  const n = new Node( options );
  rootNode.addChild( n );
  options[ associationsArrayName ] = [
    {
      otherNode: n,
      thisElementName: PDOMPeer.PRIMARY_SIBLING,
      otherElementName: PDOMPeer.LABEL_SIBLING
    }
  ];
  const o = new Node( options );
  rootNode.addChild( o );

  const nPeer = getPDOMPeerByNode( n );
  const oElement = getPrimarySiblingElementByNode( o );
  assert.ok( oElement.getAttribute( attribute )!.includes(
      nPeer.getElementId( 'label', nPeer.pdomInstance!.getPDOMInstanceUniqueId() ) ),
    `${attribute} for two nodes with setter (label).` );

  // make a list of associations to test as a setter
  const randomAssociationObject = {
    otherNode: new Node(),
    thisElementName: PDOMPeer.CONTAINER_PARENT,
    otherElementName: PDOMPeer.LABEL_SIBLING
  };
  options[ associationsArrayName ] = [
    {
      otherNode: new Node(),
      thisElementName: PDOMPeer.CONTAINER_PARENT,
      otherElementName: PDOMPeer.DESCRIPTION_SIBLING
    },
    randomAssociationObject,
    {
      otherNode: new Node(),
      thisElementName: PDOMPeer.PRIMARY_SIBLING,
      otherElementName: PDOMPeer.LABEL_SIBLING
    }
  ];

  // test getters and setters
  const m = new Node( options );
  rootNode.addChild( m );
  assert.ok( _.isEqual( m[ associationsArrayName ], options[ associationsArrayName ] ), 'test association object getter' );
  m[ associationRemovalFunction ]( randomAssociationObject );
  options[ associationsArrayName ].splice( options[ associationsArrayName ].indexOf( randomAssociationObject ), 1 );
  assert.ok( _.isEqual( m[ associationsArrayName ], options[ associationsArrayName ] ), 'test association object getter after removal' );

  m[ associationsArrayName ] = [];
  assert.ok( getPrimarySiblingElementByNode( m ).getAttribute( attribute ) === null, 'clear with setter' );

  m[ associationsArrayName ] = options[ associationsArrayName ]!;
  m.dispose();
  assert.ok( m[ associationsArrayName ].length === 0, 'cleared when disposed' );

  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );
}

QUnit.test( 'aria-labelledby', assert => {

  testAssociationAttribute( assert, 'aria-labelledby' );
  testAssociationAttributeBySetters( assert, 'aria-labelledby' );

} );
QUnit.test( 'aria-describedby', assert => {

  testAssociationAttribute( assert, 'aria-describedby' );
  testAssociationAttributeBySetters( assert, 'aria-describedby' );

} );

QUnit.test( 'aria-activedescendant', assert => {

  testAssociationAttribute( assert, 'aria-activedescendant' );
  testAssociationAttributeBySetters( assert, 'aria-activedescendant' );

} );

QUnit.test( 'ParallelDOM invalidation', assert => {

  // test invalidation of accessibility (changing content which requires )
  const a1 = new Node();
  const rootNode = new Node();

  a1.tagName = 'button';

  // accessible instances are not sorted until added to a display
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  rootNode.addChild( a1 );

  // verify that elements are in the DOM
  const a1Element = getPrimarySiblingElementByNode( a1 );
  assert.ok( a1Element, 'button in DOM' );
  assert.ok( a1Element.tagName === 'BUTTON', 'button tag name set' );

  // give the button a container parent and some empty siblings
  a1.labelTagName = 'div';
  a1.descriptionTagName = 'p';
  a1.containerTagName = 'div';

  let buttonElement = a1.pdomInstances[ 0 ].peer!.primarySibling!;
  let parentElement = buttonElement.parentElement;
  const buttonPeers = parentElement!.childNodes as unknown as HTMLElement[];

  // now html should look like
  // <div id='parent'>
  //  <div id='label'></div>
  //  <p id='description'></p>
  //  <button></button>
  // </div>
  assert.ok( document.getElementById( parentElement!.id ), 'container parent in DOM' );
  assert.ok( buttonPeers[ 0 ].tagName === 'DIV', 'label first' );
  assert.ok( buttonPeers[ 1 ].tagName === 'P', 'description second' );
  assert.ok( buttonPeers[ 2 ].tagName === 'BUTTON', 'primarySibling third' );

  // make the button a div and use an inline label, and place the description below
  a1.tagName = 'div';
  a1.appendLabel = true;
  a1.appendDescription = true;
  a1.labelTagName = null; // use aria label attribute instead
  a1.ariaLabel = TEST_LABEL;

  // now the html should look like
  // <div id='parent-id'>
  //  <div></div>
  //  <p id='description'></p>
  // </div>

  // redefine the HTML elements (references will point to old elements before mutation)
  buttonElement = a1.pdomInstances[ 0 ].peer!.primarySibling!;
  parentElement = buttonElement.parentElement;
  const newButtonPeers = parentElement!.childNodes as unknown as HTMLElement[];
  assert.ok( newButtonPeers[ 0 ] === getPrimarySiblingElementByNode( a1 ), 'div first' );
  assert.ok( newButtonPeers[ 1 ].id.includes( 'description' ), 'description after div when appending both elements' );
  assert.ok( newButtonPeers.length === 2, 'no label peer when using just aria-label attribute' );

  const elementInDom = document.getElementById( a1.pdomInstances[ 0 ].peer!.primarySibling!.id )!;
  assert.ok( elementInDom.getAttribute( 'aria-label' ) === TEST_LABEL, 'aria-label set' );

  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );
} );

QUnit.test( 'ParallelDOM setters/getters', assert => {

  const a1 = new Node( {
    tagName: 'div'
  } );
  var display = new Display( a1 ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  // set/get attributes
  let a1Element = getPrimarySiblingElementByNode( a1 );
  const initialLength = a1.getPDOMAttributes().length;
  a1.setPDOMAttribute( 'role', 'switch' );
  assert.ok( a1.getPDOMAttributes().length === initialLength + 1, 'attribute set should only add 1' );
  assert.ok( a1.getPDOMAttributes()[ a1.getPDOMAttributes().length - 1 ].attribute === 'role', 'attribute set' );
  assert.ok( a1Element.getAttribute( 'role' ) === 'switch', 'HTML attribute set' );
  assert.ok( a1.hasPDOMAttribute( 'role' ), 'should have pdom attribute' );

  a1.removePDOMAttribute( 'role' );
  assert.ok( !a1.hasPDOMAttribute( 'role' ), 'should not have pdom attribute' );
  assert.ok( !a1Element.getAttribute( 'role' ), 'attribute removed' );

  const b = new Node( { focusable: true } );
  a1.addChild( b );
  b.tagName = 'div';
  assert.ok( getPrimarySiblingElementByNode( b ).tabIndex >= 0, 'set tagName after focusable' );

  // test setting attribute as DOM property, should NOT have attribute value pair (DOM uses empty string for empty)
  a1.setPDOMAttribute( 'hidden', true, { type: 'property' } );
  a1Element = getPrimarySiblingElementByNode( a1 );
  assert.equal( a1Element.hidden, true, 'hidden set as Property' );
  assert.ok( a1Element.getAttribute( 'hidden' ) === '', 'hidden should not be set as attribute' );


  // test setting and removing PDOM classes
  a1.setPDOMClass( TEST_CLASS_ONE );
  assert.ok( getPrimarySiblingElementByNode( a1 ).classList.contains( TEST_CLASS_ONE ), 'TEST_CLASS_ONE missing from classList' );

  // two classes
  a1.setPDOMClass( TEST_CLASS_TWO );
  a1Element = getPrimarySiblingElementByNode( a1 );
  assert.ok( a1Element.classList.contains( TEST_CLASS_ONE ) && a1Element.classList.contains( TEST_CLASS_ONE ), 'One of the classes missing from classList' );

  // modify the Node in a way that would cause a full redraw, make sure classes still exist
  a1.tagName = 'button';
  a1Element = getPrimarySiblingElementByNode( a1 );
  assert.ok( a1Element.classList.contains( TEST_CLASS_ONE ) && a1Element.classList.contains( TEST_CLASS_ONE ), 'One of the classes missing from classList after changing tagName' );

  // remove them one at a time
  a1.removePDOMClass( TEST_CLASS_ONE );
  a1Element = getPrimarySiblingElementByNode( a1 );
  assert.ok( !a1Element.classList.contains( TEST_CLASS_ONE ), 'TEST_CLASS_ONE should be removed from classList' );
  assert.ok( a1Element.classList.contains( TEST_CLASS_TWO ), 'TEST_CLASS_TWO should still be in classList' );

  a1.removePDOMClass( TEST_CLASS_TWO );
  a1Element = getPrimarySiblingElementByNode( a1 );
  assert.ok( !a1Element.classList.contains( TEST_CLASS_ONE ) && !a1Element.classList.contains( TEST_CLASS_ONE ), 'classList should not contain any added classes' );

  pdomAuditRootNode( a1 );

  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );
} );

QUnit.test( 'Next/Previous focusable', assert => {
  if ( !canRunTests ) {
    assert.ok( true, 'Skipping test because document does not have focus' );
    return;
  }

  // Especially important for puppeteer which doesn't support focus/blur events
  // see https://github.com/phetsims/aqua/issues/134
  if ( !document.hasFocus() ) {
    assert.ok( true, 'Unable to run focus tests if document does not have focus.' );
  }
  else {
    const util = PDOMUtils;

    const rootNode = new Node( { tagName: 'div', focusable: true } );
    var display = new Display( rootNode ); // eslint-disable-line no-var
    display.initializeEvents();
    document.body.appendChild( display.domElement );

    // invisible is deprecated don't use in future, this is a workaround for Nodes without bounds
    const a = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    const b = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    const c = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    const d = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    const e = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    rootNode.children = [ a, b, c, d ];

    assert.ok( a.focusable, 'should be focusable' );

    // get dom elements from the body
    const rootElement = getPrimarySiblingElementByNode( rootNode );
    const aElement = getPrimarySiblingElementByNode( a );
    const bElement = getPrimarySiblingElementByNode( b );
    const cElement = getPrimarySiblingElementByNode( c );
    const dElement = getPrimarySiblingElementByNode( d );

    a.focus();
    assert.ok( document.activeElement!.id === aElement.id, 'a in focus (next)' );

    util.getNextFocusable( rootElement ).focus();
    assert.ok( document.activeElement!.id === bElement.id, 'b in focus (next)' );

    util.getNextFocusable( rootElement ).focus();
    assert.ok( document.activeElement!.id === cElement.id, 'c in focus (next)' );

    util.getNextFocusable( rootElement ).focus();
    assert.ok( document.activeElement!.id === dElement.id, 'd in focus (next)' );

    util.getNextFocusable( rootElement ).focus();
    assert.ok( document.activeElement!.id === dElement.id, 'd still in focus (next)' );

    util.getPreviousFocusable( rootElement ).focus();
    assert.ok( document.activeElement!.id === cElement.id, 'c in focus (previous)' );

    util.getPreviousFocusable( rootElement ).focus();
    assert.ok( document.activeElement!.id === bElement.id, 'b in focus (previous)' );

    util.getPreviousFocusable( rootElement ).focus();
    assert.ok( document.activeElement!.id === aElement.id, 'a in focus (previous)' );

    util.getPreviousFocusable( rootElement ).focus();
    assert.ok( document.activeElement!.id === aElement.id, 'a still in focus (previous)' );

    rootNode.removeAllChildren();
    rootNode.addChild( a );
    a.children = [ b, c ];
    c.addChild( d );
    d.addChild( e );

    // this should hide everything except a
    b.focusable = false;
    c.pdomVisible = false;

    a.focus();
    util.getNextFocusable( rootElement ).focus();
    assert.ok( document.activeElement!.id === aElement.id, 'a only element focusable' );

    pdomAuditRootNode( rootNode );
    display.dispose();
    display.domElement.parentElement!.removeChild( display.domElement );

    // NOTE: The FocusManager should not be detached here, it is used globally and is needed for other tests.
  }
} );

QUnit.test( 'Remove accessibility subtree', assert => {
  const rootNode = new Node( { tagName: 'div', focusable: true } );
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  const a = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
  const b = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
  const c = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
  const d = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
  const e = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
  const f = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
  rootNode.children = [ a, b, c, d, e ];
  d.addChild( f );

  let rootDOMElement = getPrimarySiblingElementByNode( rootNode );
  let dDOMElement = getPrimarySiblingElementByNode( d );

  // verify the dom
  assert.ok( rootDOMElement.children.length === 5, 'children added' );

  // redefine because the dom element references above have become stale
  rootDOMElement = getPrimarySiblingElementByNode( rootNode );
  dDOMElement = getPrimarySiblingElementByNode( d );
  assert.ok( rootDOMElement.children.length === 5, 'children added back' );
  assert.ok( dDOMElement.children.length === 1, 'descendant child added back' );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );

} );

QUnit.test( 'accessible-dag', assert => {

  // test accessibility for multiple instances of a node
  const rootNode = new Node( { tagName: 'div', focusable: true } );
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  const a = new Node( { tagName: 'div' } );
  const b = new Node( { tagName: 'div' } );
  const c = new Node( { tagName: 'div' } );
  const d = new Node( { tagName: 'div' } );
  const e = new Node( { tagName: 'div' } );

  rootNode.addChild( a );
  a.children = [ b, c, d ];

  // e has three parents (DAG)
  b.addChild( e );
  c.addChild( e );
  d.addChild( e );

  // each instance should have its own accessible content, HTML should look like
  // <div id="root">
  //   <div id="a">
  //     <div id="b">
  //       <div id="e-instance1">
  //     <div id="c">
  //       <div id="e-instance2">
  //     <div id="d">
  //       <div id="e-instance2">
  const instances = e.pdomInstances;
  assert.ok( e.pdomInstances.length === 3, 'node e should have 3 accessible instances' );
  assert.ok( ( instances[ 0 ].peer!.primarySibling!.id !== instances[ 1 ].peer!.primarySibling!.id ) &&
             ( instances[ 1 ].peer!.primarySibling!.id !== instances[ 2 ].peer!.primarySibling!.id ) &&
             ( instances[ 0 ].peer!.primarySibling!.id !== instances[ 2 ].peer!.primarySibling!.id ), 'each dom element should be unique' );
  assert.ok( document.getElementById( instances[ 0 ].peer!.primarySibling!.id ), 'peer primarySibling 0 should be in the DOM' );
  assert.ok( document.getElementById( instances[ 1 ].peer!.primarySibling!.id ), 'peer primarySibling 1 should be in the DOM' );
  assert.ok( document.getElementById( instances[ 2 ].peer!.primarySibling!.id ), 'peer primarySibling 2 should be in the DOM' );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );

} );

QUnit.test( 'replaceChild', assert => {

  // this suite involves focus tests which do not work on headless puppeteer
  if ( !document.hasFocus() ) {
    assert.ok( true, 'Unable to run focus tests if document does not have focus.' );
  }
  else {


    // test the behavior of replaceChild function
    const rootNode = new Node( { tagName: 'div' } );
    var display = new Display( rootNode ); // eslint-disable-line no-var
    document.body.appendChild( display.domElement );

    display.initializeEvents();

    // create some nodes for testing
    const a = new Node( { tagName: 'button', focusHighlight: focusHighlight } );
    const b = new Node( { tagName: 'button', focusHighlight: focusHighlight } );
    const c = new Node( { tagName: 'button', focusHighlight: focusHighlight } );
    const d = new Node( { tagName: 'button', focusHighlight: focusHighlight } );
    const e = new Node( { tagName: 'button', focusHighlight: focusHighlight } );
    const f = new Node( { tagName: 'button', focusHighlight: focusHighlight } );

    // a child that will be added through replaceChild()
    const testNode = new Node( { tagName: 'button', focusHighlight: focusHighlight } );

    // make sure replaceChild puts the child in the right spot
    a.children = [ b, c, d, e, f ];
    const initIndex = a.indexOfChild( e );
    a.replaceChild( e, testNode );
    const afterIndex = a.indexOfChild( testNode );

    assert.ok( a.hasChild( testNode ), 'a should have child testNode after it replaced node e' );
    assert.ok( !a.hasChild( e ), 'a should no longer have child node e after it was replaced by testNode' );
    assert.ok( initIndex === afterIndex, 'testNode should be at the same place as e was after replaceChild' );

    // create a scene graph to test how scenery manages focus
    //    a
    //   / \
    //  f   b
    //     / \
    //    c   d
    //     \ /
    //      e
    a.removeAllChildren();
    rootNode.addChild( a );
    a.children = [ f, b ];
    b.children = [ c, d ];
    c.addChild( e );
    d.addChild( e );

    f.focus();
    assert.ok( f.focused, 'f has focus before being replaced' );

    // replace f with testNode, ensure that testNode receives focus after replacing
    a.replaceChild( f, testNode );
    assert.ok( !a.hasChild( f ), 'a should no longer have child f' );
    assert.ok( a.hasChild( testNode ), 'a should now have child testNode' );
    assert.ok( !f.focused, 'f no longer has focus after being replaced' );
    assert.ok( testNode.focused, 'testNode has focus after replacing focused node f' );
    assert.ok( testNode.pdomInstances[ 0 ].peer!.primarySibling === document.activeElement, 'browser is focusing testNode' );

    testNode.blur();
    assert.ok( !!testNode, 'testNode blurred before being replaced' );

    // replace testNode with f after bluring testNode, neither should have focus after the replacement
    a.replaceChild( testNode, f );
    assert.ok( a.hasChild( f ), 'node f should replace node testNode' );
    assert.ok( !a.hasChild( testNode ), 'testNode should no longer be a child of node a' );
    assert.ok( !testNode.focused, 'testNode should not have focus after being replaced' );
    assert.ok( !f.focused, 'f should not have focus after replacing testNode, testNode did not have focus' );
    assert.ok( f.pdomInstances[ 0 ].peer!.primarySibling !== document.activeElement, 'browser should not be focusing node f' );

    // focus node d and replace with non-focusable testNode, neither should have focus since testNode is not focusable
    d.focus();
    testNode.focusable = false;
    assert.ok( d.focused, 'd has focus before being replaced' );
    assert.ok( !testNode.focusable, 'testNode is not focusable before replacing node d' );

    b.replaceChild( d, testNode );
    assert.ok( b.hasChild( testNode ), 'testNode should be a child of node b after replacing with replaceChild' );
    assert.ok( !b.hasChild( d ), 'd should not be a child of b after it was replaced with replaceChild' );
    assert.ok( !d.focused, 'd does not have focus after being replaced by testNode' );
    assert.ok( !testNode.focused, 'testNode does not have focus after replacing node d (testNode is not focusable)' );

    display.dispose();
    display.domElement.parentElement!.removeChild( display.domElement );
  }
} );

QUnit.test( 'pdomVisible', assert => {

  const rootNode = new Node();
  const display = new Display( rootNode );
  document.body.appendChild( display.domElement );

  // test with a scene graph
  //       a
  //      / \
  //     b    c
  //        / | \
  //       d  e  f
  //           \ /
  //            g
  const a = new Node();
  const b = new Node();
  const c = new Node();
  const d = new Node();
  const e = new Node();
  const f = new Node();
  const g = new Node();

  rootNode.addChild( a );
  a.children = [ b, c ];
  c.children = [ d, e, f ];
  e.children = [ g ];
  f.children = [ g ];

  // give some accessible content
  a.tagName = 'div';
  b.tagName = 'button';
  e.tagName = 'div';
  g.tagName = 'button';

  // scenery should produce this accessible DOM tree
  // <div id="a">
  //   <button id="b">
  //   <div id="e">
  //      <button id="g1">
  //   <button id="g2">

  // get the accessible primary siblings - looking into pdomInstances for testing, there is no getter for primarySibling
  const divA = a.pdomInstances[ 0 ].peer!.primarySibling!;
  const buttonB = b.pdomInstances[ 0 ].peer!.primarySibling!;
  const divE = e.pdomInstances[ 0 ].peer!.primarySibling!;
  const buttonG1 = g.pdomInstances[ 0 ].peer!.primarySibling!;
  const buttonG2 = g.pdomInstances[ 1 ].peer!.primarySibling!;

  const divAChildren = divA.childNodes;
  const divEChildren = divE.childNodes;

  assert.ok( _.includes( divAChildren, buttonB ), 'button B should be an immediate child of div A' );
  assert.ok( _.includes( divAChildren, divE ), 'div E should be an immediate child of div A' );
  assert.ok( _.includes( divAChildren, buttonG2 ), 'button G2 should be an immediate child of div A' );
  assert.ok( _.includes( divEChildren, buttonG1 ), 'button G1 should be an immediate child of div E' );

  // make node B invisible for accessibility - it should should visible, but hidden from screen readers
  b.pdomVisible = false;
  assert.equal( b.visible, true, 'b should be visible after becoming hidden for screen readers' );
  assert.equal( b.pdomVisible, false, 'b state should reflect it is hidden for screen readers' );
  assert.equal( buttonB.hidden, true, 'buttonB should be hidden for screen readers' );
  assert.equal( b.pdomDisplayed, false, 'pdomVisible=false, b should have no representation in the PDOM' );
  b.pdomVisible = true;

  // make node B invisible - it should not be visible, and it should be hidden for screen readers
  b.visible = false;
  assert.equal( b.visible, false, 'state of node b is visible' );
  assert.equal( buttonB.hidden, true, 'buttonB is hidden from screen readers after becoming invisible' );
  assert.equal( b.pdomVisible, true, 'state of node b still reflects pdom visibility when invisible' );
  assert.equal( b.pdomDisplayed, false, 'b invisible and should have no representation in the PDOM' );
  b.visible = true;

  // make node f invisible - g's trail that goes through f should be invisible to AT, tcomhe child of c should remain pdomVisible
  f.visible = false;
  assert.equal( g.isPDOMVisible(), true, 'state of pdomVisible should remain true on node g' );
  assert.ok( !buttonG1.hidden, 'buttonG1 (child of e) should not be hidden after parent node f made invisible (no accessible content on node f)' );
  assert.equal( buttonG2.hidden, true, 'buttonG2 should be hidden after parent node f made invisible (no accessible content on node f)' );
  assert.equal( g.pdomDisplayed, true, 'one parent still visible, g still has one PDOMInstance displayed in PDOM' );
  f.visible = true;

  // make node c (no accessible content) invisible to screen, e should be hidden and g2 should be hidden
  c.pdomVisible = false;
  assert.equal( c.visible, true, 'c should still be visible after becoming invisible to screen readers' );
  assert.equal( divE.hidden, true, 'div E should be hidden after parent node c (no accessible content) is made invisible to screen readers' );
  assert.equal( buttonG2.hidden, true, 'buttonG2 should be hidden after ancestor node c (no accessible content) is made invisible to screen readers' );
  assert.ok( !divA.hidden, 'div A should not have been hidden by making descendant c invisible to screen readers' );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );

} );

QUnit.test( 'inputValue', assert => {

  const rootNode = new Node();
  const display = new Display( rootNode );
  document.body.appendChild( display.domElement );

  const a = new Node( { tagName: 'input', inputType: 'radio', inputValue: 'i am value' } );
  rootNode.addChild( a );
  let aElement = getPrimarySiblingElementByNode( a );
  assert.ok( aElement.getAttribute( 'value' ) === 'i am value', 'should have correct value' );

  const differentValue = 'i am different value';
  a.inputValue = differentValue;
  aElement = getPrimarySiblingElementByNode( a );
  assert.ok( aElement.getAttribute( 'value' ) === differentValue, 'should have different value' );

  rootNode.addChild( new Node( { children: [ a ] } ) );
  aElement = a.pdomInstances[ 1 ].peer!.primarySibling!;
  assert.ok( aElement.getAttribute( 'value' ) === differentValue, 'should have the same different value' );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );

} );

QUnit.test( 'setPDOMAttribute', assert => {

  const rootNode = new Node();
  const display = new Display( rootNode );
  document.body.appendChild( display.domElement );

  const a = new Node( { tagName: 'div', labelContent: 'hello' } );
  rootNode.addChild( a );

  a.setPDOMAttribute( 'test', 'test1' );
  let aElement = getPrimarySiblingElementByNode( a );
  assert.ok( aElement.getAttribute( 'test' ) === 'test1', 'setPDOMAttribute for primary sibling' );

  a.removePDOMAttribute( 'test' );
  aElement = getPrimarySiblingElementByNode( a );
  assert.ok( aElement.getAttribute( 'test' ) === null, 'removePDOMAttribute for primary sibling' );

  a.setPDOMAttribute( 'test', 'testValue' );
  a.setPDOMAttribute( 'test', 'testValueLabel', {
    elementName: PDOMPeer.LABEL_SIBLING
  } );

  const testBothAttributes = () => {
    aElement = getPrimarySiblingElementByNode( a );
    const aLabelElement = aElement.parentElement!.children[ DEFAULT_LABEL_SIBLING_INDEX ];
    assert.ok( aElement.getAttribute( 'test' ) === 'testValue', 'setPDOMAttribute for primary sibling 2' );
    assert.ok( aLabelElement.getAttribute( 'test' ) === 'testValueLabel', 'setPDOMAttribute for label sibling' );
  };
  testBothAttributes();

  rootNode.removeChild( a );
  rootNode.addChild( new Node( { children: [ a ] } ) );
  testBothAttributes();

  a.removePDOMAttribute( 'test', {
    elementName: PDOMPeer.LABEL_SIBLING
  } );
  aElement = getPrimarySiblingElementByNode( a );
  const aLabelElement = aElement.parentElement!.children[ DEFAULT_LABEL_SIBLING_INDEX ];
  assert.ok( aElement.getAttribute( 'test' ) === 'testValue', 'removePDOMAttribute for label should not effect primary sibling ' );
  assert.ok( aLabelElement.getAttribute( 'test' ) === null, 'removePDOMAttribute for label sibling' );

  a.removePDOMAttributes();
  const attributeName = 'multiTest';
  a.setPDOMAttribute( attributeName, 'true', {
    type: 'attribute'
  } );
  aElement = getPrimarySiblingElementByNode( a );
  assert.ok( aElement.getAttribute( attributeName ) === 'true', 'type:attribute should set attribute' );

  a.setPDOMAttribute( attributeName, false, {
    type: 'property'
  } );
  assert.ok( !aElement.getAttribute( attributeName ), 'type:property should remove attribute' );

  // @ts-expect-error for testing
  assert.equal( aElement[ attributeName ], false, 'type:property should set property' );

  const testAttributes = a.getPDOMAttributes().filter( a => a.attribute === attributeName );
  assert.ok( testAttributes.length === 1, 'type change should alter the attribute, not add a new one.' );

  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );
} );

QUnit.test( 'pdomChecked', assert => {

  const rootNode = new Node();
  const display = new Display( rootNode );
  document.body.appendChild( display.domElement );

  const a = new Node( { tagName: 'input', inputType: 'radio', pdomChecked: true } );
  rootNode.addChild( a );
  let aElement = getPrimarySiblingElementByNode( a ) as HTMLInputElement;
  assert.ok( aElement.checked, 'should be checked' );

  a.pdomChecked = false;
  aElement = getPrimarySiblingElementByNode( a ) as HTMLInputElement;
  assert.ok( !aElement.checked, 'should not be checked' );

  a.inputType = 'range';
  window.assert && assert.throws( () => {
    a.pdomChecked = true;
  }, /.*/, 'should fail if inputType range' );

  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );

} );

QUnit.test( 'Aria Label Setter', assert => {

  const rootNode = new Node( { tagName: 'div' } );
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  // create some nodes for testing
  const a = new Node( { tagName: 'button', ariaLabel: TEST_LABEL_2 } );

  assert.ok( a.ariaLabel === TEST_LABEL_2, 'aria-label getter/setter' );
  assert.ok( a.labelContent === null, 'no other label set with aria-label' );
  assert.ok( a.innerContent === null, 'no inner content set with aria-label' );

  rootNode.addChild( a );
  let buttonA = a.pdomInstances[ 0 ].peer!.primarySibling!;
  assert.ok( buttonA.getAttribute( 'aria-label' ) === TEST_LABEL_2, 'setter on dom element' );
  assert.ok( buttonA.innerHTML === '', 'no inner html with aria-label setter' );

  a.ariaLabel = null;

  buttonA = a.pdomInstances[ 0 ].peer!.primarySibling!;
  assert.ok( !buttonA.hasAttribute( 'aria-label' ), 'setter can clear on dom element' );
  assert.ok( buttonA.innerHTML === '', 'no inner html with aria-label setter when clearing' );
  assert.ok( a.ariaLabel === null, 'cleared in Node model.' );

  pdomAuditRootNode( rootNode );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );

} );

QUnit.test( 'focusable option', assert => {
  if ( !canRunTests ) {
    assert.ok( true, 'Skipping test because document does not have focus' );
    return;
  }

  // test the behavior of focusable function
  const rootNode = new Node( { tagName: 'div' } );
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  display.initializeEvents();

  const a = new Node( { tagName: 'div', focusable: true } );
  rootNode.addChild( a );

  assert.equal( a.focusable, true, 'focusable option setter' );
  assert.ok( getPrimarySiblingElementByNode( a ).tabIndex === 0, 'tab index on primary sibling with setter' );

  // change the tag name, but focusable should stay the same
  a.tagName = 'p';

  assert.equal( a.focusable, true, 'tagName option should not change focusable value' );
  assert.ok( getPrimarySiblingElementByNode( a ).tabIndex === 0, 'tagName option should not change tab index on primary sibling' );

  a.focusable = false;
  assert.ok( getPrimarySiblingElementByNode( a ).tabIndex === -1, 'set focusable false' );

  const b = new Node( { tagName: 'p' } );
  rootNode.addChild( b );

  b.focusable = true;

  assert.ok( b.focusable, 'set focusable as setter' );
  assert.ok( getPrimarySiblingElementByNode( b ).tabIndex === 0, 'set focusable as setter' );

  // HTML elements that are natively focusable are focusable by default
  const c = new Node( { tagName: 'button' } );
  assert.ok( c.focusable, 'button is focusable by default' );

  // change tagName to something that is not focusable, focusable should be false
  c.tagName = 'p';
  assert.ok( !c.focusable, 'button changed to paragraph, should no longer be focusable' );

  // When focusable is set to null on an element that is not focusable by default, it should lose focus
  const d = new Node( { tagName: 'div', focusable: true, focusHighlight: focusHighlight } );
  rootNode.addChild( d );
  d.focus();
  assert.ok( d.focused, 'focusable div should be focused after calling focus()' );

  d.focusable = null;
  assert.ok( !d.focused, 'default div should lose focus after node restored to null focusable' );

  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );

} );

QUnit.test( 'append siblings/appendLabel/appendDescription setters', assert => {

  // test the behavior of focusable function
  const rootNode = new Node( { tagName: 'div' } );
  const display = new Display( rootNode );
  document.body.appendChild( display.domElement );

  const a = new Node( {
    tagName: 'li',
    innerContent: TEST_INNER_CONTENT,
    labelTagName: 'h3',
    labelContent: TEST_LABEL,
    descriptionContent: TEST_DESCRIPTION,
    containerTagName: 'section',
    appendLabel: true
  } );
  rootNode.addChild( a );

  const aElement = getPrimarySiblingElementByNode( a );
  let containerElement = aElement.parentElement!;
  assert.ok( containerElement.tagName.toUpperCase() === 'SECTION', 'container parent is set to right tag' );

  let peerElements = containerElement.childNodes as unknown as HTMLElement[];
  assert.ok( peerElements.length === 3, 'expected three siblings' );
  assert.ok( peerElements[ 0 ].tagName.toUpperCase() === DEFAULT_DESCRIPTION_TAG_NAME, 'description first sibling' );
  assert.ok( peerElements[ 1 ].tagName.toUpperCase() === 'LI', 'primary sibling second sibling' );
  assert.ok( peerElements[ 2 ].tagName.toUpperCase() === 'H3', 'label sibling last' );

  a.appendDescription = true;
  containerElement = getPrimarySiblingElementByNode( a ).parentElement!;
  peerElements = containerElement.childNodes as unknown as HTMLElement[];
  assert.ok( containerElement.childNodes.length === 3, 'expected three siblings' );
  assert.ok( peerElements[ 0 ].tagName.toUpperCase() === 'LI', 'primary sibling first sibling' );
  assert.ok( peerElements[ 1 ].tagName.toUpperCase() === 'H3', 'label sibling second' );
  assert.ok( peerElements[ 2 ].tagName.toUpperCase() === DEFAULT_DESCRIPTION_TAG_NAME, 'description last sibling' );

  // clear it out back to defaults should work with setters
  a.appendDescription = false;
  a.appendLabel = false;
  containerElement = getPrimarySiblingElementByNode( a ).parentElement!;
  peerElements = containerElement.childNodes as unknown as HTMLElement[];
  assert.ok( containerElement.childNodes.length === 3, 'expected three siblings' );
  assert.ok( peerElements[ 0 ].tagName.toUpperCase() === 'H3', 'label sibling first' );
  assert.ok( peerElements[ 1 ].tagName.toUpperCase() === DEFAULT_DESCRIPTION_TAG_NAME, 'description sibling second' );
  assert.ok( peerElements[ 2 ].tagName.toUpperCase() === 'LI', 'primary sibling last' );

  // test order when using appendLabel/appendDescription without a parent container - order should be primary sibling,
  // label sibling, description sibling
  const b = new Node( {
    tagName: 'input',
    inputType: 'checkbox',
    labelTagName: 'label',
    labelContent: TEST_LABEL,
    descriptionContent: TEST_DESCRIPTION,
    appendLabel: true,
    appendDescription: true
  } );
  rootNode.addChild( b );

  let bPeer = getPDOMPeerByNode( b );
  let bElement = getPrimarySiblingElementByNode( b );
  let bElementParent = bElement.parentElement!;
  let indexOfPrimaryElement = Array.prototype.indexOf.call( bElementParent.childNodes, bElement );

  assert.ok( bElementParent.childNodes[ indexOfPrimaryElement ] === bElement, 'b primary sibling first with no container, both appended' );
  assert.ok( bElementParent.childNodes[ indexOfPrimaryElement + 1 ] === bPeer.labelSibling, 'b label sibling second with no container, both appended' );
  assert.ok( bElementParent.childNodes[ indexOfPrimaryElement + 2 ] === bPeer.descriptionSibling, 'b description sibling third with no container, both appended' );

  // test order when only description appended and no parent container - order should be label, primary, then
  // description
  b.appendLabel = false;

  // refresh since operation may have created new Objects
  bPeer = getPDOMPeerByNode( b );
  bElement = getPrimarySiblingElementByNode( b );
  bElementParent = bElement.parentElement!;
  indexOfPrimaryElement = Array.prototype.indexOf.call( bElementParent.childNodes, bElement );

  assert.ok( bElementParent.childNodes[ indexOfPrimaryElement - 1 ] === bPeer.labelSibling, 'b label sibling first with no container, description appended' );
  assert.ok( bElementParent.childNodes[ indexOfPrimaryElement ] === bElement, 'b primary sibling second with no container, description appended' );
  assert.ok( bElementParent.childNodes[ indexOfPrimaryElement + 1 ] === bPeer.descriptionSibling, 'b description sibling third with no container, description appended' );

  pdomAuditRootNode( rootNode );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );

} );

QUnit.test( 'containerAriaRole option', assert => {

  // test the behavior of focusable function
  const rootNode = new Node( { tagName: 'div' } );
  const display = new Display( rootNode );
  document.body.appendChild( display.domElement );

  const a = new Node( {
    tagName: 'div',
    containerTagName: 'div',
    containerAriaRole: 'application'
  } );

  rootNode.addChild( a );
  assert.ok( a.containerAriaRole === 'application', 'role attribute should be on node property' );
  let aElement = getPrimarySiblingElementByNode( a );
  assert.ok( aElement.parentElement!.getAttribute( 'role' ) === 'application', 'role attribute should be on parent element' );

  a.containerAriaRole = null;
  assert.ok( a.containerAriaRole === null, 'role attribute should be cleared on node' );
  aElement = getPrimarySiblingElementByNode( a );
  assert.ok( aElement.parentElement!.getAttribute( 'role' ) === null, 'role attribute should be cleared on parent element' );

  pdomAuditRootNode( rootNode );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );

} );

QUnit.test( 'ariaRole option', assert => {

  // test the behavior of focusable function
  const rootNode = new Node( { tagName: 'div' } );
  const display = new Display( rootNode );
  document.body.appendChild( display.domElement );

  const a = new Node( {
    tagName: 'div',
    innerContent: 'Draggable',
    ariaRole: 'application'
  } );

  rootNode.addChild( a );
  assert.ok( a.ariaRole === 'application', 'role attribute should be on node property' );
  let aElement = getPrimarySiblingElementByNode( a );
  assert.ok( aElement.getAttribute( 'role' ) === 'application', 'role attribute should be on element' );

  a.ariaRole = null;
  assert.ok( a.ariaRole === null, 'role attribute should be cleared on node' );
  aElement = getPrimarySiblingElementByNode( a );
  assert.ok( aElement.getAttribute( 'role' ) === null, 'role attribute should be cleared on element' );

  pdomAuditRootNode( rootNode );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );

} );


// Higher level setter/getter options
QUnit.test( 'accessibleName option', assert => {

  assert.ok( true );

  // test the behavior of focusable function
  const rootNode = new Node( { tagName: 'div' } );
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  const a = new Node( { tagName: 'div', accessibleName: TEST_LABEL } );
  rootNode.addChild( a );

  assert.ok( a.accessibleName === TEST_LABEL, 'accessibleName getter' );

  const aElement = getPrimarySiblingElementByNode( a );
  assert.ok( aElement.textContent === TEST_LABEL, 'accessibleName setter on div' );

  const b = new Node( { tagName: 'input', accessibleName: TEST_LABEL, inputType: 'range' } );
  a.addChild( b );
  const bElement = getPrimarySiblingElementByNode( b );
  const bParent = getPrimarySiblingElementByNode( b ).parentElement!;
  const bLabelSibling = bParent.children[ DEFAULT_LABEL_SIBLING_INDEX ];
  assert.ok( bLabelSibling.textContent === TEST_LABEL, 'accessibleName sets label sibling' );
  assert.ok( bLabelSibling.getAttribute( 'for' )!.includes( bElement.id ), 'accessibleName sets label\'s "for" attribute' );

  const c = new Node( { containerTagName: 'div', tagName: 'div', ariaLabel: 'overrideThis' } );
  rootNode.addChild( c );
  const cAccessibleNameBehavior: AccessibleNameBehaviorFunction = ( node, options, accessibleName ) => {
    options.ariaLabel = accessibleName;
    return options;
  };
  c.accessibleNameBehavior = cAccessibleNameBehavior;

  assert.ok( c.accessibleNameBehavior === cAccessibleNameBehavior, 'getter works' );

  let cLabelElement = getPrimarySiblingElementByNode( c ).parentElement!.children[ DEFAULT_LABEL_SIBLING_INDEX ];
  assert.ok( cLabelElement.getAttribute( 'aria-label' ) === 'overrideThis', 'accessibleNameBehavior should not work until there is accessible name' );
  c.accessibleName = 'accessible name description';
  cLabelElement = getPrimarySiblingElementByNode( c ).parentElement!.children[ DEFAULT_LABEL_SIBLING_INDEX ];
  assert.ok( cLabelElement.getAttribute( 'aria-label' ) === 'accessible name description', 'accessible name setter' );

  c.accessibleName = '';

  cLabelElement = getPrimarySiblingElementByNode( c ).parentElement!.children[ DEFAULT_LABEL_SIBLING_INDEX ];
  assert.ok( cLabelElement.getAttribute( 'aria-label' ) === '', 'accessibleNameBehavior should work for empty string' );

  c.accessibleName = null;
  cLabelElement = getPrimarySiblingElementByNode( c ).parentElement!.children[ DEFAULT_LABEL_SIBLING_INDEX ];
  assert.ok( cLabelElement.getAttribute( 'aria-label' ) === 'overrideThis', 'accessibleNameBehavior should not work until there is accessible name' );


  const d = new Node( { containerTagName: 'div', tagName: 'div' } );
  rootNode.addChild( d );
  const dAccessibleNameBehavior: AccessibleNameBehaviorFunction = ( node, options, accessibleName ) => {

    options.ariaLabel = accessibleName;
    return options;
  };
  d.accessibleNameBehavior = dAccessibleNameBehavior;

  assert.ok( d.accessibleNameBehavior === dAccessibleNameBehavior, 'getter works' );
  let dLabelElement = getPrimarySiblingElementByNode( d ).parentElement!.children[ DEFAULT_LABEL_SIBLING_INDEX ];
  assert.ok( dLabelElement.getAttribute( 'aria-label' ) === null, 'accessibleNameBehavior should not work until there is accessible name' );
  const accessibleNameDescription = 'accessible name description';
  d.accessibleName = accessibleNameDescription;
  dLabelElement = getPrimarySiblingElementByNode( d ).parentElement!.children[ DEFAULT_LABEL_SIBLING_INDEX ];
  assert.ok( dLabelElement.getAttribute( 'aria-label' ) === accessibleNameDescription, 'accessible name setter' );

  d.accessibleName = '';

  dLabelElement = getPrimarySiblingElementByNode( d ).parentElement!.children[ DEFAULT_LABEL_SIBLING_INDEX ];
  assert.ok( dLabelElement.getAttribute( 'aria-label' ) === '', 'accessibleNameBehavior should work for empty string' );

  d.accessibleName = null;
  dLabelElement = getPrimarySiblingElementByNode( d ).parentElement!.children[ DEFAULT_LABEL_SIBLING_INDEX ];
  assert.ok( dLabelElement.getAttribute( 'aria-label' ) === null, 'accessibleNameBehavior should not work until there is accessible name' );

  //-------------------------------------------------------------------------------
  // An accessibleNameBehavior that triggers PDOMTree.createTree is not supported
  // and should throw an error
  //-------------------------------------------------------------------------------
  const e = new Node( { tagName: 'div' } );
  const f = new Node( { tagName: 'div' } );
  rootNode.addChild( e );
  e.addChild( f );

  f.accessibleNameBehavior = ( node, options, accessibleName, otherNodeCallbacks ) => {
    otherNodeCallbacks.push( () => {
      e.accessibleName = accessibleName;
    } );
    return options;
  };

  const testAccessibleName = 'This is a test';

  window.assert && assert.throws( () => {
    f.accessibleName = testAccessibleName;
  }, 'Triggering re-render on parent is not supported because it would cause recursion and should throw.' );
  rootNode.removeChild( e );

  //-------------------------------------------------------------------------------
  // An accessibleNameBehavior that triggers PDOMTree.createTree is not supported
  // and should throw an error
  //-------------------------------------------------------------------------------
  const g = new Node( { tagName: 'div' } );
  rootNode.addChild( g );
  g.accessibleNameBehavior = ( node, options, accessibleName, otherNodeCallbacks ) => {

    // Change the content so that this is not a no-op
    node.accessibleName = accessibleName + 'blarg';
    return options;
  };

  window.assert && assert.throws( () => {
    g.accessibleName = testAccessibleName;
  }, 'Doing something that would trigger a full PDOMTree.update recursively should throw.' );
  rootNode.removeChild( g );

  //-------------------------------------------------------------------------------
  // An accessibleNameBehavior that triggers PDOMTree.createTree is not supported
  // and should throw an error
  //-------------------------------------------------------------------------------
  const h = new Node( { tagName: 'div' } );
  rootNode.addChild( h );
  h.accessibleNameBehavior = ( node, options, accessibleName, otherNodeCallbacks ) => {

    // Clear the content to remove the Node form the PDOM
    node.tagName = null;
    return options;
  };

  window.assert && assert.throws( () => {
    h.accessibleName = testAccessibleName;
  }, 'Clearing an accessibleName while updating will trigger recursive updates and is not supported.' );
  rootNode.removeChild( h );

  pdomAuditRootNode( rootNode );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );
} );

QUnit.test( 'accessibleHelpText option', assert => {

  assert.ok( true );

  // test the behavior of focusable function
  const rootNode = new Node( { tagName: 'div' } );
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  // label tag needed for default sibling indices to work
  const a = new Node( {
    containerTagName: 'div',
    tagName: 'div',
    labelTagName: 'div',
    accessibleHelpText: TEST_DESCRIPTION
  } );
  rootNode.addChild( a );

  rootNode.addChild( new Node( { tagName: 'input', inputType: 'range' } ) );
  assert.ok( a.accessibleHelpText === TEST_DESCRIPTION, 'accessibleHelpText getter' );

  // default for help text is to append description after the primary sibling
  const aDescriptionElement = getPrimarySiblingElementByNode( a ).parentElement!.children[ APPENDED_DESCRIPTION_SIBLING_INDEX ];
  assert.ok( aDescriptionElement.textContent === TEST_DESCRIPTION, 'accessibleHelpText setter on div' );

  const b = new Node( {
    containerTagName: 'div',
    tagName: 'button',
    descriptionContent: 'overrideThis',
    labelTagName: 'div'
  } );
  rootNode.addChild( b );

  b.accessibleHelpTextBehavior = ( node, options, accessibleHelpText ) => {

    options.descriptionTagName = 'p';
    options.descriptionContent = accessibleHelpText;
    return options;
  };

  let bDescriptionElement = getPrimarySiblingElementByNode( b ).parentElement!.children[ DEFAULT_DESCRIPTION_SIBLING_INDEX ];
  assert.ok( bDescriptionElement.textContent === 'overrideThis', 'initial textContent from descriptionContent' );
  b.accessibleHelpText = 'help text description';
  bDescriptionElement = getPrimarySiblingElementByNode( b ).parentElement!.children[ DEFAULT_DESCRIPTION_SIBLING_INDEX ];
  assert.ok( bDescriptionElement.textContent === 'help text description', 'help text setter' );

  b.accessibleHelpText = '';

  bDescriptionElement = getPrimarySiblingElementByNode( b ).parentElement!.children[ DEFAULT_DESCRIPTION_SIBLING_INDEX ];
  assert.ok( bDescriptionElement.textContent === '', 'accessibleHelpTextBehavior should work for empty string' );

  b.accessibleHelpText = 'some content';
  bDescriptionElement = getPrimarySiblingElementByNode( b ).parentElement!.children[ DEFAULT_DESCRIPTION_SIBLING_INDEX ];
  assert.ok( bDescriptionElement.textContent === 'some content', 'some content set after empty string' );

  // Make sure that setting the accessibleHelpText to null results in restoring the description content.
  b.accessibleHelpText = null;
  bDescriptionElement = getPrimarySiblingElementByNode( b ).parentElement!.children[ DEFAULT_DESCRIPTION_SIBLING_INDEX ];
  assert.ok( bDescriptionElement.textContent === 'overrideThis', 'fall back to descriptionContent when accessibleHelpText is null' );

  pdomAuditRootNode( rootNode );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );
} );

QUnit.test( 'move to front/move to back', assert => {
  if ( !canRunTests ) {
    assert.ok( true, 'Skipping test because document does not have focus' );
    return;
  }

  // make sure state is restored after moving children to front and back
  const rootNode = new Node( { tagName: 'div' } );
  const display = new Display( rootNode );
  document.body.appendChild( display.domElement );

  display.initializeEvents();

  const a = new Node( { tagName: 'button', focusHighlight: TEST_HIGHLIGHT } );
  const b = new Node( { tagName: 'button', focusHighlight: TEST_HIGHLIGHT } );
  rootNode.children = [ a, b ];
  b.focus();

  // after moving a to front, b should still have focus
  a.moveToFront();
  assert.ok( b.focused, 'b should have focus after a moved to front' );

  // after moving a to back, b should still have focus
  a.moveToBack();

  // add a guard where we don't check this if focus has been moved somewhere else. This happens sometimes with
  // dev tools or other windows opened, see https://github.com/phetsims/scenery/issues/827
  if ( document.body.contains( document.activeElement ) && document.body !== document.activeElement ) {
    assert.ok( b.focused, 'b should have focus after a moved to back' );
  }

  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );

} );

QUnit.test( 'Node.enabledProperty with PDOM', assert => {

  const rootNode = new Node( { tagName: 'div' } );
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  const pdomNode = new Node( {
    tagName: 'p'
  } );

  rootNode.addChild( pdomNode );
  assert.ok( pdomNode.pdomInstances.length === 1, 'should have an instance when attached to display' );
  assert.ok( !!pdomNode.pdomInstances[ 0 ].peer, 'should have a peer' );

  assert.ok( pdomNode.pdomInstances[ 0 ].peer!.primarySibling!.getAttribute( 'aria-disabled' ) !== 'true', 'should be enabled to start' );
  pdomNode.enabled = false;
  assert.ok( pdomNode.pdomInstances[ 0 ].peer!.primarySibling!.getAttribute( 'aria-disabled' ) === 'true', 'should be aria-disabled when disabled' );
  pdomNode.enabled = true;
  assert.ok( pdomNode.pdomInstances[ 0 ].peer!.primarySibling!.getAttribute( 'aria-disabled' ) === 'false', 'Actually set to false since it was previously disabled.' );
  pdomNode.dispose();
  assert.ok( pdomNode.pdomInstances.length === 0, 'disposed nodes do not have a PDOM instance' );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );
} );

// these fuzzers take time, so it is nice when they are last
QUnit.test( 'Display.interactive toggling in the PDOM', assert => {

  const rootNode = new Node( { tagName: 'div' } );
  var display = new Display( rootNode ); // eslint-disable-line no-var
  display.initializeEvents();
  document.body.appendChild( display.domElement );

  const pdomRangeChild = new Node( { tagName: 'input', inputType: 'range' } );
  const pdomParagraphChild = new Node( { tagName: 'p' } );
  const pdomButtonChild = new Node( { tagName: 'button' } );

  const pdomParent = new Node( {
    tagName: 'button',
    children: [ pdomRangeChild, pdomParagraphChild, pdomButtonChild ]
  } );

  const DISABLED_TRUE = true;

  // For of list of html elements that support disabled, see https://developer.mozilla.org/en-US/docs/Web/HTML/Attributes/disabled
  const DEFAULT_DISABLED_WHEN_SUPPORTED = false;
  const DEFAULT_DISABLED_WHEN_NOT_SUPPORTED = undefined;

  rootNode.addChild( pdomParent );

  assert.ok( true, 'initial case' );

  const testDisabled = ( node: Node, disabled: boolean | undefined, message: string, pdomInstance = 0 ): void => {

    // @ts-expect-error "disabled" is only supported by certain attributes, see https://developer.mozilla.org/en-US/docs/Web/HTML/Attributes/disabled
    assert.ok( node.pdomInstances[ pdomInstance ].peer!.primarySibling!.disabled === disabled, message );
  };

  testDisabled( pdomParent, DEFAULT_DISABLED_WHEN_SUPPORTED, 'pdomParent initial no disabled' );
  testDisabled( pdomRangeChild, DEFAULT_DISABLED_WHEN_SUPPORTED, 'pdomRangeChild initial no disabled' );
  testDisabled( pdomParagraphChild, DEFAULT_DISABLED_WHEN_NOT_SUPPORTED, 'pdomParagraphChild initial no disabled' );
  testDisabled( pdomButtonChild, DEFAULT_DISABLED_WHEN_SUPPORTED, 'pdomButtonChild initial no disabled' );

  display.interactive = false;

  testDisabled( pdomParent, DISABLED_TRUE, 'pdomParent toggled not interactive' );
  testDisabled( pdomRangeChild, DISABLED_TRUE, 'pdomRangeChild toggled not interactive' );
  testDisabled( pdomParagraphChild, DISABLED_TRUE, 'pdomParagraphChild toggled not interactive' );
  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild toggled not interactive' );

  display.interactive = true;

  testDisabled( pdomParent, DEFAULT_DISABLED_WHEN_SUPPORTED, 'pdomParent toggled back to interactive' );
  testDisabled( pdomRangeChild, DEFAULT_DISABLED_WHEN_SUPPORTED, 'pdomRangeChild toggled back to interactive' );
  testDisabled( pdomParagraphChild, DEFAULT_DISABLED_WHEN_NOT_SUPPORTED, 'pdomParagraphChild toggled back to interactive' );
  testDisabled( pdomButtonChild, DEFAULT_DISABLED_WHEN_SUPPORTED, 'pdomButtonChild toggled back to interactive' );

  display.interactive = false;

  testDisabled( pdomParent, DISABLED_TRUE, 'pdomParent second toggled not interactive' );
  testDisabled( pdomRangeChild, DISABLED_TRUE, 'pdomRangeChild second toggled not interactive' );
  testDisabled( pdomParagraphChild, DISABLED_TRUE, 'pdomParagraphChild second toggled not interactive' );
  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild second toggled not interactive' );

  pdomParent.setPDOMAttribute( 'disabled', true, { type: 'property' } );
  pdomRangeChild.setPDOMAttribute( 'disabled', true, { type: 'property' } );
  pdomParagraphChild.setPDOMAttribute( 'disabled', true, { type: 'property' } );
  pdomButtonChild.setPDOMAttribute( 'disabled', true, { type: 'property' } );

  testDisabled( pdomParent, DISABLED_TRUE, 'pdomParent not interactive after setting disabled manually as property, display not interactive' );
  testDisabled( pdomRangeChild, DISABLED_TRUE, 'pdomRangeChild not interactive after setting disabled manually as property, display not interactive' );
  testDisabled( pdomParagraphChild, DISABLED_TRUE, 'pdomParagraphChild not interactive after setting disabled manually as property, display not interactive' );
  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild not interactive after setting disabled manually as property, display not interactive' );

  display.interactive = true;

  testDisabled( pdomParent, DISABLED_TRUE, 'pdomParent not interactive after setting disabled manually as property display interactive' );
  testDisabled( pdomRangeChild, DISABLED_TRUE, 'pdomRangeChild not interactive after setting disabled manually as property display interactive' );
  testDisabled( pdomParagraphChild, DISABLED_TRUE, 'pdomParagraphChild not interactive after setting disabled manually as property display interactive' );
  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild not interactive after setting disabled manually as property display interactive' );

  display.interactive = false;

  testDisabled( pdomParent, DISABLED_TRUE, 'pdomParent still disabled when display is not interactive again.' );
  testDisabled( pdomRangeChild, DISABLED_TRUE, 'pdomRangeChild still disabled when display is not interactive again.' );
  testDisabled( pdomParagraphChild, DISABLED_TRUE, 'pdomParagraphChild still disabled when display is not interactive again.' );
  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild still disabled when display is not interactive again.' );

  pdomParent.removePDOMAttribute( 'disabled' );
  pdomRangeChild.removePDOMAttribute( 'disabled' );
  pdomParagraphChild.removePDOMAttribute( 'disabled' );
  pdomButtonChild.removePDOMAttribute( 'disabled' );

  testDisabled( pdomParent, DISABLED_TRUE, 'pdomParent still disabled from display not interactive after local property removed.' );
  testDisabled( pdomRangeChild, DISABLED_TRUE, 'pdomRangeChild still disabled from display not interactive after local property removed.' );
  testDisabled( pdomParagraphChild, DISABLED_TRUE, 'pdomParagraphChild still disabled from display not interactive after local property removed.' );
  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild still disabled from display not interactive after local property removed.' );

  display.interactive = true;

  testDisabled( pdomParent, DEFAULT_DISABLED_WHEN_SUPPORTED, 'pdomParent interactive now without local property.' );
  testDisabled( pdomRangeChild, DEFAULT_DISABLED_WHEN_SUPPORTED, 'pdomRangeChild interactive now without local property.' );
  testDisabled( pdomParagraphChild, DEFAULT_DISABLED_WHEN_SUPPORTED, 'pdomParagraphChild interactive now without local property.' );
  testDisabled( pdomButtonChild, DEFAULT_DISABLED_WHEN_SUPPORTED, 'pdomButtonChild interactive now without local property.' );

  pdomParent.setPDOMAttribute( 'disabled', '' );
  pdomRangeChild.setPDOMAttribute( 'disabled', '' );
  pdomParagraphChild.setPDOMAttribute( 'disabled', '' );
  pdomButtonChild.setPDOMAttribute( 'disabled', '' );

  testDisabled( pdomParent, DISABLED_TRUE, 'pdomParent not interactive after setting disabled manually as attribute, display not interactive' );
  testDisabled( pdomRangeChild, DISABLED_TRUE, 'pdomRangeChild not interactive after setting disabled manually as attribute, display not interactive' );
  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild not interactive after setting disabled manually as attribute, display not interactive' );

  display.interactive = true;

  testDisabled( pdomParent, DISABLED_TRUE, 'pdomParent not interactive after setting disabled manually as attribute display interactive' );
  testDisabled( pdomRangeChild, DISABLED_TRUE, 'pdomRangeChild not interactive after setting disabled manually as attribute display interactive' );
  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild not interactive after setting disabled manually as attribute display interactive' );

  // This test doesn't work, because paragraphs don't support disabled, so the attribute "disabled" won't
  // automatically transfer over to the property value like for the others. For a list of Elements that support "disabled", see https://developer.mozilla.org/en-US/docs/Web/HTML/Attributes/disabled
  // testDisabled( pdomParagraphChild, DISABLED_TRUE, 'pdomParagraphChild not interactive after setting disabled manually as attribute, display  interactive' );

  display.interactive = false;

  testDisabled( pdomParent, DISABLED_TRUE, 'pdomParent still disabled when display is not interactive again.' );
  testDisabled( pdomRangeChild, DISABLED_TRUE, 'pdomRangeChild still disabled when display is not interactive again.' );
  testDisabled( pdomParagraphChild, DISABLED_TRUE, 'pdomParagraphChild still disabled when display is not interactive again.' );
  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild still disabled when display is not interactive again.' );

  pdomParent.removePDOMAttribute( 'disabled' );
  pdomRangeChild.removePDOMAttribute( 'disabled' );
  pdomParagraphChild.removePDOMAttribute( 'disabled' );
  pdomButtonChild.removePDOMAttribute( 'disabled' );

  testDisabled( pdomParent, DISABLED_TRUE, 'pdomParent still disabled from display not interactive after local attribute removed.' );
  testDisabled( pdomRangeChild, DISABLED_TRUE, 'pdomRangeChild still disabled from display not interactive after local attribute removed.' );
  testDisabled( pdomParagraphChild, DISABLED_TRUE, 'pdomParagraphChild still disabled from display not interactive after local attribute removed.' );
  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild still disabled from display not interactive after local attribute removed.' );

  display.interactive = true;

  testDisabled( pdomParent, DEFAULT_DISABLED_WHEN_SUPPORTED, 'pdomParent interactive now without local attribute.' );
  testDisabled( pdomRangeChild, DEFAULT_DISABLED_WHEN_SUPPORTED, 'pdomRangeChild interactive now without local attribute.' );
  testDisabled( pdomParagraphChild, DEFAULT_DISABLED_WHEN_SUPPORTED, 'pdomParagraphChild interactive now without local attribute.' );
  testDisabled( pdomButtonChild, DEFAULT_DISABLED_WHEN_SUPPORTED, 'pdomButtonChild interactive now without local attribute.' );

  const containerOfDAGButton = new Node( {
    children: [ pdomButtonChild ]
  } );
  pdomParent.addChild( containerOfDAGButton );

  testDisabled( pdomButtonChild, DEFAULT_DISABLED_WHEN_SUPPORTED, 'pdomButtonChild default not disabled.' );
  testDisabled( pdomButtonChild, DEFAULT_DISABLED_WHEN_SUPPORTED, 'pdomButtonChild default not disabled with dag.', 1 );

  display.interactive = false;

  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild turned disabled.' );
  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild turned disabled with dag.', 1 );

  pdomButtonChild.setPDOMAttribute( 'disabled', true, { type: 'property' } );

  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild turned disabled set property too.' );
  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild turned disabled set property too, with dag.', 1 );

  display.interactive = true;

  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild turned not disabled set property too.' );
  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild turned not disabled set property too, with dag.', 1 );

  display.interactive = false;

  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild turned disabled again.' );
  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild turned disabled again, with dag.', 1 );

  pdomButtonChild.removePDOMAttribute( 'disabled' );

  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild remove disabled while not interactive.' );
  testDisabled( pdomButtonChild, DISABLED_TRUE, 'pdomButtonChild remove disabled while not interactive, with dag.', 1 );

  display.interactive = true;

  testDisabled( pdomButtonChild, DEFAULT_DISABLED_WHEN_SUPPORTED, 'pdomButtonChild default not disabled after interactive again.' );
  testDisabled( pdomButtonChild, DEFAULT_DISABLED_WHEN_SUPPORTED, 'pdomButtonChild default not disabled after interactive again with dag.', 1 );

  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );
} );

QUnit.test( 'accessibleParagraph', assert => {

  const rootNode = new Node( { tagName: 'div' } );
  var display = new Display( rootNode ); // eslint-disable-line no-var
  display.initializeEvents();
  document.body.appendChild( display.domElement );

  const paragraphContent = 'Test Paragraph';

  //-------------------------------------------------------------------------
  // Basics
  //-------------------------------------------------------------------------
  const a = new Node();

  rootNode.addChild( a );

  a.accessibleParagraph = paragraphContent;
  assert.ok( a.accessibleParagraph === paragraphContent, 'getter and setter' );

  const aPeer = getPDOMPeerByNode( a );
  assert.ok( !!aPeer, 'A peer should be created from accessibleParagraph' );

  const aRenderedElement = aPeer.getPlaceableSibling();
  assert.ok( aRenderedElement.tagName === 'P', 'A paragraph should be created in the DOM' );

  assert.ok( a.tagName === null, 'There is no tagName on Node, but the element is still in the DOM' );

  //-------------------------------------------------------------------------
  // Integration with the lower level API and ordering
  //-------------------------------------------------------------------------
  const bLabelTagName = 'H3';
  const bTagName = 'BUTTON';
  const bDescriptionTagName = 'SPAN';
  const bParagraphTagName = 'P';
  const b = new Node( {
    tagName: 'button',

    labelTagName: 'h3',
    labelContent: 'Test Button',

    descriptionTagName: 'span',
    descriptionContent: 'Test Description',

    accessibleParagraph: paragraphContent
  } );

  rootNode.addChild( b );

  const bPeer = getPDOMPeerByNode( b );
  let bRenderedElements = bPeer.topLevelElements!;
  assert.ok( !!bRenderedElements, 'A peer should be created from tagName and accessibleParagraph' );

  assert.ok( bRenderedElements[ 0 ].tagName === bLabelTagName, 'b label element is in the DOM and before the paragraph' );
  assert.ok( bRenderedElements[ 1 ].tagName === bDescriptionTagName, 'b description element is in the DOM and after the focusable element' );
  assert.ok( bRenderedElements[ 2 ].tagName === bTagName, 'b primary element is in the DOM' );
  assert.ok( bRenderedElements[ 3 ].tagName === bParagraphTagName, 'b paragraph is in the DOM and after the primary element' );

  // Update order with appendLabel and appendDescription
  b.appendLabel = true;
  b.appendDescription = true;
  display.updateDisplay(); // required as the reorder happens in display update

  bRenderedElements = bPeer.topLevelElements!;
  assert.ok( bRenderedElements[ 0 ].tagName === bTagName, 'b label element is in the DOM and before the paragraph' );
  assert.ok( bRenderedElements[ 1 ].tagName === bLabelTagName, 'b description element is in the DOM and after the focusable element' );
  assert.ok( bRenderedElements[ 2 ].tagName === bDescriptionTagName, 'b primary element is in the DOM' );
  assert.ok( bRenderedElements[ 3 ].tagName === bParagraphTagName, 'b paragraph is in the DOM and after the primary element' );

  //-------------------------------------------------------------------------
  // Using behavior functions to forward to another Node
  //-------------------------------------------------------------------------
  const testParagraph = 'This is a test';

  const c = new Node();
  const d = new Node();
  const e = new Node();

  rootNode.addChild( c );
  c.addChild( d );
  c.addChild( e );

  // The behavior function cannot add a Node to the PDOM, so it must
  // have some accessibleContent first.
  d.accessibleParagraph = '';
  c.accessibleParagraphBehavior = ( node, options, accessibleParagraph, callbacksForOtherNodes ) => {
    callbacksForOtherNodes.push( () => {
      d.accessibleParagraph = accessibleParagraph;
    } );
    return options;
  };
  c.accessibleParagraph = testParagraph;

  assert.ok( d.accessibleParagraph === testParagraph );

  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );
} );

QUnit.test( 'accessibleHeading', assert => {

  const rootNode = new Node( { tagName: 'div' } );
  var display = new Display( rootNode ); // eslint-disable-line no-var
  display.initializeEvents();
  document.body.appendChild( display.domElement );

  const testHeadingContent = 'Heading';

  //--------------------------------------------------------------------------
  // Basic heading creation
  //--------------------------------------------------------------------------
  const t1 = new Node( { tagName: 'div', accessibleHeading: testHeadingContent } );
  rootNode.addChild( t1 );
  const t1Heading = getDOMElementByName( t1, PDOMPeer.HEADING_SIBLING );
  assert.ok( t1Heading.tagName === 'H1', 'Heading should be created in the DOM' );

  //--------------------------------------------------------------------------
  // Basic level computation
  //--------------------------------------------------------------------------
  const t2 = new Node( { tagName: 'div', accessibleHeading: testHeadingContent } );
  const t3 = new Node( { tagName: 'div', accessibleHeading: testHeadingContent } );
  const t4 = new Node( { tagName: 'div', accessibleHeading: testHeadingContent } );
  t1.addChild( t2 );
  t2.addChild( t3 );
  t3.addChild( t4 );

  const t2Heading = getDOMElementByName( t2, PDOMPeer.HEADING_SIBLING );
  const t3Heading = getDOMElementByName( t3, PDOMPeer.HEADING_SIBLING );
  const t4Heading = getDOMElementByName( t4, PDOMPeer.HEADING_SIBLING );
  assert.ok( t2Heading.tagName === 'H2', 'Next child should be an h2' );
  assert.ok( t3Heading.tagName === 'H3', 'Next child should be an h3' );
  assert.ok( t4Heading.tagName === 'H4', 'Next child should be an h4' );

  //--------------------------------------------------------------------------
  // Override with setAccessibleHeadingIncrement
  //--------------------------------------------------------------------------
  const t5 = new Node( { tagName: 'div', accessibleHeading: testHeadingContent, accessibleHeadingIncrement: 0 } );
  t4.addChild( t5 );
  const t5Heading = getDOMElementByName( t5, PDOMPeer.HEADING_SIBLING );
  assert.ok( t5Heading.tagName === 'H4', 'Child under h4 with accessibleHeadingIncrement 0 should result in h4' );

  //--------------------------------------------------------------------------
  // Change accessibleHeadingIncrement causes recompute
  //--------------------------------------------------------------------------
  t5.accessibleHeadingIncrement = 2;
  const t5Heading2 = getDOMElementByName( t5, PDOMPeer.HEADING_SIBLING );
  assert.ok( t5Heading2.tagName === 'H6', 'Child under h4 with accessibleHeadingIncrement 2 should result in h6' );

  t5.accessibleHeadingIncrement = 0;
  const t5Heading3 = getDOMElementByName( t5, PDOMPeer.HEADING_SIBLING );
  assert.ok( t5Heading3.tagName === 'H4', 'Child under h4 with accessibleHeadingIncrement 0 should result in h4' );

  //--------------------------------------------------------------------------
  // Change accessibleHeading updates content
  //--------------------------------------------------------------------------
  const updatedHeading = 'New Heading';
  t5.accessibleHeading = updatedHeading;
  const t5Heading4 = getDOMElementByName( t5, PDOMPeer.HEADING_SIBLING );
  assert.ok( t5Heading4.textContent === updatedHeading, 'Heading should have the correct content' );

  //--------------------------------------------------------------------------
  // Child under Node with accessibleHeadingIncrement respects parent level
  //--------------------------------------------------------------------------
  const t6 = new Node( { tagName: 'div', accessibleHeading: testHeadingContent } );
  t5.addChild( t6 );
  const t6Heading = getDOMElementByName( t6, PDOMPeer.HEADING_SIBLING );
  assert.ok( t6Heading.tagName === 'H5', 'Next child should be an h2' );

  //--------------------------------------------------------------------------
  // accessibleHeading works with associations
  //--------------------------------------------------------------------------
  const t7 = new Node( { tagName: 'div' } );
  t7.addAriaLabelledbyAssociation( { otherNode: t6, otherElementName: PDOMPeer.HEADING_SIBLING, thisElementName: PDOMPeer.PRIMARY_SIBLING } );
  t6.addChild( t7 );
  const t7Element = getDOMElementByName( t7, PDOMPeer.PRIMARY_SIBLING );
  const t6Element = getDOMElementByName( t6, PDOMPeer.HEADING_SIBLING );
  const t7LabelledBy = t7Element.getAttribute( 'aria-labelledby' );
  assert.ok( t7LabelledBy === t6Element.id, 'aria-labelledby should be set to the heading id' );

  //--------------------------------------------------------------------------
  // accessibleHeading respects pdomOrder
  //--------------------------------------------------------------------------
  const t8 = new Node( { tagName: 'div', accessibleHeading: testHeadingContent } );
  const t9 = new Node( { tagName: 'div', accessibleHeading: testHeadingContent } );
  const t10 = new Node( { tagName: 'div', accessibleHeading: testHeadingContent } );
  const t11 = new Node( { tagName: 'div', accessibleHeading: testHeadingContent } );

  // Test Nodes are siblings of eachother
  rootNode.addChild( t8 );
  rootNode.addChild( t9 );
  rootNode.addChild( t10 );
  rootNode.addChild( t11 );

  /// pdomOrder creates a hierarchy
  t8.pdomOrder = [ t9, t10 ];
  t10.pdomOrder = [ t11 ];

  const t8Heading = getDOMElementByName( t8, PDOMPeer.HEADING_SIBLING );
  const t9Heading = getDOMElementByName( t9, PDOMPeer.HEADING_SIBLING );
  const t10Heading = getDOMElementByName( t10, PDOMPeer.HEADING_SIBLING );
  const t11Heading = getDOMElementByName( t11, PDOMPeer.HEADING_SIBLING );

  assert.ok( t8Heading.tagName === 'H1', 't8 should be an h1' );
  assert.ok( t9Heading.tagName === 'H2', 't9 should be an h2' );
  assert.ok( t10Heading.tagName === 'H2', 't10 should be an h2' );
  assert.ok( t11Heading.tagName === 'H3', 't11 should be an h3' );

  //--------------------------------------------------------------------------
  // Changing pdomOrder causes recompute
  //--------------------------------------------------------------------------
  t8.pdomOrder = null;
  t10.pdomOrder = null;

  const t8Heading2 = getDOMElementByName( t8, PDOMPeer.HEADING_SIBLING );
  const t9Heading2 = getDOMElementByName( t9, PDOMPeer.HEADING_SIBLING );
  const t10Heading2 = getDOMElementByName( t10, PDOMPeer.HEADING_SIBLING );
  const t11Heading2 = getDOMElementByName( t11, PDOMPeer.HEADING_SIBLING );
  assert.ok( t8Heading2.tagName === 'H1', 't8 should be an h1' );
  assert.ok( t9Heading2.tagName === 'H1', 't9 should be an h1' );
  assert.ok( t10Heading2.tagName === 'H1', 't10 should be an h1' );
  assert.ok( t11Heading2.tagName === 'H1', 't11 should be an h2' );

  //--------------------------------------------------------------------------
  // accessibleHeading works with string Properties
  //--------------------------------------------------------------------------
  const valueA = 'testA';
  const valueB = 'testB';
  const headingStringProperty = new StringProperty( valueA );
  const t12 = new Node( { tagName: 'div', accessibleHeading: headingStringProperty } );
  rootNode.addChild( t12 );

  const t12Heading = getDOMElementByName( t12, PDOMPeer.HEADING_SIBLING );
  assert.ok( t12Heading.tagName === 'H1', 'Heading should be created in the DOM' );
  assert.ok( t12Heading.textContent === valueA, 'Heading should have the correct content' );

  // Change the value of the StringProperty
  headingStringProperty.value = valueB;
  const t12Heading2 = getDOMElementByName( t12, PDOMPeer.HEADING_SIBLING );
  assert.ok( t12Heading2.tagName === 'H1', 'Heading should be created in the DOM' );
  assert.ok( t12Heading2.textContent === valueB, 'Heading should have the correct content' );

  //--------------------------------------------------------------------------
  // Heading levels are constrained to 1-6
  // NOTE: Unfortunately these tests put scenery in a broken state for
  // further tests (because it can't recover from assertions). Commenting out
  // for now.
  //--------------------------------------------------------------------------
  // window.assert && assert.throws( () => {
  //   const t13 = new Node( { tagName: 'div', accessibleHeading: testHeadingContent, accessibleHeadingIncrement: -1 } );
  //   rootNode.addChild( t13 );
  // }, 'Setting an accessibleHeadingIncrement that is less than 1 should throw' );
  //
  // window.assert && assert.throws( () => {
  //   const t14 = new Node( { tagName: 'div', accessibleHeading: testHeadingContent, accessibleHeadingIncrement: 7 } );
  //   rootNode.addChild( t14 );
  // }, 'Setting an accessibleHeadingIncrement that is greater than 6 should throw' );

  //--------------------------------------------------------------------------
  // A more complicated subtree
  //--------------------------------------------------------------------------
  const t15 = new Node();
  const t16 = new Node( { accessibleHeading: headingStringProperty, tagName: 'div' } );
  const t17 = new Node();
  const t18 = new Node( { accessibleHeading: testHeadingContent, tagName: 'div', accessibleHeadingIncrement: 2 } );
  const t19 = new Node( { accessibleHeading: testHeadingContent, tagName: 'div', accessibleHeadingIncrement: 2 } );

  // t15
  //   ├── t16 (H1)
  //   └── t17
  //        └── t18 (H2)
  //        └── t19 (H2)
  rootNode.addChild( t15 );
  t15.addChild( t16 );
  t15.addChild( t17 );
  t17.addChild( t18 );
  t17.addChild( t19 );

  const t16Heading = getDOMElementByName( t16, PDOMPeer.HEADING_SIBLING );
  const t18Heading = getDOMElementByName( t18, PDOMPeer.HEADING_SIBLING );
  const t19Heading = getDOMElementByName( t19, PDOMPeer.HEADING_SIBLING );
  assert.ok( t16Heading.tagName === 'H1', 't16 should be an h1' );

  assert.ok( t18Heading.tagName === 'H2', 't18 should be an h2' );
  assert.ok( t19Heading.tagName === 'H2', 't19 should be an h2' );

  //--------------------------------------------------------------------------
  // accessibleHeading without tagName
  //--------------------------------------------------------------------------
  // basic default tag name
  const t20 = new Node( { accessibleHeading: testHeadingContent } );
  rootNode.addChild( t20 );

  const t20Heading = getDOMElementByName( t20, PDOMPeer.HEADING_SIBLING );
  assert.ok( t20Heading.tagName === 'H1', 'Heading should be created in the DOM' );
  assert.ok( t20Heading.parentElement, 'The heading should be placed in the DOM' );

  // Clearing the heading should fully remove the content from the DOM because there is no set tagName
  t20.accessibleHeading = null;
  assert.ok( t20.pdomInstances.length === 0, 'Clearing accessibleHeading removes all instances' );

  const t21 = new Node( { accessibleParagraph: 'Test paragraph', accessibleHeading: testHeadingContent } );
  const t22 = new Node( { tagName: 'button' } );

  rootNode.addChild( t21 );
  t21.addChild( t22 );

  const t21Heading = getDOMElementByName( t21, PDOMPeer.HEADING_SIBLING );
  const t21Paragraph = getDOMElementByName( t21, PDOMPeer.PARAGRAPH_SIBLING );
  const t21Primary = getDOMElementByName( t21, PDOMPeer.PRIMARY_SIBLING );
  const t22Primary = getDOMElementByName( t22, PDOMPeer.PRIMARY_SIBLING );

  assert.ok( t21Heading.tagName === 'H1', 'Heading should be created in the DOM' );
  assert.ok( t21Paragraph.tagName === 'P', 'Paragraph should be created in the DOM' );
  assert.ok( t21Primary.tagName === 'DIV', 'Primary should be created in the DOM by default.' );
  assert.ok( t22Primary.tagName === 'BUTTON', 'Button should be created in the DOM' );
  assert.ok( t22Primary.parentElement === t21Primary, 'The t22 element is added under the default primary element of t21' );

  t21.accessibleHeading = null;
  t21.accessibleParagraph = null;
  assert.ok( t21.pdomInstances.length === 0, 'Clearing accessibleHeading and accessibleParagraph removes all instances' );

  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );
} );

QUnit.test( 'accessibleRoleDescription', assert => {


  const rootNode = new Node( { tagName: 'div' } );
  const display = new Display( rootNode );
  display.initializeEvents();
  document.body.appendChild( display.domElement );

  const testRoleA = 'Movable';
  const testRoleB = 'Thing';
  const testRoleProperty = new StringProperty( 'Draggable' );

  //-------------------------------------------------------------------------
  // Basic getter/setter
  //-------------------------------------------------------------------------
  const a = new Node( { tagName: 'div', accessibleRoleDescription: testRoleA } );
  rootNode.addChild( a );
  assert.ok( a.accessibleRoleDescription === testRoleA, 'Option for accessibleRoleDescription works.' );
  const elementRoleA = getPrimarySiblingElementByNode( a ).getAttribute( 'aria-roledescription' );
  assert.ok( elementRoleA === testRoleA, 'Element should have the aria-roledescription attribute set' );

  a.accessibleRoleDescription = testRoleB;
  assert.ok( a.accessibleRoleDescription === testRoleB, 'Setter for accessibleRoleDescription works.' );
  const elementRoleB = getPrimarySiblingElementByNode( a ).getAttribute( 'aria-roledescription' );
  assert.ok( elementRoleB === testRoleB, 'Element should have the aria-roledescription from the setter' );

  //-------------------------------------------------------------------------
  // Test Property behavior
  //-------------------------------------------------------------------------
  a.accessibleRoleDescription = testRoleProperty;
  assert.ok( a.accessibleRoleDescription === testRoleProperty.value, 'Setter works with Property' );
  const elementRoleC = getPrimarySiblingElementByNode( a ).getAttribute( 'aria-roledescription' );
  assert.ok( elementRoleC === testRoleProperty.value, 'Element should have the aria-roledescription from the Property' );

  // Update the Property value
  testRoleProperty.value = 'New Value';
  assert.ok( a.accessibleRoleDescription === testRoleProperty.value, 'String Property update updates the accessibleRoleDescription state' );
  const elementRoleD = getPrimarySiblingElementByNode( a ).getAttribute( 'aria-roledescription' );
  assert.ok( elementRoleD === testRoleProperty.value, 'Element should have the aria-roledescription from the Property value change' );

  //-------------------------------------------------------------------------
  // Setting to null removes the value
  //-------------------------------------------------------------------------
  a.accessibleRoleDescription = null;
  const elementRoleE = getPrimarySiblingElementByNode( a ).getAttribute( 'aria-roledescription' );
  assert.ok( elementRoleE === null, 'The element should not have a roledescription once it has been cleared' );

  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );
} );


// these fuzzers take time, so it is nice when they are last
QUnit.test( 'PDOMFuzzer with 3 nodes', assert => {
  const fuzzer = new PDOMFuzzer( 3, false );
  for ( let i = 0; i < 5000; i++ ) {
    fuzzer.step();
  }
  assert.expect( 0 );
  fuzzer.dispose();
} );

QUnit.test( 'PDOMFuzzer with 4 nodes', assert => {
  const fuzzer = new PDOMFuzzer( 4, false );
  for ( let i = 0; i < 1000; i++ ) {
    fuzzer.step();
  }
  assert.expect( 0 );
  fuzzer.dispose();
} );

QUnit.test( 'PDOMFuzzer with 5 nodes', assert => {
  const fuzzer = new PDOMFuzzer( 5, false );
  for ( let i = 0; i < 300; i++ ) {
    fuzzer.step();
  }
  assert.expect( 0 );
  fuzzer.dispose();
} );