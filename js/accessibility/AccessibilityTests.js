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
  var AccessiblePeer = require( 'SCENERY/accessibility/AccessiblePeer' );
  var Circle = require( 'SCENERY/nodes/Circle' );
  var Display = require( 'SCENERY/display/Display' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Rectangle = require( 'SCENERY/nodes/Rectangle' );

  QUnit.module( 'Accessibility' );

  var TEST_LABEL = 'Test label';
  var TEST_LABEL_2 = 'Test label 2';
  var TEST_DESCRIPTION = 'Test decsription';

  /**
   * Get the id of a dom element representing a node in the DOM.  The accessible content must exist and be unique,
   * there should only be one accessible instance and one dom element for the node.
   *
   * @param  {Node} node
   * @return {string}
   */
  function getPeerElementId( node ) {
    if ( node.accessibleInstances.length > 0 && !node.accessibleInstances[ 0 ] && !node.accessibleInstances[ 0 ].peer ) {
      throw new Error( 'There should one and only one accessible instance for the node, and the peer should exist' );
    }

    return node.accessibleInstances[ 0 ].peer.domElement.id;
  }

  QUnit.test( 'Accessibility options', function( assert ) {

    var rootNode = new Node();
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    // test setting of accessible content through options
    var buttonNode = new Node( {
      focusHighlight: new Circle( 5 ),
      containerTagName: 'div', // contained in parent element 'div'
      tagName: 'input', // dom element with tag name 'input'
      inputType: 'button', // input type 'button'
      labelTagName: 'label', // label with tagname 'label'
      accessibleLabel: TEST_LABEL, // label text content
      descriptionTagName: 'p', // description tag name
      accessibleDescription: TEST_DESCRIPTION, // description text content
      focusable: false, // remove from focus order
      ariaRole: 'button', // uses the ARIA button role
      prependLabels: true // labels placed above DOM element in read order
    } );
    rootNode.addChild( buttonNode );

    var accessibleNode = new Node( {
      tagName: 'div',
      ariaLabel: TEST_LABEL, // use ARIA label attribute
      accessibleVisible: false, // hidden from screen readers (and browser)
      descriptionTagName: 'p',
      accessibleDescription: TEST_DESCRIPTION
    } );
    rootNode.addChild( accessibleNode );

    // verify that setters and getters worked correctly
    assert.ok( buttonNode.labelTagName === 'label', 'Label tag name' );
    assert.ok( buttonNode.containerTagName === 'div', 'Parent container tag name' );
    assert.ok( buttonNode.accessibleLabel === TEST_LABEL, 'Accessible label' );
    assert.ok( buttonNode.descriptionTagName === 'p', 'Description tag name' );
    assert.ok( buttonNode.focusable === false, 'Focusable' );
    assert.ok( buttonNode.ariaRole === 'button', 'Aria role' );
    assert.ok( buttonNode.accessibleDescription === TEST_DESCRIPTION, 'Accessible Description' );
    assert.ok( buttonNode.prependLabels === true, 'prepend labels' );
    assert.ok( buttonNode.focusHighlight instanceof Circle, 'Focus highlight' );
    assert.ok( buttonNode.tagName === 'input', 'Tag name' );
    assert.ok( buttonNode.inputType === 'button', 'Input type' );

    assert.ok( accessibleNode.tagName === 'div', 'Tag name' );
    assert.ok( accessibleNode.ariaLabel === TEST_LABEL, 'Use aria label' );
    assert.ok( accessibleNode.accessibleVisible === false, 'Accessible visible' );
    assert.ok( accessibleNode.labelTagName === null, 'Label tag name with aria label is independent' );
    assert.ok( accessibleNode.descriptionTagName === 'p', 'Description tag name' );


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
    var buttonElementId = getPeerElementId( buttonNode );
    var buttonElement = document.getElementById( buttonElementId );

    var buttonParent = buttonElement.parentNode;
    var buttonPeers = buttonParent.childNodes;
    var buttonLabel = buttonPeers[ 0 ];
    var buttonDescription = buttonPeers[ 1 ];
    var divElement = document.getElementById( getPeerElementId( accessibleNode ) );
    var divDescription = divElement.childNodes[ 0 ];

    assert.ok( buttonParent.tagName === 'DIV', 'parent container' );
    assert.ok( buttonLabel.tagName === 'LABEL', 'Label first with prependLabels' );
    assert.ok( buttonLabel.getAttribute( 'for' ) === buttonElementId, 'label for attribute' );
    assert.ok( buttonLabel.textContent === TEST_LABEL, 'label content' );
    assert.ok( buttonDescription.tagName === 'P', 'description second with prependLabels' );
    assert.ok( buttonDescription.textContent, TEST_DESCRIPTION, 'description content' );
    assert.ok( buttonPeers[ 2 ] === buttonElement, 'Button third for prepend labels' );
    assert.ok( buttonElement.type === 'button', 'input type set' );
    assert.ok( buttonElement.getAttribute( 'role' ) === 'button', 'button role set' );
    assert.ok( buttonElement.tabIndex === -1, 'not focusable' );

    assert.ok( divElement.getAttribute( 'aria-label' ) === TEST_LABEL, 'aria label set' );
    assert.ok( divElement.hidden === true, 'hidden set' );
    assert.ok( divDescription.textContent === TEST_DESCRIPTION, 'description content' );
    assert.ok( divDescription.parentElement === divElement, 'description is child' );
    assert.ok( divElement.childNodes.length === 1, 'no label element for aria-label' );

  } );

  QUnit.test( 'aria-labelledby, aria-describedby', function( assert ) {
    var rootNode = new Node();
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    // two new nodes that will be related with the aria-labelledby and aria-describedby associations
    var nodeA = new Node( { tagName: 'button', labelTagName: 'p', descriptionTagName: 'p', prependLabels: true } );
    var nodeB = new Node( { tagName: 'p', accessibleLabel: TEST_LABEL } );
    var nodeC = new Node();
    var nodeD = new Node();
    rootNode.children = [ nodeA, nodeB ];

    // node B describes node A
    nodeA.setAriaDescribedByNode( nodeB );
    nodeA.setAriaLabelledByNode( nodeB );

    var nodeAElement = document.getElementById( getPeerElementId( nodeA ) );
    var nodeBElement = document.getElementById( getPeerElementId( nodeB ) );

    assert.ok( nodeAElement.getAttribute( 'aria-describedby' ) === nodeBElement.id, 'describedby attribute wrong in normal use case' );
    assert.ok( nodeAElement.getAttribute( 'aria-labelledby' ) === nodeBElement.id, 'labelledby attribute wrong in normal use case' );

    // set up the relation on nodes that do not have accessible content yet
    nodeC.setAriaDescribedByNode( nodeD );
    nodeC.setAriaLabelledByNode( nodeD );

    // give both accessible content
    nodeC.tagName = 'button';
    nodeD.tagName = 'p';

    // add to DOM so elements can be queried
    rootNode.addChild( nodeC );
    rootNode.addChild( nodeD );

    var nodeCElement = document.getElementById( getPeerElementId( nodeC ) );
    var nodeDElement = document.getElementById( getPeerElementId( nodeD ) );

    assert.ok( nodeCElement.getAttribute( 'aria-describedby' ) === nodeDElement.id, 'describedby attribute wrong in case of pre-invalidation' );
    assert.ok( nodeCElement.getAttribute( 'aria-labelledby' ) === nodeDElement.id, 'labelledby attribute wrong in case of pre-invalidation' );

    // change the association so that nodeA's label is the label for nodeB, and nodeA's description is the description for nodeC
    nodeA.ariaDescriptionContent = AccessiblePeer.DESCRIPTION;
    nodeC.setAriaDescribedByNode( nodeA );

    nodeA.ariaLabelContent = AccessiblePeer.LABEL;
    nodeB.setAriaLabelledByNode( nodeA );

    // order of label and description with prependLabels will be labelElement, descriptionElement, domElement
    var nodeALabel = nodeAElement.parentElement.childNodes[ 0 ];
    var nodeADescription = nodeAElement.parentElement.childNodes[ 1 ];

    assert.ok( nodeCElement.getAttribute( 'aria-describedby' ) === nodeADescription.id, 'aria-describedby wrong using explicit association' );
    assert.ok( nodeBElement.getAttribute( 'aria-labelledby' ) === nodeALabel.id, 'aria-labelledby wrong using explicit association' );
  } );

  QUnit.test( 'Accessibility invalidation', function( assert ) {

    // test invalidation of accessibility (changing content which requires )
    var a1 = new Node();
    var rootNode = new Node();

    a1.tagName = 'button';

    // accessible instances are not sorted until added to a display
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    rootNode.addChild( a1 );

    // verify that elements are in the DOM
    var a1ElementId = getPeerElementId( a1 );
    assert.ok( document.getElementById( a1ElementId ), 'button in DOM' );
    assert.ok( document.getElementById( a1ElementId ).tagName === 'BUTTON', 'button tag name set' );

    // give the button a parent container and some prepended empty labels
    a1.containerTagName = 'div';
    a1.prependLabels = true;
    a1.labelTagName = 'div';
    a1.descriptionTagName = 'p';

    var buttonElement = a1.accessibleInstances[ 0 ].peer.domElement;
    var parentElement = buttonElement.parentElement;
    var buttonPeers = parentElement.childNodes;

    // now html should look like
    // <div id='parent'>
    //  <div id='label'></div>
    //  <p id='description'></p>
    //  <button></button>
    // </div>
    assert.ok( document.getElementById( parentElement.id ), 'parent container in DOM' );
    assert.ok( buttonPeers[ 0 ].tagName === 'DIV', 'label first for prependLabels' );
    assert.ok( buttonPeers[ 1 ].tagName === 'P', 'description second for prependLabels' );
    assert.ok( buttonPeers[ 2 ].tagName === 'BUTTON', 'domElement third for prependLabels' );

    // make the button a div and use an inline label, and place the description below
    a1.tagName = 'div';
    a1.prependLabels = false;
    a1.labelTagName = null; // use aria label attribute instead
    a1.ariaLabel = TEST_LABEL;

    // now the html should look like
    // <div id='parent-id'>
    //  <div></div>
    //  <p id='description'></p>
    // </div>

    // redefine the HTML elements (references will point to old elements before mutation)
    buttonElement = a1.accessibleInstances[ 0 ].peer.domElement;
    parentElement = buttonElement.parentElement;
    assert.ok( parentElement.childNodes[ 0 ] === document.getElementById( getPeerElementId( a1 ) ), 'div first' );
    assert.ok( parentElement.childNodes[ 1 ].id.indexOf('description') >=0, 'description after div without prependLabels' );
    assert.ok( parentElement.childNodes.length === 2, 'no label peer when using just aria-label attribute' );

    var elementInDom = document.getElementById( a1.accessibleInstances[ 0 ].peer.domElement.id );
    assert.ok( elementInDom.getAttribute( 'aria-label' ) === TEST_LABEL, 'aria-label set' );

  } );

  QUnit.test( 'Accessibility setters/getters', function( assert ) {

    var a1 = new Node( {
      tagName: 'div'
    } );
    var display = new Display( a1 ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    // set/get attributes
    var a1Element = document.getElementById( getPeerElementId( a1 ) );
    a1.setAccessibleAttribute( 'role', 'switch' );
    assert.ok( a1.getAccessibleAttributes()[ 0 ].attribute === 'role', 'attribute set' );
    assert.ok( a1Element.getAttribute( 'role' ) === 'switch', 'HTML attribute set' );

    a1.removeAccessibleAttribute( 'role' );
    assert.ok( !a1Element.getAttribute( 'role' ), 'attribute removed' );
  } );

  QUnit.test( 'Accessibility input listeners', function( assert ) {

    // create a node
    var a1 = new Node( {
      tagName: 'button'
    } );
    var display = new Display( a1 ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    var a1Element = document.getElementById( getPeerElementId( a1 ) );

    assert.ok( a1.accessibleInputListeners.length === 0, 'no input accessible listeners on instantiation' );
    assert.ok( a1.accessibleLabel === null, 'no label on instantiation' );

    // add a listener
    var listener = { click: function() { a1.accessibleLabel = TEST_LABEL; } };
    var addedListener = a1.addAccessibleInputListener( listener );
    assert.ok( a1.accessibleInputListeners.length === 1, 'accessible listener added' );

    // fire the event
    a1Element.click();
    assert.ok( a1.accessibleLabel === TEST_LABEL, 'click fired, label set' );

    // remove the listener
    a1.removeAccessibleInputListener( addedListener );
    assert.ok( a1.accessibleInputListeners.length === 0, 'accessible listener removed' );

    // make sure event listener was also removed from DOM element
    // click should not change the label
    a1.accessibleLabel = TEST_LABEL_2;
    a1Element.click();
    assert.ok( a1.accessibleLabel === TEST_LABEL_2 );

  } );

  QUnit.test( 'Next/Previous focusable', function( assert ) {
    var util = AccessibilityUtil;

    var rootNode = new Node( { tagName: 'div', focusable: true } );
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    var a = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var b = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var c = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var d = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var e = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    rootNode.children = [ a, b, c, d ];

    // get dom elements from the body
    var rootElement = document.getElementById( getPeerElementId( rootNode ) );
    var aElement = document.getElementById( getPeerElementId( a ) );
    var bElement = document.getElementById( getPeerElementId( b ) );
    var cElement = document.getElementById( getPeerElementId( c ) );
    var dElement = document.getElementById( getPeerElementId( d ) );

    a.focus();
    assert.ok( document.activeElement.id === aElement.id, 'a in focus (next)' );

    util.getNextFocusable( rootElement ).focus();
    assert.ok( document.activeElement.id === bElement.id, 'b in focus (next)' );

    util.getNextFocusable( rootElement ).focus();
    assert.ok( document.activeElement.id === cElement.id, 'c in focus (next)' );

    util.getNextFocusable( rootElement ).focus();
    assert.ok( document.activeElement.id === dElement.id, 'd in focus (next)' );

    util.getNextFocusable( rootElement ).focus();
    assert.ok( document.activeElement.id === dElement.id, 'd still in focus (next)' );

    util.getPreviousFocusable( rootElement ).focus();
    assert.ok( document.activeElement.id === cElement.id, 'c in focus (previous)' );

    util.getPreviousFocusable( rootElement ).focus();
    assert.ok( document.activeElement.id === bElement.id, 'b in focus (previous)' );

    util.getPreviousFocusable( rootElement ).focus();
    assert.ok( document.activeElement.id === aElement.id, 'a in focus (previous)' );

    util.getPreviousFocusable( rootElement ).focus();
    assert.ok( document.activeElement.id === aElement.id, 'a still in focus (previous)' );

    rootNode.removeAllChildren();
    rootNode.addChild( a );
    a.children = [ b, c ];
    c.addChild( d );
    d.addChild( e );

    // this should hide everything except a
    b.focusable = false;
    c.accessibleVisible = false;

    a.focus();
    util.getNextFocusable( rootElement ).focus();
    assert.ok( document.activeElement.id === aElement.id, 'a only element focusable' );

  } );

  QUnit.test( 'Remove accessibility subtree', function( assert ) {
    var rootNode = new Node( { tagName: 'div', focusable: true } );
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    var a = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var b = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var c = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var d = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var e = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var f = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    rootNode.children = [ a, b, c, d, e ];
    d.addChild( f );

    var rootDOMElement = document.getElementById( getPeerElementId( rootNode ) );
    var dDOMElement = document.getElementById( getPeerElementId( d ) );

    // verify the dom
    assert.ok( rootDOMElement.children.length === 5, 'children added' );

    rootNode.accessibleContentDisplayed = false;
    assert.ok( rootDOMElement.children.length === 0, 'sub tree removed from DOM' );
    assert.ok( dDOMElement.children.length === 0, 'sub tree removed from DOM' );

    // invalidation should not add content back to the DOM
    rootNode.tagName = 'button';
    d.tagName = 'span';
    assert.ok( rootDOMElement.children.length === 0, 'invalidate without addition' );

    window.debug_freez = true;
    rootNode.accessibleContentDisplayed = true;

    // redefine because the dom element references above have become stale
    rootDOMElement = document.getElementById( getPeerElementId( rootNode ) );
    dDOMElement = document.getElementById( getPeerElementId( d ) );
    assert.ok( rootDOMElement.children.length === 5, 'children added back' );
    assert.ok( dDOMElement.children.length === 1, 'descendant child added back' );

  } );

  QUnit.test( 'accessible-dag', function( assert ) {

    // test accessibility for multiple instances of a node
    var rootNode = new Node( { tagName: 'div', focusable: true } );
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    var a = new Node( { tagName: 'div' } );
    var b = new Node( { tagName: 'div' } );
    var c = new Node( { tagName: 'div' } );
    var d = new Node( { tagName: 'div' } );
    var e = new Node( { tagName: 'div' } );

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
    var instances = e.accessibleInstances;
    assert.ok( e.accessibleInstances.length === 3, 'node e should have 3 accessible instances' );
    assert.ok( ( instances[ 0 ].peer.domElement.id !== instances[ 1 ].peer.domElement.id ) &&
               ( instances[ 1 ].peer.domElement.id !== instances[ 2 ].peer.domElement.id ) &&
               ( instances[ 0 ].peer.domElement.id !== instances[ 2 ].peer.domElement.id ), 'each dom element should be unique' );
    assert.ok( document.getElementById( instances[ 0 ].peer.domElement.id ), 'peer domElement 0 should be in the DOM' );
    assert.ok( document.getElementById( instances[ 1 ].peer.domElement.id ), 'peer domElement 1 should be in the DOM' );
    assert.ok( document.getElementById( instances[ 2 ].peer.domElement.id ), 'peer domElement 2 should be in the DOM' );
  } );

  QUnit.test( 'replaceChild', function( assert ) {

    // test the behavior of replaceChild function
    var rootNode = new Node( { tagName: 'div' } );
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    // a custom focus highlight (since dummy node's have no bounds)
    var focusHighlight = new Rectangle( 0, 0, 10, 10 );

    // create some nodes for testing
    var a = new Node( { tagName: 'button', focusHighlight: focusHighlight } );
    var b = new Node( { tagName: 'button', focusHighlight: focusHighlight } );
    var c = new Node( { tagName: 'button', focusHighlight: focusHighlight } );
    var d = new Node( { tagName: 'button', focusHighlight: focusHighlight } );
    var e = new Node( { tagName: 'button', focusHighlight: focusHighlight } );
    var f = new Node( { tagName: 'button', focusHighlight: focusHighlight } );

    // a child that will be added through replaceChild()
    var testNode = new Node( { tagName: 'button', focusHighlight: focusHighlight } );

    // make sure replaceChild puts the child in the right spot
    a.children = [ b, c, d, e, f ];
    var initIndex = a.indexOfChild( e );
    a.replaceChild( e, testNode );
    var afterIndex = a.indexOfChild( testNode );

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
    assert.ok( testNode.accessibleInstances[ 0 ].peer.domElement === document.activeElement, 'browser is focusing testNode' );

    testNode.blur();
    assert.ok( testNode, 'testNode blurred before being replaced' );

    // replace testNode with f after bluring testNode, neither should have focus after the replacement
    a.replaceChild( testNode, f );
    assert.ok( a.hasChild( f ), 'node f should replace node testNode' );
    assert.ok( !a.hasChild( testNode ), 'testNode should no longer be a child of node a' );
    assert.ok( !testNode.focused, 'testNode should not have focus after being replaced' );
    assert.ok( !f.focused, 'f should not have focus after replacing testNode, testNode did not have focus' );
    assert.ok( f.accessibleInstances[ 0 ].peer.domElement !== document.activeElement, 'browser should not be focusing node f' );

    // focus node d and replace with non-focusable testNode, neither should have focus since testNode is not focusable
    d.focus();
    testNode.focusable = false;
    assert.ok( d.focused, 'd has focus before being replaced' );
    assert.ok( !testNode.focusable, 'testNode is not focusable before replacing node d' );

    b.replaceChild( d, testNode );
    assert.ok( b.hasChild( testNode ), 'testNode should be a child of node b after replacing with replaceChild' );
    assert.ok( !b.hasChild( d ), 'd should not be a child of b after it was replaced with replaceChild' );
    assert.ok( !d.focused, 'do does not have focus after being replaced by testNode' );
    assert.ok( !testNode.focused, 'testNode does not have focus after replacing node d (testNode is not focusable)' );
  } );

  QUnit.test( 'accessibleVisible', function( assert ) {

    var rootNode = new Node();
    var display = new Display( rootNode );
    document.body.appendChild( display.domElement );

    // test with a scene graph
    //       a
    //      / \
    //     b    c
    //        / | \
    //       d  e  f
    //           \ /
    //            g
    var a = new Node();
    var b = new Node();
    var c = new Node();
    var d = new Node();
    var e = new Node();
    var f = new Node();
    var g = new Node();

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

    // get the accessible DOM elements - looking into accessibleInstances for testing, there is no getter for domElement
    var divA = a.accessibleInstances[ 0 ].peer.domElement;
    var buttonB = b.accessibleInstances[ 0 ].peer.domElement;
    var divE = e.accessibleInstances[ 0 ].peer.domElement;
    var buttonG1 = g.accessibleInstances[ 0 ].peer.domElement;
    var buttonG2 = g.accessibleInstances[ 1 ].peer.domElement;

    var divAChildren = divA.childNodes;
    var divEChildren = divE.childNodes;

    assert.ok( _.includes( divAChildren, buttonB ), 'button B should be an immediate child of div A' );
    assert.ok( _.includes( divAChildren, divE ), 'div E should be an immediate child of div A' );
    assert.ok( _.includes( divAChildren, buttonG2 ), 'button G2 should be an immediate child of div A' );
    assert.ok( _.includes( divEChildren, buttonG1 ), 'button G1 should be an immediate child of div E' );

    // make node B invisible for accessibility - it should should visible, but hidden from screen readers
    b.accessibleVisible = false;
    assert.ok( b.visible === true, 'b should be visible after becoming hidden for screen readers' );
    assert.ok( b.accessibleVisible === false, 'b state should reflect it is hidden for screen readers' );
    assert.ok( buttonB.hidden === true, 'buttonB should be hidden for screen readers' );
    b.accessibleVisible = true;

    // make node B invisible - it should not be visible, and it should be hidden for screen readers
    b.visible = false;
    assert.ok( b.visible === false, 'state of node b is visible' );
    assert.ok( buttonB.hidden === true, 'buttonB is hidden from screen readers after becoming invisible' );
    assert.ok( b.accessibleVisible === true, 'state of node b still reflects accessible visibility when invisible' );
    b.visible = true;

    // make node f invisible - g's trail that goes through f should be invisible to AT, the child of c should remain accessibleVisible
    f.visible = false;
    assert.ok( g.getAccessibleVisible() === true, 'state of accessibleVisible should remain true on node g' );
    assert.ok( !buttonG1.hidden, 'buttonG1 (child of e) should not be hidden after parent node f made invisible (no accessible content on node f)' );
    assert.ok( buttonG2.hidden === true, 'buttonG2 should be hidden after parent node f made invisible (no accessible content on node f)' );
    f.visible = true;

    // make node c (no accessible content) invisible to screen, e should be hidden and g2 should be hidden
    c.accessibleVisible = false;
    assert.ok( c.visible === true, 'c should still be visible after becoming invisible to screen readers' );
    assert.ok( divE.hidden === true, 'div E should be hidden after parent node c (no accessible content) is made invisible to screen readers' );
    assert.ok( buttonG2.hidden === true, 'buttonG2 should be hidden after ancestor node c (no accessible content) is made invisible to screen readers' );
    assert.ok( !buttonG1.hidden, 'buttonG1 should not NOT be hidden after ancestor node c is made invisible (parent div E already marked)' );
    assert.ok( !divA.hidden, 'div A should not have been hidden by making descendant c invisible to screen readers' );
  } );

  QUnit.test( 'swapVisibility', function( assert ) {


    // test the behavior of swapVisibility function
    var rootNode = new Node( { tagName: 'div' } );
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    // a custom focus highlight (since dummy node's have no bounds)
    var focusHighlight = new Rectangle( 0, 0, 10, 10 );

    // create some nodes for testing
    var a = new Node( { tagName: 'button', focusHighlight: focusHighlight } );
    var b = new Node( { tagName: 'button', focusHighlight: focusHighlight } );
    var c = new Node( { tagName: 'button', focusHighlight: focusHighlight } );

    rootNode.addChild( a );
    a.children = [ b, c ];

    // swap visibility between two nodes, visibility should be swapped and neither should have keyboard focus
    b.visible = true;
    c.visible = false;
    b.swapVisibility( c );
    assert.ok( b.visible === false, 'b should now be invisible' );
    assert.ok( c.visible === true, 'c should now be visible' );
    assert.ok( b.focused === false, 'b should not have focus after being made invisible' );
    assert.ok( c.focused === false, 'c should not have  focus since b did not have focus' );

    // swap visibility between two nodes where the one that is initially visible has keyboard focus, the newly visible
    // node then receive focus
    b.visible = true;
    c.visible = false;
    b.focus();
    b.swapVisibility( c );
    assert.ok( b.visible === false, 'b should be invisible after swapVisibility' );
    assert.ok( c.visible === true, 'c should be visible after  swapVisibility' );
    assert.ok( b.focused === false, 'b should no longer have focus  after swapVisibility' );
    assert.ok( c.focused === true, 'c should now have focus after swapVisibility' );

    // swap visibility between two nodes where the one that is initially visible has keyboard focus, the newly visible
    // node then receive focus - like the previous test but c.swapVisibility( b ) is the same as b.swapVisibility( c )
    b.visible = true;
    c.visible = false;
    b.focus();
    b.swapVisibility( c );
    assert.ok( b.visible === false, 'b should be invisible after swapVisibility' );
    assert.ok( c.visible === true, 'c should be visible after  swapVisibility' );
    assert.ok( b.focused === false, 'b should no longer have focus  after swapVisibility' );
    assert.ok( c.focused === true, 'c should now have focus after swapVisibility' );

    // swap visibility between two nodes where the first node has focus, but the second node is not focusable. After
    // swapping, neither should have focus
    b.visible = true;
    c.visible = false;
    b.focus();
    c.focusable = false;
    b.swapVisibility( c );
    assert.ok( b.visible === false, 'b should be invisible after visibility is swapped' );
    assert.ok( c.visible === true, 'c should be visible after visibility is swapped' );
    assert.ok( b.focused === false, 'b should no longer have focus after visibility is swapped' );
    assert.ok( c.focused === false, 'c should not have focus after visibility is swapped because it is not focusable' );
  } );

  QUnit.test( 'Aria Label Setter', function( assert ) {


    // test the behavior of swapVisibility function
    var rootNode = new Node( { tagName: 'div' } );
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    // create some nodes for testing
    var a = new Node( { tagName: 'button', ariaLabel: TEST_LABEL_2 } );

    assert.ok( a.ariaLabel === TEST_LABEL_2, 'aria-label getter/setter');
    assert.ok( a.accessibleLabel === null, 'no other label set with aria-label' );

    rootNode.addChild( a);
    var buttonA = a.accessibleInstances[ 0 ].peer.domElement;
    assert.ok( buttonA.getAttribute( 'aria-label' ) === TEST_LABEL_2, 'setter on dom element');
  });

  } );