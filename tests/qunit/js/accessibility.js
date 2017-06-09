// Copyright 2002-2016, University of Colorado Boulder

(function() {
  'use strict';

  module( 'Scenery: Accessibility' );

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

  test( 'Accessibility options', function() {

    var rootNode = new scenery.Node();
    var display = new scenery.Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    // test setting of accessible content through options
    var buttonNode = new scenery.Node( {
      focusHighlight: new scenery.Circle( 5 ),
      parentContainerTagName: 'div', // contained in parent element 'div'
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

    var accessibleNode = new scenery.Node( {
      tagName: 'div',
      useAriaLabel: true, // use ARIA label attribute
      accessibleHidden: true, // hidden from screen readers (and browser)
      accessibleLabel: TEST_LABEL,
      labelTagName: 'p',
      descriptionTagName: 'p',
      accessibleDescription: TEST_DESCRIPTION
    } );
    rootNode.addChild( accessibleNode );

    // verify that setters and getters worked correctly
    ok( buttonNode.labelTagName === 'label', 'Label tag name' );
    ok( buttonNode.parentContainerTagName === 'div', 'Parent container tag name' );
    ok( buttonNode.accessibleLabel === TEST_LABEL, 'Accessible label' );
    ok( buttonNode.descriptionTagName === 'p', 'Description tag name' );
    ok( buttonNode.focusable === false, 'Focusable' );
    ok( buttonNode.ariaRole === 'button', 'Aria role' );
    ok( buttonNode.accessibleDescription === TEST_DESCRIPTION, 'Accessible Description' );
    ok( buttonNode.prependLabels === true, 'prepend labels' );
    ok( buttonNode.focusHighlight instanceof scenery.Circle, 'Focus highlight' );
    ok( buttonNode.tagName === 'input', 'Tag name' );
    ok( buttonNode.inputType === 'button', 'Input type' );

    ok( accessibleNode.tagName === 'div', 'Tag name' );
    ok( accessibleNode.useAriaLabel === true, 'Use aria label' );
    ok( accessibleNode.accessibleHidden === true, 'Accessible hidden' );
    ok( accessibleNode.accessibleLabel === TEST_LABEL, 'Accessible label' );
    ok( accessibleNode.labelTagName === null, 'Label tag name with aria label' );
    ok( accessibleNode.descriptionTagName === 'p', 'Description tag name' );


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

    ok( buttonParent.tagName === 'DIV', 'parent container' );
    ok( buttonLabel.tagName === 'LABEL', 'Label first with prependLabels' );
    ok( buttonLabel.getAttribute( 'for' ) === buttonElementId, 'label for attribute' );
    ok( buttonLabel.textContent === TEST_LABEL, 'label content' );
    ok( buttonDescription.tagName === 'P', 'description second with prependLabels' );
    ok( buttonDescription.textContent, TEST_DESCRIPTION, 'description content' );
    ok( buttonPeers[ 2 ] === buttonElement, 'Button third for prepend labels' );
    ok( buttonElement.type === 'button', 'input type set' );
    ok( buttonElement.getAttribute( 'role' ) === 'button', 'button role set' );
    ok( buttonElement.tabIndex === -1, 'not focusable' );

    ok( divElement.getAttribute( 'aria-label' ) === TEST_LABEL, 'aria label set' );
    ok( divElement.hidden === true, 'hidden set' );
    ok( divDescription.textContent === TEST_DESCRIPTION, 'description content' );
    ok( divDescription.parentElement === divElement, 'description is child' );
    ok( divElement.childNodes.length === 1, 'no label element for aria-label' );

  } );

  test( 'aria-labelledby, aria-describedby', function() {
    var rootNode = new scenery.Node();
    var display = new scenery.Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );
    
    // two new nodes that will be related with the aria-labelledby and aria-describedby associations
    var nodeA = new scenery.Node( { tagName: 'button', labelTagName: 'p', descriptionTagName: 'p', prependLabels: true } );
    var nodeB = new scenery.Node( {tagName: 'p', accessibleLabel: TEST_LABEL } );
    var nodeC = new scenery.Node();
    var nodeD = new scenery.Node();
    rootNode.children = [ nodeA, nodeB ];

    // node B describes node A
    nodeB.setAriaDescribesNode( nodeA );
    nodeB.setAriaLabelsNode( nodeA );

    var nodeAElement = document.getElementById( getPeerElementId( nodeA ) );
    var nodeBElement = document.getElementById( getPeerElementId( nodeB ) );

    ok( nodeAElement.getAttribute( 'aria-describedby' ) === nodeBElement.id, 'describedby attribute wrong in normal use case' );
    ok( nodeAElement.getAttribute( 'aria-labelledby' ) === nodeBElement.id, 'labelledby attribute wrong in normal use case' );

    // set up the relation on nodes that do not have accessible content yet
    nodeD.setAriaDescribesNode( nodeC );
    nodeD.setAriaLabelsNode( nodeC );

    // give both accessible content
    nodeC.tagName = 'button';
    nodeD.tagName = 'p';

    // add to DOM so elements can be queried
    rootNode.addChild( nodeC );
    rootNode.addChild( nodeD );

    var nodeCElement = document.getElementById( getPeerElementId( nodeC ) );
    var nodeDElement = document.getElementById( getPeerElementId( nodeD ) );

    ok( nodeCElement.getAttribute( 'aria-describedby' ) === nodeDElement.id, 'describedby attribute wrong in case of pre-invalidation' );
    ok( nodeCElement.getAttribute( 'aria-labelledby' ) === nodeDElement.id, 'labelledby attribute wrong in case of pre-invalidation' );

    // change the association so that nodeA's label is the label for nodeB, and nodeA's description is the description for nodeC
    nodeA.setAriaDescribesNode( nodeC, scenery.AccessiblePeer.DESCRIPTION );
    nodeA.setAriaLabelsNode( nodeB, scenery.AccessiblePeer.LABEL );

    // order of label and description with prependLabels will be labelElement, descriptionElement, domElement
    var nodeALabel = nodeAElement.parentElement.childNodes[ 0 ];
    var nodeADescription = nodeAElement.parentElement.childNodes[ 1 ];

    ok( nodeCElement.getAttribute( 'aria-describedby' ) === nodeADescription.id, 'aria-describedby wrong using explicit association' );
    ok( nodeBElement.getAttribute( 'aria-labelledby' ) === nodeALabel.id, 'aria-labelledby wrong using explicit association' );
  } );

  test( 'Accessibility invalidation', function() {

    // test invalidation of accessibility (changing content which requires )
    var a1 = new scenery.Node();
    var rootNode = new scenery.Node();

    a1.tagName = 'button';

    // accessible instances are not sorted until added to a display
    var display = new scenery.Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );
    
    rootNode.addChild( a1 );

    // verify that elements are in the DOM
    var a1ElementId = getPeerElementId( a1 );
    ok( document.getElementById( a1ElementId ), 'button in DOM' );
    ok( document.getElementById( a1ElementId ).tagName === 'BUTTON', 'button tag name set' );

    // give the button a parent container and some prepended empty labels
    a1.parentContainerTagName = 'div';
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
    ok( document.getElementById( parentElement.id ), 'parent container in DOM' );
    ok( buttonPeers[ 0 ].tagName === 'DIV', 'label first for prependLabels' );
    ok( buttonPeers[ 1 ].tagName === 'P', 'description second for prependLabels' );
    ok( buttonPeers[ 2 ].tagName === 'BUTTON', 'domElement third for prependLabels' );

    // make the button a div and use an inline label, and place the description below
    a1.tagName = 'div';
    a1.accessibleLabel = TEST_LABEL;
    a1.prependLabels = false;
    a1.useAriaLabel = true;

    // now the html should look like
    // <div id='parent-id'>
    //  <div></div>
    //  <p id='description'></p>
    // </div>
     
    // redefine the HTML elements (references will point to old elements before mutation)
    buttonElement = a1.accessibleInstances[ 0 ].peer.domElement;
    parentElement = buttonElement.parentElement;
    buttonPeers = parentElement.childNodes;
    ok( parentElement.childNodes[ 0 ] === document.getElementById( getPeerElementId( a1 ) ), 'div first' );
    ok( parentElement.childNodes[ 1 ].tagName === 'P', 'description after div without prependLabels' );
    ok( parentElement.childNodes.length === 2, 'label removed' );

    var elementInDom = document.getElementById( a1.accessibleInstances[ 0 ].peer.domElement.id );
    ok( elementInDom.getAttribute( 'aria-label' ) === TEST_LABEL, 'aria-label set' );

  } );

  test( 'Accessibility setters/getters', function() {

    var a1 = new scenery.Node( {
      tagName: 'div'
    } );
    var display = new scenery.Display( a1 ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    // set/get attributes
    var a1Element = document.getElementById( getPeerElementId( a1 ) );
    a1.setAccessibleAttribute( 'role', 'switch' );
    ok( a1.getAccessibleAttributes()[ 0 ].attribute === 'role', 'attribute set' );
    ok( a1Element.getAttribute( 'role' ) === 'switch', 'HTML attribute set' );

    a1.removeAccessibleAttribute( 'role' );
    ok( !a1Element.getAttribute( 'role' ), 'attribute removed' );
  } );

  test( 'Accessibility input listeners', function() {

    // create a node
    var a1 = new scenery.Node( {
      tagName: 'button'
    } );
    var display = new scenery.Display( a1 ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    var a1Element = document.getElementById( getPeerElementId( a1 ) );

    ok( a1.accessibleInputListeners.length === 0, 'no input accessible listeners on instantiation' );
    ok( a1.accessibleLabel === null, 'no label on instantiation' );

    // add a listener
    var listener = { click: function() { a1.accessibleLabel = TEST_LABEL; } };
    var addedListener = a1.addAccessibleInputListener( listener );
    ok( a1.accessibleInputListeners.length === 1, 'accessible listener added' );

    // fire the event
    a1Element.click();
    ok( a1.accessibleLabel === TEST_LABEL, 'click fired, label set' );

    // remove the listener
    a1.removeAccessibleInputListener( addedListener );
    ok( a1.accessibleInputListeners.length === 0, 'accessible listener removed' );

    // make sure event listener was also removed from DOM element
    // click should not change the label
    a1.accessibleLabel = TEST_LABEL_2;
    a1Element.click();
    ok( a1.accessibleLabel === TEST_LABEL_2 );

  } );

  test( 'Next/Previous focusable', function() {
    var util = scenery.AccessibilityUtil;

    var rootNode = new scenery.Node( { tagName: 'div', focusable: true } );
    var display = new scenery.Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    var a = new scenery.Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var b = new scenery.Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var c = new scenery.Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var d = new scenery.Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var e = new scenery.Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    rootNode.children = [ a, b, c, d ];

    // get dom elements from the body
    var rootElement = document.getElementById( getPeerElementId( rootNode ) );
    var aElement = document.getElementById( getPeerElementId( a ) );
    var bElement = document.getElementById( getPeerElementId( b ) );
    var cElement = document.getElementById( getPeerElementId( c ) );
    var dElement = document.getElementById( getPeerElementId( d ) );
    
    a.focus();
    ok( document.activeElement.id === aElement.id, 'a in focus (next)' );

    util.getNextFocusable( rootElement ).focus();
    ok( document.activeElement.id === bElement.id, 'b in focus (next)' );

    util.getNextFocusable( rootElement ).focus();
    ok( document.activeElement.id === cElement.id, 'c in focus (next)' );

    util.getNextFocusable( rootElement ).focus();
    ok( document.activeElement.id === dElement.id, 'd in focus (next)' );

    util.getNextFocusable( rootElement ).focus();
    ok( document.activeElement.id === dElement.id, 'd still in focus (next)' );

    util.getPreviousFocusable( rootElement ).focus();
    ok( document.activeElement.id === cElement.id, 'c in focus (previous)' );

    util.getPreviousFocusable( rootElement ).focus();
    ok( document.activeElement.id === bElement.id, 'b in focus (previous)' );

    util.getPreviousFocusable( rootElement ).focus();
    ok( document.activeElement.id === aElement.id, 'a in focus (previous)' );

    util.getPreviousFocusable( rootElement ).focus();
    ok( document.activeElement.id === aElement.id, 'a still in focus (previous)' );

    rootNode.removeAllChildren();
    rootNode.addChild( a );
    a.children = [ b, c ];
    c.addChild( d );
    d.addChild( e );

    // this should hide everything except a
    b.focusable = false;
    c.accessibleHidden = true;

    a.focus();
    util.getNextFocusable( rootElement ).focus();
    ok( document.activeElement.id === aElement.id, 'a only element focusable' );

  } );

  test( 'Remove accessibility subtree', function() {
    var rootNode = new scenery.Node( { tagName: 'div', focusable: true } );
    var display = new scenery.Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    var a = new scenery.Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var b = new scenery.Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var c = new scenery.Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var d = new scenery.Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var e = new scenery.Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var f = new scenery.Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    rootNode.children = [ a, b, c, d, e ];
    d.addChild( f );

    var rootDOMElement = document.getElementById( getPeerElementId( rootNode ) );
    var dDOMElement = document.getElementById( getPeerElementId( d ) );

    // verify the dom
    ok( rootDOMElement.children.length === 5, 'children added' );

    rootNode.accessibleContentDisplayed = false;
    ok ( rootDOMElement.children.length === 0, 'sub tree removed from DOM' );
    ok ( dDOMElement.children.length === 0, 'sub tree removed from DOM' );

    // invalidation should not add content back to the DOM
    rootNode.tagName = 'button';
    d.tagName = 'span';
    ok ( rootDOMElement.children.length === 0, 'invalidate without addition' );

    window.debug_freez = true;
    rootNode.accessibleContentDisplayed = true;

    // redefine because the dom element references above have become stale
    rootDOMElement = document.getElementById( getPeerElementId( rootNode ) );
    dDOMElement = document.getElementById( getPeerElementId( d ) );
    ok( rootDOMElement.children.length === 5, 'children added back' );
    ok( dDOMElement.children.length === 1, 'descendant child added back' );

  } );

  test( 'accessible-dag', function() {

    // test accessibility for multiple instances of a node
    var rootNode = new scenery.Node( { tagName: 'div', focusable: true } );
    var display = new scenery.Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    var a = new scenery.Node( { tagName: 'div' } );
    var b = new scenery.Node( { tagName: 'div' } );
    var c = new scenery.Node( { tagName: 'div' } );
    var d = new scenery.Node( { tagName: 'div' } );
    var e = new scenery.Node( { tagName: 'div' } );

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
    ok( e.accessibleInstances.length === 3, 'node e should have 3 accessible instances' );
    ok( ( instances[ 0 ].peer.domElement.id !== instances[ 1 ].peer.domElement.id ) &&
        ( instances[ 1 ].peer.domElement.id !== instances[ 2 ].peer.domElement.id ) &&
        ( instances[ 0 ].peer.domElement.id !== instances[ 2 ].peer.domElement.id ), 'each dom element should be unique' );
    ok( document.getElementById( instances[ 0 ].peer.domElement.id ), 'peer domElement 0 should be in the DOM' );
    ok( document.getElementById( instances[ 1 ].peer.domElement.id ), 'peer domElement 1 should be in the DOM' );
    ok( document.getElementById( instances[ 2 ].peer.domElement.id ), 'peer domElement 2 should be in the DOM' );
  } );
})();
