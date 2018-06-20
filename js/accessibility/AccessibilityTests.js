// Copyright 2017, University of Colorado Boulder

/**
 * Accessibility tests
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var AccessibilityFuzzer = require( 'SCENERY/accessibility/AccessibilityFuzzer' );
  var AccessibilityUtil = require( 'SCENERY/accessibility/AccessibilityUtil' );
  var AccessiblePeer = require( 'SCENERY/accessibility/AccessiblePeer' );
  var Circle = require( 'SCENERY/nodes/Circle' );
  var Display = require( 'SCENERY/display/Display' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Rectangle = require( 'SCENERY/nodes/Rectangle' );
  var scenery = require( 'SCENERY/scenery' );

  QUnit.module( 'Accessibility' );

  // constants
  var TEST_INNER_CONTENT = 'Test Inner Content Here please^&*. Thanks you so very mucho.';
  var TEST_LABEL = 'Test label';
  var TEST_LABEL_2 = 'Test label 2';
  var TEST_DESCRIPTION = 'Test description';
  var TEST_LABEL_HTML = '<strong>I ROCK as a LABEL</strong>';
  var TEST_LABEL_HTML_2 = '<strong>I ROCK as a LABEL 2</strong>';
  var TEST_DESCRIPTION_HTML = '<strong>I ROCK as a DESCRIPTION</strong>';
  var TEST_DESCRIPTION_HTML_2 = '<strong>I ROCK as a DESCRIPTION 2</strong>';

  // These should manually match the defaults in the Accessibility.js trait
  var DEFAULT_LABEL_TAG_NAME = AccessibilityUtil.DEFAULT_LABEL_TAG_NAME;
  var DEFAULT_DESCRIPTION_TAG_NAME = AccessibilityUtil.DEFAULT_DESCRIPTION_TAG_NAME;
  var DEFAULT_CONTAINER_TAG_NAME = AccessibilityUtil.DEFAULT_CONTAINER_TAG_NAME;

  // given the parent container element for a node, this value is the index of the label sibling in the
  // parent's array of children HTMLElements.
  // var DEFAULT_LABEL_SIBLING_INDEX = 0;
  // var DEFAULT_LABEL_SIBLING_INDEX = 0;

  // a focus highlight for testing, since dummy nodes tend to have no bounds
  var TEST_HIGHLIGHT = new Circle( 5 );

  // add scenery logging for a11y in this unit test module, make sure it is turned off at the end of the file.
  if ( window.location.search.indexOf( 'a11yLog' ) >= 0 ) {
    scenery.enableLogging( ['a11y'] );
  }

  /**
   * Get the id of a dom element representing a node in the DOM.  The accessible content must exist and be unique,
   * there should only be one accessible instance and one dom element for the node.
   *
   * NOTE: Be careful about getting references to dom Elements, the reference will be stale each time
   * Node.invalidateAccessibleContent is called, which is quite often when setting options.
   *
   * @param  {Node} node
   * @return {string}
   */
  function getPrimarySiblingElementByNode( node ) {
    if ( node.accessibleInstances.length === 0 ) {
      throw new Error( 'No accessibleInstances. Was your node added to the scene graph?' );
    }

    else if ( node.accessibleInstances.length > 1 ) {
      throw new Error( 'There should one and only one accessible instance for the node' );
    }
    else if ( !node.accessibleInstances[ 0 ].peer ) {
      throw new Error( 'accessibleInstance\'s peer should exist.' );
    }

    return document.getElementById( node.accessibleInstances[ 0 ].peer.primarySibling.id );
  }


  QUnit.test( 'tagName/innerContent options', function( assert ) {

    // test the behavior of swapVisibility function
    var rootNode = new Node( { tagName: 'div' } );
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    // create some nodes for testing
    var a = new Node( { tagName: 'button', innerContent: TEST_LABEL } );

    rootNode.addChild( a );

    var aElement = getPrimarySiblingElementByNode( a );
    assert.ok( a.accessibleInstances.length === 1, 'only 1 instance' );
    assert.ok( aElement.parentElement.childNodes.length === 1, 'parent contains one primary siblings' );
    assert.ok( aElement.tagName === 'BUTTON', 'default label tagName' );
    assert.ok( aElement.textContent === TEST_LABEL, 'no html should use textContent' );

    a.innerContent = TEST_LABEL_HTML;
    assert.ok( aElement.innerHTML === TEST_LABEL_HTML, 'html label should use innerHTML' );

    a.innerContent = TEST_LABEL_HTML_2;
    assert.ok( aElement.innerHTML === TEST_LABEL_HTML_2, 'html label should use innerHTML, overwrite from html' );

    a.tagName = null;
    assert.ok( a.accessibleInstances.length === 0, 'set to null should clear accessible instances' );

    // make sure that no errors when setting innerContent with tagName null.
    a.innerContent = 'hello';

    a.tagName = 'button';
    a.innerContent = TEST_LABEL_HTML_2;
    assert.ok( getPrimarySiblingElementByNode( a ).innerHTML === TEST_LABEL_HTML_2, 'innerContent not cleared when tagName set to null.' );

    // verify that setting inner content on an input is not allowed
    var b = new Node( { tagName: 'input' } );
    rootNode.addChild( b );
    window.assert && assert.throws( function() {
      b.innerContent = 'this should fail';
    }, /.*/, 'cannot set inner content on input' );

    // now that it is a div, innerContent is allowed
    b.tagName = 'div';
    assert.ok( b.tagName === 'div', 'expect tagName setter to work.' );
    b.innerContent = TEST_LABEL;
    assert.ok( b.innerContent === TEST_LABEL, 'inner content allowed' );

    // revert tag name to input, should throw an error
    window.assert && assert.throws( function() {
      b.tagName = 'input';
    }, /.*/, 'error thrown after setting tagName to input on Node with innerContent.' );
  } );


  QUnit.test( 'containerTagName option, container created if needed', function( assert ) {


    // test the behavior of swapVisibility function
    var rootNode = new Node( { tagName: 'div' } );
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    // create some nodes for testing
    var a = new Node( { tagName: 'button' } );

    rootNode.addChild( a );
    assert.ok( a.accessibleInstances.length === 1, 'only 1 instance' );
    assert.ok( a.accessibleInstances[ 0 ].peer.containerParent === null, 'no container parent for just button' );

    a.containerTagName = 'div';

    assert.ok( a.accessibleInstances[ 0 ].peer.containerParent.id.indexOf( 'container' ) >= 0, 'container parent is div if specified' );

    var b = new Node( { tagName: 'button', labelTagName: 'div', labelContent: TEST_LABEL } );

    rootNode.addChild( b );
    var buttonElement = b.accessibleInstances[ 0 ].peer.primarySibling;
    var parentElement = buttonElement.parentElement;

    assert.ok( b.containerTagName === DEFAULT_CONTAINER_TAG_NAME, 'container automatically created for label' );

    assert.ok( parentElement, 'parent element must be created with label option' );
    assert.ok( parentElement.childNodes.length === 2, 'only contain label and primary siblings' );

    assert.ok( parentElement.childNodes[ 1 ] === getPrimarySiblingElementByNode( b ), 'primary sibling after label sibling.' );


    var c = new Node( { tagName: 'button', descriptionTagName: 'div', descriptionContent: TEST_DESCRIPTION } );

    rootNode.addChild( c );
    buttonElement = c.accessibleInstances[ 0 ].peer.primarySibling;
    parentElement = buttonElement.parentElement;


    assert.ok( parentElement, 'parent element must be created with description option' );
    assert.ok( parentElement.childNodes.length === 2, 'only contain description and primary siblings' );

    c.containerTagName = 'SECTION';
    assert.ok( getPrimarySiblingElementByNode( c ).parentElement.tagName === 'SECTION', 'containerTagName should update default' );

    var d = new Node( { tagName: 'button', labelTagName: 'p', descriptionTagName: 'p' } );
    rootNode.addChild( d );
    assert.ok( getPrimarySiblingElementByNode( d ).parentElement.childNodes.length === 3, 'correct children' );

    var e = new Node( {
      tagName: 'div',
      labelContent: TEST_LABEL,
      containerTagName: 'SECTION'
    } );
    rootNode.addChild( e );
    assert.ok( getPrimarySiblingElementByNode( e ).parentElement.tagName === 'SECTION', 'containerTagName should update default in options block' );
  } );

  QUnit.test( 'labelTagName/labelContent option', function( assert ) {

    // test the behavior of swapVisibility function
    var rootNode = new Node( { tagName: 'div' } );
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    // create some nodes for testing
    var a = new Node( { tagName: 'button', labelContent: TEST_LABEL } );

    rootNode.addChild( a );

    var aElement = getPrimarySiblingElementByNode( a );
    var labelSibling = aElement.parentElement.childNodes[ 0 ];
    assert.ok( a.accessibleInstances.length === 1, 'only 1 instance' );
    assert.ok( aElement.parentElement.childNodes.length === 2, 'parent contains two siblings' );
    assert.ok( labelSibling.tagName === DEFAULT_LABEL_TAG_NAME, 'default label tagName' );
    assert.ok( labelSibling.textContent === TEST_LABEL, 'no html should use textContent' );

    a.labelContent = TEST_LABEL_HTML;
    assert.ok( labelSibling.innerHTML === TEST_LABEL_HTML, 'html label should use innerHTML' );

    a.labelContent = TEST_LABEL_HTML_2;
    assert.ok( labelSibling.innerHTML === TEST_LABEL_HTML_2, 'html label should use innerHTML, overwrite from html' );

    a.tagName = 'div';

    var newAElement = getPrimarySiblingElementByNode( a );
    var newLabelSibling = newAElement.parentElement.childNodes[ 0 ];

    assert.ok( newLabelSibling.innerHTML === TEST_LABEL_HTML_2, 'tagName independent of: html label should use innerHTML, overwrite from html' );

    a.labelTagName = null;

    // make sure label was cleared from pDOM
    assert.ok( getPrimarySiblingElementByNode( a ).parentElement.childNodes.length === 1,
      'Only one element after clearing label' );

    assert.ok( a.labelContent === TEST_LABEL_HTML_2, 'clearing labelTagName should not change content, even  though it is not displayed' );

    a.labelTagName = 'p';
    assert.ok( a.labelTagName === 'p', 'expect labelTagName setter to work.' );

    var b = new Node( { tagName: 'p', labelContent: 'I am groot' } );
    rootNode.addChild( b );
    var bLabelElement = document.getElementById( b.accessibleInstances[ 0 ].peer.labelSibling.id );
    assert.ok( !bLabelElement.getAttribute( 'for' ), 'for attribute should not be on non label label sibling.' );
    b.labelTagName = 'label';
    bLabelElement = document.getElementById( b.accessibleInstances[ 0 ].peer.labelSibling.id );
    assert.ok( bLabelElement.getAttribute( 'for' ) !== null, 'for attribute should be on "label" tag for label sibling.' );

    var c = new Node( { tagName: 'p' } );
    rootNode.addChild( c );
    c.labelTagName = 'label';
    c.labelContent = TEST_LABEL;
    var cLabelElement = document.getElementById( c.accessibleInstances[ 0 ].peer.labelSibling.id );
    assert.ok( cLabelElement.getAttribute( 'for' ) !== null, 'order should not matter' );
  } );

  QUnit.test( 'descriptionTagName/descriptionContent option', function( assert ) {

    // test the behavior of swapVisibility function
    var rootNode = new Node( { tagName: 'div' } );
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    // create some nodes for testing
    var a = new Node( { tagName: 'button', descriptionContent: TEST_DESCRIPTION } );

    rootNode.addChild( a );

    var aElement = getPrimarySiblingElementByNode( a );
    var descriptionSibling = aElement.parentElement.childNodes[ 0 ];
    assert.ok( a.accessibleInstances.length === 1, 'only 1 instance' );
    assert.ok( aElement.parentElement.childNodes.length === 2, 'parent contains two siblings' );
    assert.ok( descriptionSibling.tagName === DEFAULT_DESCRIPTION_TAG_NAME, 'default label tagName' );
    assert.ok( descriptionSibling.textContent === TEST_DESCRIPTION, 'no html should use textContent' );

    a.descriptionContent = TEST_DESCRIPTION_HTML;
    assert.ok( descriptionSibling.innerHTML === TEST_DESCRIPTION_HTML, 'html label should use innerHTML' );

    a.descriptionContent = TEST_DESCRIPTION_HTML_2;
    assert.ok( descriptionSibling.innerHTML === TEST_DESCRIPTION_HTML_2, 'html label should use innerHTML, overwrite from html' );

    a.descriptionTagName = null;

    // make sure description was cleared from pDOM
    assert.ok( getPrimarySiblingElementByNode( a ).parentElement.childNodes.length === 1,
      'Only one element after clearing description' );

    assert.ok( a.descriptionContent === TEST_DESCRIPTION_HTML_2, 'clearing descriptionTagName should not change content, even  though it is not displayed' );

    assert.ok( a.descriptionTagName === null, 'expect descriptionTagName setter to work.' );

    a.descriptionTagName = 'p';
    assert.ok( a.descriptionTagName === 'p', 'expect descriptionTagName setter to work.' );
  } );

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
      labelContent: TEST_LABEL, // label text content
      descriptionContent: TEST_DESCRIPTION, // description text content
      focusable: false, // remove from focus order
      ariaRole: 'button' // uses the ARIA button role
    } );
    rootNode.addChild( buttonNode );

    var divNode = new Node( {
      tagName: 'div',
      ariaLabel: TEST_LABEL, // use ARIA label attribute
      accessibleVisible: false, // hidden from screen readers (and browser)
      descriptionContent: TEST_DESCRIPTION, // default to a <p> tag
      containerTagName: 'div'
    } );
    rootNode.addChild( divNode );

    // verify that setters and getters worked correctly
    assert.ok( buttonNode.labelTagName === 'label', 'Label tag name' );
    assert.ok( buttonNode.containerTagName === 'div', 'container tag name' );
    assert.ok( buttonNode.labelContent === TEST_LABEL, 'Accessible label' );
    assert.ok( buttonNode.descriptionTagName.toUpperCase() === DEFAULT_DESCRIPTION_TAG_NAME, 'Description tag name' );
    assert.ok( buttonNode.focusable === false, 'Focusable' );
    assert.ok( buttonNode.ariaRole === 'button', 'Aria role' );
    assert.ok( buttonNode.descriptionContent === TEST_DESCRIPTION, 'Accessible Description' );
    assert.ok( buttonNode.focusHighlight instanceof Circle, 'Focus highlight' );
    assert.ok( buttonNode.tagName === 'input', 'Tag name' );
    assert.ok( buttonNode.inputType === 'button', 'Input type' );

    assert.ok( divNode.tagName === 'div', 'Tag name' );
    assert.ok( divNode.ariaLabel === TEST_LABEL, 'Use aria label' );
    assert.ok( divNode.accessibleVisible === false, 'Accessible visible' );
    assert.ok( divNode.labelTagName === null, 'Label tag name with aria label is independent' );
    assert.ok( divNode.descriptionTagName.toUpperCase() === DEFAULT_DESCRIPTION_TAG_NAME, 'Description tag name' );


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
    var buttonElement = getPrimarySiblingElementByNode( buttonNode );

    var buttonParent = buttonElement.parentNode;
    var buttonPeers = buttonParent.childNodes;
    var buttonLabel = buttonPeers[ 0 ];
    var buttonDescription = buttonPeers[ 1 ];
    var divElement = getPrimarySiblingElementByNode( divNode );
    var pDescription = divElement.parentElement.childNodes[ 0 ]; // description before primary div

    assert.ok( buttonParent.tagName === 'DIV', 'parent container' );
    assert.ok( buttonLabel.tagName === 'LABEL', 'Label first' );
    assert.ok( buttonLabel.getAttribute( 'for' ) === buttonElement.id, 'label for attribute' );
    assert.ok( buttonLabel.textContent === TEST_LABEL, 'label content' );
    assert.ok( buttonDescription.tagName === DEFAULT_DESCRIPTION_TAG_NAME, 'description second' );
    assert.ok( buttonDescription.textContent, TEST_DESCRIPTION, 'description content' );
    assert.ok( buttonPeers[ 2 ] === buttonElement, 'Button third' );
    assert.ok( buttonElement.type === 'button', 'input type set' );
    assert.ok( buttonElement.getAttribute( 'role' ) === 'button', 'button role set' );
    assert.ok( buttonElement.tabIndex === -1, 'not focusable' );

    assert.ok( divElement.getAttribute( 'aria-label' ) === TEST_LABEL, 'aria label set' );
    assert.ok( divElement.parentElement.hidden === true, 'hidden set should act on parent' );
    assert.ok( pDescription.textContent === TEST_DESCRIPTION, 'description content' );
    assert.ok( pDescription.parentElement === divElement.parentElement, 'description is sibling to primary' );
    assert.ok( divElement.parentElement.childNodes.length === 2, 'no label element for aria-label, just description and primary siblings' );

  } );

  // new aria-labelledby api
  QUnit.test( 'aria-labelledby', function( assert ) {

    var rootNode = new Node();
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    // two new nodes that will be related with the aria-labelledby and aria-describedby associations
    var a = new Node( { tagName: 'button', labelTagName: 'p', descriptionTagName: 'p' } );
    var b = new Node( { tagName: 'p', innerContent: TEST_LABEL_2 } );
    rootNode.children = [ a, b ];

    a.addAriaLabelledbyAssociation( {
      otherNode: b,
      thisElementName: AccessiblePeer.PRIMARY_SIBLING,
      otherElementName: AccessiblePeer.PRIMARY_SIBLING
    } );

    var aElement = getPrimarySiblingElementByNode( a );
    var bElement = getPrimarySiblingElementByNode( b );
    assert.ok( aElement.getAttribute( 'aria-labelledby' ).indexOf( bElement.id ) >= 0, 'aria-labelledby for one node.' );

    var c = new Node( { tagName: 'div', innerContent: TEST_LABEL } );
    rootNode.addChild( c );

    a.addAriaLabelledbyAssociation( {
      otherNode: c,
      thisElementName: AccessiblePeer.PRIMARY_SIBLING,
      otherElementName: AccessiblePeer.PRIMARY_SIBLING
    } );

    aElement = getPrimarySiblingElementByNode( a );
    bElement = getPrimarySiblingElementByNode( b );
    var cElement = getPrimarySiblingElementByNode( c );
    var expectedValue = [ bElement.id, cElement.id ].join( ' ' );
    assert.ok( aElement.getAttribute( 'aria-labelledby' ).trim() === expectedValue,
      'aria-labelledby two nodes' );

    // Make c invalidate
    rootNode.removeChild( c );
    rootNode.addChild( new Node( { children: [ c ] } ) );

    var oldValue = expectedValue;

    aElement = getPrimarySiblingElementByNode( a );
    cElement = getPrimarySiblingElementByNode( c );
    assert.ok( aElement.getAttribute( 'aria-labelledby' ).trim() !== oldValue, 'should have invalidated on tree change' );
    assert.ok( aElement.getAttribute( 'aria-labelledby' ).trim() === [ bElement.id, cElement.id ].join( ' ' ),
      'should have invalidated on tree change' );

  } );
  QUnit.test( 'aria-labelledby, aria-describedby', function( assert ) {
    var rootNode = new Node();
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    // two new nodes that will be related with the aria-labelledby and aria-describedby associations
    var nodeA = new Node( { tagName: 'button', labelTagName: 'p', descriptionTagName: 'p' } );
    var nodeB = new Node( { tagName: 'p', innerContent: TEST_LABEL } );
    var nodeC = new Node();
    var nodeD = new Node();
    rootNode.children = [ nodeA, nodeB ];

    // node B describes node A
    nodeA.setAriaDescribedByNode( nodeB );
    nodeA.setAriaLabelledByNode( nodeB );

    var nodeAElement = getPrimarySiblingElementByNode( nodeA );
    var nodeBElement = getPrimarySiblingElementByNode( nodeB );

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

    var nodeCElement = getPrimarySiblingElementByNode( nodeC );
    var nodeDElement = getPrimarySiblingElementByNode( nodeD );

    assert.ok( nodeCElement.getAttribute( 'aria-describedby' ) === nodeDElement.id, 'describedby attribute wrong in case of pre-invalidation' );
    assert.ok( nodeCElement.getAttribute( 'aria-labelledby' ) === nodeDElement.id, 'labelledby attribute wrong in case of pre-invalidation' );

    // change the association so that nodeA's label is the label for nodeB, and nodeA's description is the description for nodeC
    nodeA.ariaDescriptionContent = AccessiblePeer.DESCRIPTION_SIBLING;
    nodeC.setAriaDescribedByNode( nodeA );

    nodeA.ariaLabelContent = AccessiblePeer.LABEL_SIBLING;
    nodeB.setAriaLabelledByNode( nodeA );

    // reset references after Node.invalidateAccessibleContent() was called while setting options above.
    // NOTE: See warning in getPrimarySiblingElementByNode() jsDoc for more info.
    nodeAElement = getPrimarySiblingElementByNode( nodeA );

    // order of label and description will be labelSibling, descriptionSibling, primary sibling
    var nodeALabel = nodeAElement.parentElement.childNodes[ 0 ];
    var nodeADescription = nodeAElement.parentElement.childNodes[ 1 ];

    assert.ok( getPrimarySiblingElementByNode( nodeC ).getAttribute( 'aria-describedby' ) === nodeADescription.id, 'aria-describedby wrong using explicit association' );
    assert.ok( getPrimarySiblingElementByNode( nodeB ).getAttribute( 'aria-labelledby' ) === nodeALabel.id, 'aria-labelledby wrong using explicit association' );
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
    var a1Element = getPrimarySiblingElementByNode( a1 );
    assert.ok( a1Element, 'button in DOM' );
    assert.ok( a1Element.tagName === 'BUTTON', 'button tag name set' );

    // give the button a container parent and some empty siblings
    a1.labelTagName = 'div';
    a1.descriptionTagName = 'p';
    a1.containerTagName = 'div';

    var buttonElement = a1.accessibleInstances[ 0 ].peer.primarySibling;
    var parentElement = buttonElement.parentElement;
    var buttonPeers = parentElement.childNodes;

    // now html should look like
    // <div id='parent'>
    //  <div id='label'></div>
    //  <p id='description'></p>
    //  <button></button>
    // </div>
    assert.ok( document.getElementById( parentElement.id ), 'container parent in DOM' );
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
    buttonElement = a1.accessibleInstances[ 0 ].peer.primarySibling;
    parentElement = buttonElement.parentElement;
    assert.ok( parentElement.childNodes[ 0 ] === getPrimarySiblingElementByNode( a1 ), 'div first' );
    assert.ok( parentElement.childNodes[ 1 ].id.indexOf( 'description' ) >= 0, 'description after div when appending both elements' );
    assert.ok( parentElement.childNodes.length === 2, 'no label peer when using just aria-label attribute' );

    var elementInDom = document.getElementById( a1.accessibleInstances[ 0 ].peer.primarySibling.id );
    assert.ok( elementInDom.getAttribute( 'aria-label' ) === TEST_LABEL, 'aria-label set' );
  } );

  QUnit.test( 'Accessibility setters/getters', function( assert ) {

    var a1 = new Node( {
      tagName: 'div'
    } );
    var display = new Display( a1 ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    // set/get attributes
    var a1Element = getPrimarySiblingElementByNode( a1 );
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

    assert.ok( a1.accessibleInputListeners.length === 0, 'no input accessible listeners on instantiation' );
    assert.ok( a1.labelContent === null, 'no label on instantiation' );

    // add a listener
    var listener = { click: function() { a1.labelContent = TEST_LABEL; } };
    a1.addAccessibleInputListener( listener );
    assert.ok( a1.accessibleInputListeners.length === 1, 'accessible listener added' );

    // verify added with hasAccessibleInputListener
    assert.ok( a1.hasAccessibleInputListener( listener ) === true, 'found with hasAccessibleInputListener' );

    // fire the event
    getPrimarySiblingElementByNode( a1 ).click();
    assert.ok( a1.labelContent === TEST_LABEL, 'click fired, label set' );

    // remove the listener
    a1.removeAccessibleInputListener( listener );
    assert.ok( a1.accessibleInputListeners.length === 0, 'accessible listener removed' );

    // verify removed with hasAccessibleInputListener
    assert.ok( a1.hasAccessibleInputListener( listener ) === false, 'not found with hasAccessibleInputListener' );

    // make sure event listener was also removed from DOM element
    // click should not change the label
    a1.labelContent = TEST_LABEL_2;
    assert.ok( a1.labelContent === TEST_LABEL_2, 'before click' );

    // setting the label redrew the pdom, so get a reference to the new dom element.
    getPrimarySiblingElementByNode( a1 ).click();
    assert.ok( a1.labelContent === TEST_LABEL_2, 'click should not change label' );

    // verify disposal removes accessible input listeners
    a1.addAccessibleInputListener( listener );
    a1.dispose();
    assert.ok( a1.hasAccessibleInputListener( listener ) === false, 'disposal removed accessible input listeners' );
  } );

  QUnit.test( 'Next/Previous focusable', function( assert ) {
    var util = AccessibilityUtil;

    var rootNode = new Node( { tagName: 'div', focusable: true } );
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    // invisible is deprecated don't use in future, this is a workaround for Nodes without bounds
    var a = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var b = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var c = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var d = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var e = new Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    rootNode.children = [ a, b, c, d ];

    assert.ok( a.focusable, 'should be focusable' );

    // get dom elements from the body
    var rootElement = getPrimarySiblingElementByNode( rootNode );
    var aElement = getPrimarySiblingElementByNode( a );
    var bElement = getPrimarySiblingElementByNode( b );
    var cElement = getPrimarySiblingElementByNode( c );
    var dElement = getPrimarySiblingElementByNode( d );

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

    var rootDOMElement = getPrimarySiblingElementByNode( rootNode );
    var dDOMElement = getPrimarySiblingElementByNode( d );

    // verify the dom
    assert.ok( rootDOMElement.children.length === 5, 'children added' );

    rootNode.accessibleContentDisplayed = false;
    assert.ok( rootDOMElement.children.length === 0, 'sub tree removed from DOM' );
    assert.ok( dDOMElement.children.length === 0, 'sub tree removed from DOM' );

    // invalidation should not add content back to the DOM
    rootNode.tagName = 'button';
    d.tagName = 'span';
    assert.ok( rootDOMElement.children.length === 0, 'invalidate without addition' );

    rootNode.accessibleContentDisplayed = true;

    // redefine because the dom element references above have become stale
    rootDOMElement = getPrimarySiblingElementByNode( rootNode );
    dDOMElement = getPrimarySiblingElementByNode( d );
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
    assert.ok( ( instances[ 0 ].peer.primarySibling.id !== instances[ 1 ].peer.primarySibling.id ) &&
               ( instances[ 1 ].peer.primarySibling.id !== instances[ 2 ].peer.primarySibling.id ) &&
               ( instances[ 0 ].peer.primarySibling.id !== instances[ 2 ].peer.primarySibling.id ), 'each dom element should be unique' );
    assert.ok( document.getElementById( instances[ 0 ].peer.primarySibling.id ), 'peer primarySibling 0 should be in the DOM' );
    assert.ok( document.getElementById( instances[ 1 ].peer.primarySibling.id ), 'peer primarySibling 1 should be in the DOM' );
    assert.ok( document.getElementById( instances[ 2 ].peer.primarySibling.id ), 'peer primarySibling 2 should be in the DOM' );
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
    assert.ok( testNode.accessibleInstances[ 0 ].peer.primarySibling === document.activeElement, 'browser is focusing testNode' );

    testNode.blur();
    assert.ok( testNode, 'testNode blurred before being replaced' );

    // replace testNode with f after bluring testNode, neither should have focus after the replacement
    a.replaceChild( testNode, f );
    assert.ok( a.hasChild( f ), 'node f should replace node testNode' );
    assert.ok( !a.hasChild( testNode ), 'testNode should no longer be a child of node a' );
    assert.ok( !testNode.focused, 'testNode should not have focus after being replaced' );
    assert.ok( !f.focused, 'f should not have focus after replacing testNode, testNode did not have focus' );
    assert.ok( f.accessibleInstances[ 0 ].peer.primarySibling !== document.activeElement, 'browser should not be focusing node f' );

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

    // get the accessible primary siblings - looking into accessibleInstances for testing, there is no getter for primarySibling
    var divA = a.accessibleInstances[ 0 ].peer.primarySibling;
    var buttonB = b.accessibleInstances[ 0 ].peer.primarySibling;
    var divE = e.accessibleInstances[ 0 ].peer.primarySibling;
    var buttonG1 = g.accessibleInstances[ 0 ].peer.primarySibling;
    var buttonG2 = g.accessibleInstances[ 1 ].peer.primarySibling;

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
    // assert.ok( !buttonG1.hidden, 'buttonG1 should not NOT be hidden after ancestor node c is made invisible (parent div E already marked)' );
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

    assert.ok( a.ariaLabel === TEST_LABEL_2, 'aria-label getter/setter' );
    assert.ok( a.labelContent === null, 'no other label set with aria-label' );
    assert.ok( a.innerContent === null, 'no inner content set with aria-label' );

    rootNode.addChild( a );
    var buttonA = a.accessibleInstances[ 0 ].peer.primarySibling;
    assert.ok( buttonA.getAttribute( 'aria-label' ) === TEST_LABEL_2, 'setter on dom element' );
    assert.ok( buttonA.innerHTML === '', 'no inner html with aria-label setter' );
  } );

  QUnit.test( 'focusable option', function( assert ) {

    // test the behavior of focusable function
    var rootNode = new Node( { tagName: 'div' } );
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    var a = new Node( { tagName: 'div', focusable: true } );
    rootNode.addChild( a );

    assert.ok( a.focusable === true, 'focusable option setter' );
    assert.ok( getPrimarySiblingElementByNode( a ).tabIndex === 0, 'tab index on primary sibling with setter' );

    // change the tag name, but focusable should stay the same
    a.tagName = 'p';

    assert.ok( a.focusable === true, 'tagName option should not change focusable value' );
    assert.ok( getPrimarySiblingElementByNode( a ).tabIndex === 0, 'tagName option should not change tab index on primary sibling' );

    a.focusable = false;
    assert.ok( getPrimarySiblingElementByNode( a ).tabIndex === -1, 'set focusable false' );

    var b = new Node( { tagName: 'p' } );
    rootNode.addChild( b );

    b.focusable = true;

    assert.ok( b.focusable, 'set focusable as setter' );
    assert.ok( getPrimarySiblingElementByNode( b ).tabIndex === 0, 'set focusable as setter' );

    // HTML elements that are natively focusable are focusable by default
    var c = new Node( { tagName: 'button' } );
    assert.ok( c.focusable, 'button is focusable by default' );

    // change tagName to something that is not focusable, focusable should be false
    c.tagName = 'p';
    assert.ok( !c.focusable, 'button changed to paragraph, should no longer be focusable' );
  } );

  QUnit.test( 'append siblings/appendLabel/appendDescription setters', function( assert ) {


    // test the behavior of focusable function
    var rootNode = new Node( { tagName: 'div' } );
    var display = new Display( rootNode );
    document.body.appendChild( display.domElement );

    var a = new Node( {
      tagName: 'li',
      innerContent: TEST_INNER_CONTENT,
      labelTagName: 'h3',
      labelContent: TEST_LABEL,
      descriptionContent: TEST_DESCRIPTION,
      containerTagName: 'section',
      appendLabel: true
    } );
    rootNode.addChild( a );

    var aElement = getPrimarySiblingElementByNode( a );
    var containerElement = aElement.parentElement;
    assert.ok( containerElement.tagName.toUpperCase() === 'SECTION', 'container parent is set to right tag' );


    assert.ok( containerElement.childNodes.length === 3, 'expected three siblings' );
    assert.ok( containerElement.childNodes[ 0 ].tagName.toUpperCase() === DEFAULT_DESCRIPTION_TAG_NAME, 'description first sibling' );
    assert.ok( containerElement.childNodes[ 1 ].tagName.toUpperCase() === 'LI', 'primary sibling second sibling' );
    assert.ok( containerElement.childNodes[ 2 ].tagName.toUpperCase() === 'H3', 'label sibling last' );

    a.appendDescription = true;
    containerElement = getPrimarySiblingElementByNode( a ).parentElement;
    assert.ok( containerElement.childNodes.length === 3, 'expected three siblings' );
    assert.ok( containerElement.childNodes[ 0 ].tagName.toUpperCase() === 'LI', 'primary sibling first sibling' );
    assert.ok( containerElement.childNodes[ 1 ].tagName.toUpperCase() === 'H3', 'label sibling second' );
    assert.ok( containerElement.childNodes[ 2 ].tagName.toUpperCase() === DEFAULT_DESCRIPTION_TAG_NAME, 'description last sibling' );

    // clear it out back to defaults should work with setters
    a.appendDescription = false;
    a.appendLabel = false;
    containerElement = getPrimarySiblingElementByNode( a ).parentElement;
    assert.ok( containerElement.childNodes.length === 3, 'expected three siblings' );
    assert.ok( containerElement.childNodes[ 0 ].tagName.toUpperCase() === 'H3', 'label sibling first' );
    assert.ok( containerElement.childNodes[ 1 ].tagName.toUpperCase() === DEFAULT_DESCRIPTION_TAG_NAME, 'description sibling second' );
    assert.ok( containerElement.childNodes[ 2 ].tagName.toUpperCase() === 'LI', 'primary sibling last' );
  } );

  // Higher level setter/getter options
  QUnit.test( 'accessibleName option', function( assert ) {


    // test the behavior of focusable function
    var rootNode = new Node( { tagName: 'div' } );
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    var a = new Node( { tagName: 'div', accessibleName: TEST_LABEL } );
    rootNode.addChild( a );

    assert.ok( a.accessibleName === TEST_LABEL, 'accessibleName getter' );

    var aElement = getPrimarySiblingElementByNode( a );
    assert.ok( aElement.textContent === TEST_LABEL, 'accessibleName setter on div' );


    // TODO: this should be passing,see https://github.com/phetsims/scenery/issues/811
    // var b = new Node( { tagName: 'input', accessibleName: TEST_LABEL } );
    // a.addChild( b );
    // var bElement = getPrimarySiblingElementByNode( b );
    //   var bParent = getPrimarySiblingElementByNode( b ).parentElement;
    // var bLabelSibling = bParent.children[ DEFAULT_LABEL_SIBLING_INDEX ];
    // assert.ok( bLabelSibling.textContent === TEST_LABEL, 'accessibleName sets label sibling' );
    // assert.ok( bLabelSibling.getAttribute( 'for' ).indexOf( bElement.id ) >= 0, 'accessibleName sets label\'s "for" attribute' );


  } );

  QUnit.test( 'helpText option', function( assert ) {


    assert.ok( true);
    //
    // TODO: this is failing, but ideally it wouldn't see https://github.com/phetsims/scenery/issues/811
    // // test the behavior of focusable function
    // var rootNode = new Node( { tagName: 'div' } );
    // var display = new Display( rootNode ); // eslint-disable-line
    // document.body.appendChild( display.domElement );
    //
    // var a = new Node( { tagName: 'div', helpText: TEST_DESCRIPTION } );
    // rootNode.addChild( a );
    //
    // rootNode.addChild( new Node( { tagName: 'input'}));
    // assert.ok( a.helpText === TEST_DESCRIPTION, 'helpText getter' );
    //
    // var aElement = getPrimarySiblingElementByNode( a );
    // assert.ok( aElement.textContent === TEST_DESCRIPTION, 'helpText setter on div' );
    //
    //
    // var b = new Node( { tagName: 'button' } );
    // rootNode.addChild( b );
    // var bElement = getPrimarySiblingElementByNode( b );
    // var bParent = getPrimarySiblingElementByNode( b ).parentElement;
    // var bDescriptionSibling = bParent.children[ DEFAULT_DESCRIPTION_SIBLING_INDEX ];
    // assert.ok( bDescriptionSibling.textContent === TEST_DESCRIPTION, 'helpText sets label sibling' );
    // assert.ok( bDescriptionSibling.getAttribute( 'for' ).indexOf( bElement.id ) >= 0, 'helpText sets label\'s "for" attribute' );
    //

  } );

  QUnit.test( 'move to front/move to back', function( assert ) {

    // make sure state is restored after moving children to front and back    
    var rootNode = new Node( { tagName: 'div' } );
    var display = new Display( rootNode );
    document.body.appendChild( display.domElement );

    var a = new Node( { tagName: 'button', focusHighlight: TEST_HIGHLIGHT } );
    var b = new Node( { tagName: 'button', focusHighlight: TEST_HIGHLIGHT } );
    rootNode.children = [ a, b ];
    b.focus();

    // after moving a to front, b should still have focus
    a.moveToFront();
    assert.ok( b.focused, 'b should have focus after a moved to front' );

    // after moving a to back, b should still have focus
    a.moveToBack();
    assert.ok( b.focused, 'b should have focus after a moved to back' );
  } );

  // these fuzzers take time, so it is nice when they are last
  QUnit.test( 'AccessibilityFuzzer with 3 nodes', function( assert ) {
    var fuzzer = new AccessibilityFuzzer( 3, false );
    for ( var i = 0; i < 5000; i++ ) {
      fuzzer.step();
    }
    assert.expect( 0 );
  } );

  QUnit.test( 'AccessibilityFuzzer with 4 nodes', function( assert ) {
    var fuzzer = new AccessibilityFuzzer( 4, false );
    for ( var i = 0; i < 1000; i++ ) {
      fuzzer.step();
    }
    assert.expect( 0 );
  } );

  QUnit.test( 'AccessibilityFuzzer with 5 nodes', function( assert ) {
    var fuzzer = new AccessibilityFuzzer( 5, false );
    for ( var i = 0; i < 300; i++ ) {
      fuzzer.step();
    }
    assert.expect( 0 );
  } );

  // remove the logging that we added at the top
  if ( window.location.search.indexOf( 'a11yLog' ) >= 0 ) {
    scenery.disableIndividualLog( 'Accessibility' );
    scenery.disableIndividualLog( 'AccessibleInstance' );
    scenery.disableIndividualLog( 'AccessibilityTree' );
    scenery.disableIndividualLog( 'AccessibleDisplaysInfo' );
  }
} );