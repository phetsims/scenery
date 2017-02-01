// Copyright 2002-2016, University of Colorado Boulder

(function() {
  'use strict';

  module( 'Scenery: Accessibility' );

  var TEST_LABEL = 'Test label';
  var TEST_LABEL_2 = 'Test label 2';
  var TEST_DESCRIPTION = 'Test decsription';
  var TEST_DESCRIPTION_2 = 'Test decsription 2';
  var TEST_TITLE = 'Test title';

  test( 'Accessibility options', function() {

    var rootNode = new scenery.Node();
    var display = new scenery.Display( rootNode );

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
      accessibleDescription: TEST_DESCRIPTION,
      ariaLabelledByElement: buttonNode.domElement, // ARIA label relation
      ariaDescribedByElement: buttonNode.domElement // ARIA description relation
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
    ok( accessibleNode.ariaLabelledByElement === buttonNode.domElement, 'aria-labelledby id' );
    ok( accessibleNode.ariaDescribedByElement === buttonNode.domElement, ' aria-describedby id' );


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
    var buttonElement = document.getElementById( buttonNode.accessibleId );
    var buttonParent = buttonElement.parentNode;
    var buttonPeers = buttonParent.childNodes;
    var buttonLabel = buttonPeers[ 0 ];
    var buttonDescription = buttonPeers[ 1 ];
    var divElement = document.getElementById( accessibleNode.accessibleId );
    var divDescription = divElement.childNodes[ 0 ];

    ok( buttonParent.tagName === 'DIV', 'parent container' );
    ok( buttonLabel.tagName === 'LABEL', 'Label first with prependLabels' );
    ok( buttonLabel.getAttribute( 'for' ) === buttonNode.accessibleId, 'label for attribute' );
    ok( buttonLabel.textContent === TEST_LABEL, 'label content' );
    ok( buttonDescription.tagName === 'P', 'description second with prependLabels' );
    ok( buttonDescription.textContent, TEST_DESCRIPTION, 'description content' );
    ok( buttonPeers[ 2 ] === buttonElement, 'Button third for prepend labels' );
    ok( buttonElement.type === 'button', 'input type set' );
    ok( buttonElement.getAttribute( 'role' ) === 'button', 'button role set' );
    ok( buttonElement.tabIndex === -1, 'not focusable' );

    ok( divElement.getAttribute( 'aria-label' ) === TEST_LABEL, 'aria label set' );
    ok( divElement.hidden === true, 'hidden set' );
    ok( divElement.getAttribute( 'aria-labelledby' ) === buttonNode.accessibleId, 'aria-labelledby set' );
    ok( divElement.getAttribute( 'aria-describedby' ) === buttonNode.accessibleId, 'aria-describedby set' );
    ok( divDescription.textContent === TEST_DESCRIPTION, 'description content' );
    ok( divDescription.parentElement === divElement, 'description is child' );
    ok( divElement.childNodes.length === 1, 'no label element for aria-label' );

  } );

  test( 'Accessibility invalidation', function() {

    // test invalidation of accessibility (changing tag names)
    var a1 = new scenery.Node();
    var rootNode = new scenery.Node();

    a1.tagName = 'button';

    // accessible instances are not sorted until added to a display
    var display = new scenery.Display( rootNode );
    rootNode.addChild( a1 );

    // verify that elements are in the DOM
    ok( document.getElementById( a1.accessibleId ), 'button in DOM' );
    ok( document.getElementById( a1.accessibleId ).tagName === 'BUTTON', 'button tag name set' )

    // give the button a parent container and some prepended empty labels
    a1.parentContainerTagName = 'div';
    a1.prependLabels = true;
    a1.labelTagName = 'div';
    a1.descriptionTagName = 'p';

    // now html should look like
    // <div id='parent'>
    //  <div id='label'></div>
    //  <p id='description'></p>
    //  <button></button>
    // </div>
    var buttonPeers = a1.parentContainerElement.childNodes;
    ok( document.getElementById( a1.parentContainerElement.id ), 'parent container in DOM' );
    ok( buttonPeers[ 0 ].tagName === 'DIV', 'label first for prependLabels' );
    ok( buttonPeers[ 1 ].tagName === 'P', 'description second for prependLabels' );
    ok( buttonPeers[ 2 ].tagName === 'BUTTON', 'domElement third for prependLabels' );

    // make the button a div and use an inline label, and place the description below
    a1.tagName = 'div';
    a1.accessibleLabel = TEST_LABEL;
    a1.labelTagName = null;
    a1.prependLabels = false;
    a1.useAriaLabel = true;

    // now the html should look like
    // <div id='parent-id'>
    //  <div></div>
    //  <p id='description'></p>
    // </div>
    ok( a1.parentContainerElement.childNodes[ 0 ] === document.getElementById( a1.accessibleId ), 'div first' );
    ok( a1.parentContainerElement.childNodes[ 1 ].tagName === 'P', 'description after div without prependLabels' );
    ok( a1.parentContainerElement.childNodes.length === 2, 'label removed' );
    ok( document.getElementById( a1.accessibleId).getAttribute( 'aria-label' ) === TEST_LABEL, 'aria-label set' );

  } );

  test( 'Accessibility setters/getters', function() {

    var a1 = new scenery.Node( {
      tagName: 'div'
    } );
    var display = new scenery.Display( a1 );

    // set/get attributes
    a1.setAccessibleAttribute( 'role', 'switch' );
    ok( a1.getAccessibleAttributes()[ 0 ].attribute === 'role', 'attribute set' );
    ok( document.getElementById( a1.accessibleId ).getAttribute( 'role' ) === 'switch', 'HTML attribute set' );

    a1.removeAccessibleAttribute( 'role' );
    ok( !document.getElementById( a1.accessibleId ).getAttribute( 'role' ), 'attribute removed' );

    // set/get list item and description
    a1.descriptionTagName = 'ul';
    var itemId = a1.addDescriptionItem( TEST_DESCRIPTION );
    ok( document.getElementById( itemId ).textContent === TEST_DESCRIPTION, 'description item added' );

    a1.updateDescriptionItem( itemId, TEST_DESCRIPTION_2 );
    ok( document.getElementById( itemId ).textContent === TEST_DESCRIPTION_2, 'description item updated' );

  } );

  test( 'Accessibility input listeners', function() {

    // create a node
    var a1 = new scenery.Node( {
      tagName: 'button'
    } );
    var display = new scenery.Display( a1 );

    ok( a1.accessibleInputListeners.length === 0, 'no input accessible listeners on instantiation' );
    ok( a1.accessibleLabel === null, 'no label on instantiation' );

    // add a listener
    var listener = { click: function() { a1.accessibleLabel = TEST_LABEL } };
    a1.addAccessibleInputListener( listener );
    ok( a1.accessibleInputListeners.length === 1, 'accessible listener added' );

    // fire the event
    document.getElementById( a1.accessibleId ).click();
    ok( a1.accessibleLabel === TEST_LABEL, 'click fired, label set' );

    // remove the listener
    a1.removeAccessibleInputListener( listener );
    ok( a1.accessibleInputListeners.length === 0, 'accessible listener removed' );

    // make sure event listener was also removed from DOM element
    // click should not change the label
    a1.accessibleLabel = TEST_LABEL_2;
    document.getElementById( a1.accessibleId ).click();
    ok( a1.accessibleLabel === TEST_LABEL_2 );

  } );

  test( 'Next/Previous focusable', function() {
    var util = scenery.AccessibilityUtil;

    var rootNode = new scenery.Node( { tagName: 'div', focusable: true } );
    var display = new scenery.Display( rootNode );
    var rootElement = document.getElementById( rootNode.accessibleId );

    var a = new scenery.Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var b = new scenery.Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var c = new scenery.Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var d = new scenery.Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    var e = new scenery.Node( { tagName: 'div', focusable: true, focusHighlight: 'invisible' } );
    rootNode.children = [ a, b, c, d ];

    a.focus();
    ok( document.activeElement.id === a.accessibleId, 'a in focus (next)' );

    util.getNextFocusable( rootElement ).focus();
    ok( document.activeElement.id === b.accessibleId, 'b in focus (next)' );

    util.getNextFocusable( rootElement ).focus(); 
    ok( document.activeElement.id === c.accessibleId, 'c in focus (next)' );

    util.getNextFocusable( rootElement ).focus(); 
    ok( document.activeElement.id === d.accessibleId, 'd in focus (next)' );

    util.getNextFocusable( rootElement ).focus(); 
    ok( document.activeElement.id === d.accessibleId, 'd still in focus (next)' );

    util.getPreviousFocusable( rootElement ).focus();
    ok( document.activeElement.id === c.accessibleId, 'c in focus (previous)' );

    util.getPreviousFocusable( rootElement ).focus(); 
    ok( document.activeElement.id === b.accessibleId, 'b in focus (previous)' );

    util.getPreviousFocusable( rootElement ).focus(); 
    ok( document.activeElement.id === a.accessibleId, 'a in focus (previous)' );

    util.getPreviousFocusable( rootElement ).focus(); 
    ok( document.activeElement.id === a.accessibleId, 'a still in focus (previous)' );

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
    ok( document.activeElement.id === a.accessibleId, 'a only element focusable' );

  } );
  
})();
