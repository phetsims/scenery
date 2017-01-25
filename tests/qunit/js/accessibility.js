// Copyright 2002-2016, University of Colorado Boulder

(function() {
  'use strict';

  module( 'Scenery: Accessibility' );

  var TEST_LABEL = 'Test label';
  var TEST_LABEL_2 = 'Test label 2';
  var TEST_DESCRIPTION = 'Test decsription';
  var TEST_TITLE = 'Test title';

  test( 'Accessibility options', function() {

    var rootNode = new scenery.Node();

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

    var accessibleNode = new scenery.Node( {
      tagName: 'div',
      useAriaLabel: true, // use ARIA label attribute
      accessibleHidden: true, // hidden from screen readers (and browser)
      accessibleLabel: TEST_LABEL,
      labelTagName: 'p',
      descriptionTagName: 'p',
      accessibleDescription: TEST_DESCRIPTION,
      ariaLabelledById: buttonNode.accessibleId, // ARIA label relation
      ariaDescribedById: buttonNode.accessibleId // ARIA description relation
    } );

    // accessible instances are not sorted until added to a display
    var display = new scenery.Display( rootNode );
    rootNode.children = [ buttonNode, accessibleNode ];

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
    ok( accessibleNode.ariaLabelledById === buttonNode.accessibleId, 'aria-labelledby id' );
    ok( accessibleNode.ariaDescribedById === buttonNode.accessibleId, ' aria-describedby id' );


    // verify DOM structure - options above should create
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
    ok( buttonPeers[ 0  ].tagName === 'DIV', 'label first for prependLabels' );
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
  
})();
