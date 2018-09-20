// Copyright 2015, University of Colorado Boulder

/**
 * Some scripting for the prototype parallel DOM.  This mimics some dynamic content for the parallel DOM that would
 * occur in a simulation to test complex tab navigation and aria-live.
 *
 * @author Jesse Greenberg
 */


/**
 * Group behavior for accessibility.  On 'enter' or 'spacebar' enter the group by setting all child indices
 * to 0, removing the hidden attribute of the children, and set focus to the first child.
 *
 * @param {event} event
 * @param {domElement} parent
 */
function enterGroup( event, parent ) {
  'use strict';

  parent.hidden = false;
  parent.children[0].focus();
}

/**
 * Exit focus from the group.  'escape' should hide all elements in the group and set focus to the parent.
 * 'tab' should hide all elements and set focus to the next element in the accessibility tree.
 *
 * @param event
 * @param parent
 */
function exitGroup( event, parent ) { // eslint-disable-line no-unused-vars
  'use strict';
  
  if ( event.keyCode === 27 || event.keyCode === 9 ) {
    parent.hidden = true;
    parent.focus();
  }
}

/**
 * Focus the next sibling element when inside of a nested group.  Uses left and right arrow keys to navigate.  If
 * focus is already on first or last child, it loops around for continuity.
 * NOTE: I am unconvinced that looping behavior is a good thing.  My concern is that it will be disorienting.
 *
 * @param event
 * @param child
 */
function focusNextElement( event, child ) { // eslint-disable-line no-unused-vars
  'use strict';

  // isolate children, and the first and last children in the group
  var children = child.parentElement.children;
  var firstChild = children[ 0 ];
  var lastChild = children[ children.length - 1 ];

  if ( event.keyCode === 39 ) {
    //right arrow pressed
    var nextSibling = child.nextElementSibling;

    // if you are already at the last element, loop around and focus the first child
    if ( nextSibling ) {
      nextSibling.focus();
    }
    else {
      firstChild.focus();
    }
  }
  if ( event.keyCode === 37 ) {
    //left arrow pressed
    var previousSibling = child.previousElementSibling;

    // if you are already at the first element, loop around and focus the last child
    if ( previousSibling ) {
      previousSibling.focus();
    }
    else {
      lastChild.focus();
    }
  }
}

/**
 * Lightweight function that 'places' a puller on the first knot of a given side.
 * This essentially just changes the description of the puller and should trigger an aria-live event.
 *
 * NOTE: In Chrome + NVDA + Windows 7, it seems that the aria event is ONLY fired if the string changes in some way.
 * Otherwise, nothing is read.  A description string must be unique for it to be read.
 *
 * @param event
 * @param puller
 * @param knot
 */
function placePullerOnKnot( event, puller, knotGroup ) { // eslint-disable-line no-unused-vars
  'use strict';
  
  if ( event.keyCode === 13 ) {

    // pick up the object for drag and drop with the aria attribute
    puller.setAttribute( 'aria-grabbed', 'true');

    // notify the user that the element has been selected for drag and drop by updating the live region.
    var textNode = document.createTextNode( puller.innerText + 'selected for drag and drop.' );
    var targetNode = document.getElementById( 'ariaActionElement' );
    while ( targetNode.firstChild ) {
      targetNode.removeChild( targetNode.firstChild );
    }
    targetNode.appendChild( textNode );

    // =======================================
    // THE REST HERE IS BE SPECIFIC TO FAMB
    
    // Hide the puller in the list.
    puller.hidden = true;

    // unhide the knot group
    knotGroup.hidden = false;

    // unhid the aria-description for the group
    document.getElementById( knotGroup.getAttribute( 'aria-labelledby' ) ).hidden = false;

    // enter the group of knots
    enterGroup( event, knotGroup, knotGroup.children[0] );

  }
}

/**
 * Place a puller on the selected knot, searching through the document for the puller being dragged
 * with aria-grabbed
 */ 
function selectKnot( event, pullerGroup, knot ) { // eslint-disable-line no-unused-vars
  'use strict';

  if ( event.keyCode === 13 ) {

    // get the puller that is currently being dragged.
    var pullerChildren = pullerGroup.children;
    var grabbedChild;
    for( var i = 0; i < pullerChildren.length; i++ ) {
      if( pullerChildren[i].getAttribute( 'aria-grabbed' ) ) {
        grabbedChild = pullerChildren[i];
        break;
      }
    }
    assert && assert( grabbedChild, 'A puller must be grabbed in order to select a knot' );

    // drop the puller by setting the aria attribute to false
    console.log( 'dropping ' + grabbedChild );
    grabbedChild.setAttribute( 'aria-grabbed', 'false' );

    // the knot is now taken by a puller. hide it from other pullers.
    // knot.hidden = true;

  } 
}

