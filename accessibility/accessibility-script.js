// Copyright 2015, University of Colorado Boulder
'use strict';

/**
 * Some scripting for the the prototype parallel DOM.  This mimics some dynamic content for the parallel DOM that would
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
function enterGroup( event, parent, firstChild ) {
  // go through all children of the parent we are entering
  if ( event.keyCode === 13 || event.keyCode === 32 ) {
    for ( var i = 0; i < parent.children.length; i++ ) {
      // NOTE: WE ARE NOT CHANGING THE NAVIGATION ORDER FOR THIS PROTOTYPE!
      // THE HIDDEN ATTRIBUTE IS NOW COVERING THIS FOR US!
      parent.children[ i ].hidden = false;
    }
    // set focus to the first child
    firstChild.focus();
  }
}

/**
 * Exit focus from the group.  'escape' should hide all elements in the group and set focus to the parent.
 * 'tab' should hide all elements and set focus to the next element in the accessibility tree.
 *
 * @param event
 * @param parent
 */
function exitGroup( event, parent ) {
  if ( event.keyCode === 27 || event.keyCode === 9 ) {
    for ( var i = 0; i < parent.children.length; i++ ) {
      // hide the children!
      parent.children[ i ].hidden = true;
    }
    // set focus back to the parent?
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
function focusNextElement( event, child ) {

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
 * NOTE: There seems to be no
 *
 * @param event
 * @param puller
 * @param knot
 */
var global = 0;
function placePullerOnKnot( event, puller, knot ) {
  if ( event.keyCode === 13 ) {
    var textNode = document.createTextNode( puller.alt + 'selected for drag and drop.' );
    var targetNode = document.getElementById( 'ariaActionElement' );
    while ( targetNode.firstChild ) {
      targetNode.removeChild( targetNode.firstChild );
    }
    targetNode.appendChild( textNode );
    global++;
    if ( global > 5 ) {
      global = 0;
    }
  }
}

