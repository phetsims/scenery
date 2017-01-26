// Copyright 2013-2016, University of Colorado Boulder

/**
 * Utility functions for scenery that are specifically useful for Accessibility.
 * These generally pertain to DOM traversal and manipulation.
 *
 * @author Jesse Greenberg
 */

define( function( require ) {
  'use strict';

  // modules
  var scenery = require( 'SCENERY/scenery' );

  // constants
  // specific HTML tag names, used for validation
  var INPUT_TAG = 'INPUT';
  var LABEL_TAG = 'LABEL';
  var UNORDERED_LIST_TAG = 'UL';
  var BUTTON_TAG = 'BUTTON';
  var TEXTAREA_TAG = 'TEXTAREA';
  var SELECT_TAG = 'SELECT';
  var OPTGROUP_TAG = 'OPTGROUP';
  var DATALIST_TAG = 'DATALIST';
  var OUTPUT_TAG = 'OUTPUT';
  var PARAGRAPH_TAG = 'P';

  // these elements are typically associated with forms, and support certain attributes
  var FORM_ELEMENTS = [ INPUT_TAG, BUTTON_TAG, TEXTAREA_TAG, SELECT_TAG, OPTGROUP_TAG, DATALIST_TAG, OUTPUT_TAG ];

  // these elements support inner text
  var ELEMENTS_WITH_INNER_TEXT = [ BUTTON_TAG, PARAGRAPH_TAG ];

  // these elements require a minimum width to be visible in Safari
  var ELEMENTS_THAT_NEED_WIDTH = [ 'INPUT' ];

  // flags for specifying direction of DOM traversal
  var NEXT = 'NEXT';
  var PREVIOUS = 'PREVIOUS';

  /**
   * Get all 'element' nodes off the parent element, placing them in an array for easy traversal.  Note that this
   * includes all elements, even those that are 'hidden' or purely for structure.
   *
   * @param  {HTMLElement} domElement - parent whose children will be linearized
   * @return {HTMLElement[]}
   * @private
   */
  function getLinearDOMElements( domElement ) {

    // gets ALL descendant children for the element
    var children = domElement.getElementsByTagName( '*' );

    var linearDOM = [];
    for ( var i = 0; i < children.length; i++ ) {

      // searching for the HTML elemetn nodes
      if ( children[ i ].nodeType === HTMLElement.ELEMENT_NODE ) {
        linearDOM[ i ] = ( children[ i ] );
      }
    }
    return linearDOM;
  }

  /**
   * Verify that an id is unique in the document, by searching through the id's of all HTML elements in the body.
   * 
   * @param  {string} id
   * @return {boolean}
   */
  function verifyUniqueId( id ) {
    var unique = true;

    // search through the elements
    var elements = getLinearDOMElements( document.body );
    for ( var i = 0; i < elements.length; i++ ) {
      if ( elements[ i ].id === id ) {
        unique = false;
        break;
      }
    }
    return unique;
  }

  /**
   * Determine if an element is hidden.  An element is considered 'hidden' if it (or any of its ancestors) has the
   * 'hidden' attribute.
   * @private
   *
   * @param {HTMLElement} domElement
   * @return {Boolean}
   */
  function isElementHidden( domElement ) {
    if ( domElement.hidden ) {
      return true;
    }
    else if ( domElement === document.body ) {
      return false;
    }
    else {
      return isElementHidden( domElement.parentElement );
    }
  }

  /**
   * Get the next or previous focusable element in the parallel DOM, relative to this Node's domElement
   * depending on the direction. Useful if you need to set focus dynamically or need to prevent default behavior
   * when focus changes. If no next or previous focusable is found, it returns the currently focused element.
   * This function should not be used directly, use getNextFocusable() or getPreviousFocusable() instead.
   * @private
   *
   * @param {string} direction - direction of traversal, one of 'NEXT' | 'PREVIOUS'
   * @param {HTMLElement} [parentElement] - optional, search will be limited to children of this element
   * @return {HTMLElement}
   */
  function getNextPreviousFocusable( direction, parentElement ) {

    // linearize the document [or the desired parent] for traversal
    var parent = parentElement || document.body;
    var linearDOM = AccessibilityUtil.getLinearDOMElements( parent );

    var activeElement = document.activeElement;
    var activeIndex = linearDOM.indexOf( activeElement );
    var delta = direction === NEXT ? +1 : -1;

    // find the next focusable element in the DOM
    var nextIndex = activeIndex + delta;
    var nextFocusable;
    while ( !nextFocusable && nextIndex < linearDOM.length && nextIndex >= 0 ) {
      var nextElement = linearDOM[ nextIndex ];
      nextIndex += delta;

      // continue to next element if this one is meant to be hidden
      if ( isElementHidden( nextElement ) ) {
        continue;
      }

      // if tabindex is greater than -1, the element is focusable so break
      if ( nextElement.tabIndex > -1 ) {
        nextFocusable = nextElement;
        break;
      }
    }

    // if no next focusable is found, return the active DOM element
    return nextFocusable || activeElement;
  }

  var AccessibilityUtil = {

    /**
     * Returns whether or not the attribute exists on the DOM element.
     * 
     * @param  {HTMLElement}  domElement
     * @param  {string}  attribute
     * @return {string|null}
     */
    hasAttribute: function( domElement, attribute ) {
      return !!domElement.getAttribute( attribute );
    },

    /**
     * Get all 'element' nodes off the parent element, placing them in an array for easy traversal.  Note that this
     * includes all elements, even those that are 'hidden' or purely for structure.
     *
     * @param  {HTMLElement} domElement - parent whose children will be linearized
     * @return {HTMLElement[]}
     * @private
     */
    getLinearDOMElements: function( domElement ) {

      // gets ALL descendant children for the element
      var children = domElement.getElementsByTagName( '*' );

      var linearDOM = [];
      for ( var i = 0; i < children.length; i++ ) {

        // searching for the HTML elemetn nodes
        if ( children[ i ].nodeType === HTMLElement.ELEMENT_NODE ) {
          linearDOM[ i ] = ( children[ i ] );
        }
      }
      return linearDOM;
    },

    /**
     * Get the next focusable element. This should very rarely be used.  The next focusable element can almost
     * always be focused automatically with 'Tab'.  However, if the 'Tab' key needs to be emulated this can be 
     * helpful. If no next focusable can be found, it will return the currently focused element.
     * @public
     *
     * @param{HTMLElement} [parentElement] - optional, search will be limited to elements under this element
     * @return {HTMLElement}
     */
    getNextFocusable: function( parentElement ) {
      return getNextPreviousFocusable( NEXT, parentElement );
    },

    /**
     * Get the previous focusable element. This should very rarely be used.  The previous focusable element can almost
     * always be found automatically with default 'Shift+Tab' behavior.  However, if the 'Tab' key needs to be emulated
     * this can be helpful.  If no next focusable can be found, it will return the currently focused element.
     * @public
     *
     * @param {HTMLElement} [parentElement] - optional, search will be limited to elements under this parent
     * @return {HTMLElement}
     */
    getPreviousFocusable: function( parentElement ) {
      return getNextPreviousFocusable( PREVIOUS, parentElement );
    },

    /**
     * Create a DOM element with the given tagname
     * @private
     * 
     * @param  {string} tagName
     */
    createDOMElement: function( tagName ) {
      return document.createElement( tagName );
    },

    /**
     * Generate a random unique id for a DOM element.  After generating
     * a unique id, we verify that the id is not shared with any other id in the
     * document. The return value is a string because the DOM API generally
     * handles id references by string.if
     * 
     * @return {string}
     */
    generateHTMLElementId: function() {
      var id;
      
      var isUnique = false;
      while( !isUnique ) {

        // try a random sequence of values
        id = Math.random().toString().substr( 2, 16 );
        isUnique = verifyUniqueId( id );
      }

      return id;
    },

    /**
     * Get a child element with an id.  This should only be used if the element has not been added to the document yet.
     * If the element is in the document, document.getElementById is a faster and more conventional option.
     *
     * @param  {HTMLElement} parentElement
     * @param  {string} childId
     * @return {HTMLElement}
     */
    getChildElementWithId: function( parentElement, childId ) {
      var childElement;
      var children = parentElement.children;

      for ( var i = 0; i < children.length; i++ ) {
        if ( children[ i ].id === childId ) {
          childElement = children[ i ];
          break;
        }
      }

      if ( !childElement ) {
        throw new Error( 'No child element under ' + parentElement + ' with id ' + childId );
      }

      return childElement;
    },

    /**
     * Returns whether or not the element supports inner text.
     * @private
     *
     * @return {boolean}
     */
    elementSupportsInnerText: function( domElement ) {
      return _.contains( ELEMENTS_WITH_INNER_TEXT, domElement.tagName );
    },

    // static tag names
    INPUT_TAG: INPUT_TAG,
    LABEL_TAG:LABEL_TAG,
    UNORDERED_LIST_TAG: UNORDERED_LIST_TAG,
    BUTTON_TAG: BUTTON_TAG,
    TEXTAREA_TAG: TEXTAREA_TAG,
    SELECT_TAG: SELECT_TAG,
    OPTGROUP_TAG: OPTGROUP_TAG,
    DATALIST_TAG: DATALIST_TAG,
    OUTPUT_TAG: OUTPUT_TAG,
    PARAGRAPH_TAG: PARAGRAPH_TAG,

    // groups of elements with special behavior
    FORM_ELEMENTS: FORM_ELEMENTS,
    ELEMENTS_WITH_INNER_TEXT: ELEMENTS_WITH_INNER_TEXT,
    ELEMENTS_THAT_NEED_WIDTH: ELEMENTS_THAT_NEED_WIDTH,

    // traversal flags
    NEXT: NEXT,
    PREVIOUS: PREVIOUS
  };

  scenery.register( 'AccessibilityUtil', AccessibilityUtil );

  return AccessibilityUtil;
} );
