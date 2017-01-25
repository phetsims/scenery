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

  // flags for specifying direction of DOM traversal
  var NEXT = 'NEXT';
  var PREVIOUS = 'PREVIOUS';

  /**
   * Get all 'element' nodes off the parent element, placing them in an
   * array for easy traversal.  Note that this includes all elements, even
   * those that are 'hidden' or purely for structure.
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
   * Verify that an id is unique in the document, by searching through the
   * id's of all HTML elements.
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

    // traversal flags
    NEXT: NEXT,
    PREVIOUS: PREVIOUS
  };

  scenery.register( 'AccessibilityUtil', AccessibilityUtil );

  return AccessibilityUtil;
} );
