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
  var NEXT = 'NEXT';
  var PREVIOUS = 'PREVIOUS';

  // tag names with special behavior
  var BOLD_TAG = 'B';
  var STRONG_TAG = 'STRONG';
  var I_TAG = 'I';
  var EM_TAG = 'EM';
  var MARK_TAG = 'MARK';
  var SMALL_TAG = 'SMALL';
  var DEL_TAG = 'DEL';
  var INS_TAG = 'INS';
  var SUB_TAG = 'SUB';
  var SUP_TAG = 'SUP';

  // collection of tags that are used for formatting text
  var FORMATTING_TAGS = [ BOLD_TAG, STRONG_TAG, I_TAG, EM_TAG, MARK_TAG, SMALL_TAG, DEL_TAG, INS_TAG, SUB_TAG, SUP_TAG ];

  /**
   * Get all 'element' nodes off the parent element, placing them in an array for easy traversal.  Note that this
   * includes all elements, even those that are 'hidden' or purely for structure.
   *
   * @param  {HTMLElement} domElement - parent whose children will be linearized
   * @returns {HTMLElement[]}
   * @private
   */
  function getLinearDOMElements( domElement ) {

    // gets ALL descendant children for the element
    var children = domElement.getElementsByTagName( '*' );

    var linearDOM = [];
    for ( var i = 0; i < children.length; i++ ) {

      // searching for the HTML element nodes (NOT Scenery nodes)
      if ( children[ i ].nodeType === Node.ELEMENT_NODE ) {
        linearDOM[ i ] = ( children[ i ] );
      }
    }
    return linearDOM;
  }

  /**
   * Determine if an element is hidden.  An element is considered 'hidden' if it (or any of its ancestors) has the
   * 'hidden' attribute.
   * @private
   *
   * @param {HTMLElement} domElement
   * @returns {Boolean}
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
   * @returns {HTMLElement}
   */
  function getNextPreviousFocusable( direction, parentElement ) {

    // linearize the document [or the desired parent] for traversal
    var parent = parentElement || document.body;
    var linearDOM = getLinearDOMElements( parent );

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

      // if element is for formatting, skipe over it - required since IE gives these tabindex="0" 
      if ( _.includes( FORMATTING_TAGS, nextElement.tagName ) ) {
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

  /**
   * Trims the white space from the left of the string.
   * Solution from https://stackoverflow.com/questions/1593859/left-trim-in-javascript
   * @param  {string} string
   * @return {string}
   */
  function trimLeft( string ) {

    // ^ - from the beginning of the string
    // \s - whitespace character
    // + - greedy
    return string.replace(/^\s+/,'');
  }

  var AccessibilityUtil = {

    /**
     * Get the next focusable element. This should very rarely be used.  The next focusable element can almost
     * always be focused automatically with 'Tab'.  However, if the 'Tab' key needs to be emulated this can be 
     * helpful. If no next focusable can be found, it will return the currently focused element.
     * @public
     *
     * @param{HTMLElement} [parentElement] - optional, search will be limited to elements under this element
     * @returns {HTMLElement}
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
     * @returns {HTMLElement}
     */
    getPreviousFocusable: function( parentElement ) {
      return getNextPreviousFocusable( PREVIOUS, parentElement );
    },

    /**
     * If the textContent has any tags that are not formatting tags, return false. Only checking for
     * tags that are not in the whitelist FORMATTING_TAGS, if there are no tags this will still return
     * true.
     * @public
     * 
     * @param  {string} textContent
     * @returns {boolean}
     */
    usesFormattingTagsExclusive: function( textContent ) {

      var i = 0;
      var openIndices = [];
      var closeIndices = [];

      // find open/close tag pairs in the text content
      while ( i < textContent.length ) {
        var openIndex = textContent.indexOf( '<', i );
        var closeIndex = textContent.indexOf( '>', i );

        if ( openIndex > -1 ) {
          openIndices.push( openIndex );
          i = openIndex + 1;
        }
        if ( closeIndex > -1 ) {
          closeIndices.push( closeIndex );
          i = closeIndex + 1;
        }
        else {
          i++;
        }
      }

      // check the name in between the open and close brackets - if anything other than formatting tags, return false
      var onlyFormatting = true;
      var upperCaseContent = textContent.toUpperCase();
      for ( var j = 0; j < openIndices.length; j++ ) {

        // get the name and remove the closing slash
        var subString = upperCaseContent.substring( openIndices[ j ] + 1, closeIndices[ j ] );
        subString = subString.replace( '/', '' );

        // if the left of the substring contains space, it is not a valid tag so allow
        var trimmed = trimLeft( subString );
        if ( subString.length - trimmed.length > 0 ) {
          continue;
        }

        if ( !_.includes( FORMATTING_TAGS, subString ) ) {
          onlyFormatting = false;
        }
      }

      return onlyFormatting;
    }
  };

  scenery.register( 'AccessibilityUtil', AccessibilityUtil );

  return AccessibilityUtil;
} );
