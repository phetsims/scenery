// Copyright 2013-2016, University of Colorado Boulder
/* eslint-disable bad-sim-text */

/**
 * Utility functions for scenery that are specifically useful for Accessibility.
 * These generally pertain to DOM traversal and manipulation.
 *
 * For the most part this file's methods are public in a scenery-internal context. Some exceptions apply. Please
 * consult @jessegreenberg and/or @zepumph before using this outside of scenery.
 *
 * @author Jesse Greenberg
 */

define( function( require ) {
  'use strict';

  // modules
  var AccessibleSiblingStyle = require( 'SCENERY/accessibility/AccessibleSiblingStyle' );
  var Random = require( 'DOT/Random' );
  var scenery = require( 'SCENERY/scenery' );

  // constants
  var NEXT = 'NEXT';
  var PREVIOUS = 'PREVIOUS';

  // HTML tag names
  var INPUT_TAG = 'INPUT';
  var LABEL_TAG = 'LABEL';
  var BUTTON_TAG = 'BUTTON';
  var TEXTAREA_TAG = 'TEXTAREA';
  var SELECT_TAG = 'SELECT';
  var OPTGROUP_TAG = 'OPTGROUP';
  var DATALIST_TAG = 'DATALIST';
  var OUTPUT_TAG = 'OUTPUT';
  var DIV_TAG = 'DIV';
  var A_TAG = 'A';
  var AREA_TAG = 'AREA';
  var P_TAG = 'P';
  var IFRAME_TAG = 'IFRAME';

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
  var BR_TAG = 'BR';

  // These browser tags are a definition of default focusable elements, converted from Javascript types,
  // see https://stackoverflow.com/questions/1599660/which-html-elements-can-receive-focus
  var DEFAULT_FOCUSABLE_TAGS = [ A_TAG, AREA_TAG, INPUT_TAG, SELECT_TAG, TEXTAREA_TAG, BUTTON_TAG, IFRAME_TAG ];

  // collection of tags that are used for formatting text
  var FORMATTING_TAGS = [ BOLD_TAG, STRONG_TAG, I_TAG, EM_TAG, MARK_TAG, SMALL_TAG, DEL_TAG, INS_TAG, SUB_TAG,
    SUP_TAG, BR_TAG ];

  // these elements do not have a closing tag, so they won't support features like innerHTML. This is how PhET treats
  // these elements, not necessary what is legal html.
  var ELEMENTS_WITHOUT_CLOSING_TAG = [ INPUT_TAG ];

  // valid DOM events that the display adds listeners to. For a list of scenery events that support a11y features
  // see Input.A11Y_EVENT_TYPES
  var DOM_EVENTS = [ 'focusin', 'focusout', 'input', 'change', 'click', 'keydown', 'keyup' ];

  var ARIA_LABELLEDBY = 'aria-labelledby';
  var ARIA_DESCRIBEDBY = 'aria-describedby';
  var ARIA_ACTIVE_DESCENDANT = 'aria-activedescendant';

  // data attribute to flag whether an element is focusable - cannot check tabindex because IE11 and Edge assign
  // tabIndex=0 internally for all HTML elements, including those that should not receive focus
  var DATA_FOCUSABLE = 'data-focusable';

  // data attribute which contains the unique ID of a Trail that allows us to find the AccessiblePeer associated
  // with a particular DOM element. This is used in several places in scenery accessibility, mostly AccessiblePeer.
  var DATA_TRAIL_ID = 'data-trail-id';

  // {Array.<String>} attributes that put an ID of another attribute as the value, see https://github.com/phetsims/scenery/issues/819
  var ASSOCIATION_ATTRIBUTES = [ ARIA_LABELLEDBY, ARIA_DESCRIBEDBY, ARIA_ACTIVE_DESCENDANT ];

  /**
   * Get all 'element' nodes off the parent element, placing them in an array for easy traversal.  Note that this
   * includes all elements, even those that are 'hidden' or purely for structure.
   *
   * @param  {HTMLElement} domElement - parent whose children will be linearized
   * @returns {HTMLElement[]}
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
   * Get the next or previous focusable element in the parallel DOM, relative to the parent element passed in and
   * depending on the direction. Useful if you need to set focus dynamically or need to prevent default behavior
   * when focus changes. If no next or previous focusable is found, it returns the currently focused element.
   * This function should not be used directly, use getNextFocusable() or getPreviousFocusable() instead.
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
    while ( nextIndex < linearDOM.length && nextIndex >= 0 ) {
      var nextElement = linearDOM[ nextIndex ];
      nextIndex += delta;

      if ( AccessibilityUtil.isElementFocusable( nextElement ) ) {
        return nextElement;
      }
    }

    // if no next focusable is found, return the active DOM element
    return activeElement;
  }

  /**
   * Trims the white space from the left of the string.
   * Solution from https://stackoverflow.com/questions/1593859/left-trim-in-javascript
   * @param  {string} string
   * @returns {string}
   */
  function trimLeft( string ) {

    // ^ - from the beginning of the string
    // \s - whitespace character
    // + - greedy
    return string.replace( /^\s+/, '' );
  }


  /**
   * Returns whether or not the tagName supports innerHTML or textContent in PhET.
   * @private
   * @param {string} tagName
   * @returns {boolean}
   */
  function tagNameSupportsContent( tagName ) {
    return !_.includes( ELEMENTS_WITHOUT_CLOSING_TAG, tagName.toUpperCase() );
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
     * Return a random focusable element in the document. Particularly useful for fuzz testing.
     * @public
     *
     * @parma {Random} [random]
     * @returns {HTMLElement}
     */
    getRandomFocusable: function( random ) {

      random = random || new Random();

      var linearDOM = getLinearDOMElements( document.body );
      var focusableElements = [];
      for ( var i = 0; i < linearDOM.length; i++ ) {
        AccessibilityUtil.isElementFocusable( linearDOM[ i ] ) && focusableElements.push( linearDOM[ i ] );
      }

      return focusableElements[ random.nextInt( focusableElements.length ) ];
    },

    /**
     * If the textContent has any tags that are not formatting tags, return false. Only checking for
     * tags that are not in the whitelist FORMATTING_TAGS. If there are no tags at all, return false.
     * @public
     *
     * @param {string} textContent
     * @returns {boolean}
     */
    containsFormattingTags: function( textContent ) {

      // no-op for null case
      if ( textContent === null ) {
        return false;
      }
      assert && assert( typeof textContent === 'string', 'unsupported type for textContent.' );

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

      // malformed tags or no tags at all, return false immediately
      if ( openIndices.length !== closeIndices.length || openIndices.length === 0 ) {
        return false;
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
    },

    /**
     * If the text content uses formatting tags, set the content as innerHTML. Otherwise, set as textContent.
     * In general, textContent is more secure and much faster because it doesn't trigger DOM styling and
     * element insertions.
     * @public
     *
     * @param {Element} domElement
     * @param {string} textContent - could have acceptable HTML "formatting" tags in it
     */
    setTextContent: function( domElement, textContent ) {
      assert && assert( domElement instanceof Element ); // parent to HTMLElement, to support other namespaces
      assert && assert( typeof textContent === 'string' );
      if ( tagNameSupportsContent( domElement.tagName ) ) {

        // only returns true if content contains listed formatting tags
        if ( AccessibilityUtil.containsFormattingTags( textContent ) ) {
          domElement.innerHTML = textContent;
        }
        else {
          domElement.textContent = textContent;
        }
      }
    },

    /**
     * Given a tagName, test if the element will be focuable by default by the browser.
     * Different from isElementFocusable, because this only looks at tags that the browser will automatically put
     * a >=0 tab index on.
     * @public
     *
     * NOTE: Uses a set of browser types as the definition of default focusable elements,
     * see https://stackoverflow.com/questions/1599660/which-html-elements-can-receive-focus
     *
     * @param tagName
     * @returns {boolean}
     */
    tagIsDefaultFocusable: function( tagName ) {
      return _.includes( DEFAULT_FOCUSABLE_TAGS, tagName.toUpperCase() );
    },

    /**
     * Returns true if the element is focusable. Assumes that all focusable  elements have tabIndex >= 0, which
     * is only true for elements of the Parallel DOM.
     *
     * @param {HTMLElement} domElement
     * @returns {boolean}
     */
    isElementFocusable: function( domElement ) {

      if ( !document.body.contains( domElement ) ) {
        return false;
      }

      // continue to next element if this one is meant to be hidden
      if ( isElementHidden( domElement ) ) {
        return false;
      }

      // if element is for formatting, skipe over it - required since IE gives these tabindex="0"
      if ( _.includes( FORMATTING_TAGS, domElement.tagName ) ) {
        return false;
      }

      return domElement.getAttribute( DATA_FOCUSABLE ) === 'true';
    },

    /**
     * @public
     *
     * @param {string} tagName
     * @returns {boolean} - true if the tag does support inner content
     */
    tagNameSupportsContent: function( tagName ) {
      return tagNameSupportsContent( tagName );
    },

    /**
     * Helper function to remove multiple HTMLElements from another HTMLElement
     * @public
     *
     * @param {HTMLElement} element
     * @param {Array.<HTMLElement>} childrenToRemove
     */
    removeElements: function( element, childrenToRemove ) {

      for ( var i = 0; i < childrenToRemove.length; i++ ) {
        var childToRemove = childrenToRemove[ i ];

        assert && assert( element.contains( childToRemove ), 'element does not contain child to be removed: ', childToRemove );

        element.removeChild( childToRemove );
      }

    },

    /**
     * Helper function to add multiple elements as children to a parent
     * @public
     *
     * @param {HTMLElement} element - to add children to
     * @param {Array.<HTMLElement>} childrenToAdd
     * @param {HTMLElement} [beforeThisElement] - if not supplied, the insertBefore call will just use 'null'
     */
    insertElements: function( element, childrenToAdd, beforeThisElement ) {

      for ( var i = 0; i < childrenToAdd.length; i++ ) {
        var childToAdd = childrenToAdd[ i ];
        element.insertBefore( childToAdd, beforeThisElement || null );
      }
    },

    /**
     * Given an associationObject for either aria-labelledby or aria-describedby, make sure it has the right signature.
     * @public
     *
     * @param {Object} associationObject
     */
    validateAssociationObject: function( associationObject ) {
      assert && assert( typeof associationObject === 'object' );

      var expectedKeys = [ 'thisElementName', 'otherNode', 'otherElementName' ];

      var objectKeys = Object.keys( associationObject );

      assert && assert( objectKeys.length === 3, 'wrong number of keys in associationObject, expected:', expectedKeys, ' got:', objectKeys );


      for ( var i = 0; i < objectKeys.length; i++ ) {
        var objectKey = objectKeys[ i ];
        assert && assert( expectedKeys.indexOf( objectKey ) >= 0, 'unexpected key: ' + objectKey );
      }

      assert && assert( associationObject.otherNode instanceof scenery.Node );
      assert && assert( typeof associationObject.thisElementName === 'string' );
      assert && assert( typeof associationObject.otherElementName === 'string' );
    },

    /**
     * Create an HTML element.  Unless this is a form element or explicitly marked as focusable, add a negative
     * tab index. IE gives all elements a tabIndex of 0 and handles tab navigation internally, so this marks
     * which elements should not be in the focus order.
     *
     * @public
     * @param  {string} tagName
     * @param {boolean} focusable - should the element be explicitly added to the focus order?
     * @param {Object} [options]
     * @returns {HTMLElement}
     */
    createElement: function( tagName, focusable, options ) {
      options = _.extend( {
        // {string|null} - If non-null, the element will be created with the specific namespace
        namespace: null,

        // {string|null} - A string id that uniquely represents this element in the DOM, must be completely
        // unique in the DOM.
        id: null,

        // {string|null} - A string id from Trail.getUnqiqueId pointing to the node that is being
        // represented by this element in the PDOM. Will by used to dispatch events received by this
        // DOM element to the scenery Node being represented. Should be unique to the AccessibleInstance
        // but each sibling for an AccessiblePeer should have the same trailId.
        trailId: null
      }, options );

      var domElement = options.namespace
                       ? document.createElementNS( options.namespace, tagName )
                       : document.createElement( tagName );

      if ( options.trailId ) {

        // NOTE: dataset isn't supported by all namespaces (like MathML) so we need to use setAttribute
        domElement.setAttribute( AccessibilityUtil.DATA_TRAIL_ID, options.trailId );
      }
      if ( options.id ) {
        domElement.id = options.id;
      }

      // set tab index if we are overriding default browser behavior
      AccessibilityUtil.overrideFocusWithTabIndex( domElement, focusable );

      // gives this element styling from SceneryStyle
      domElement.className = AccessibleSiblingStyle.SIBLING_CLASS_NAME;

      return domElement;
    },

    /**
     * Add a tab index to an element when overriding the default focus behavior for the element. Adding tabindex
     * to an element can only be done when overriding the default browser behavior because tabindex interferes with
     * the way JAWS reads through content on Chrome, see https://github.com/phetsims/scenery/issues/893
     *
     * If default behavior and focusable align, the tabindex attribute is removed so that can't interfere with a
     * screen reader.
     * @public (scenery-internal)
     *
     * @param {HTMLElement} element
     * @param {boolean} focusable
     */
    overrideFocusWithTabIndex: function( element, focusable ) {
      var defaultFocusable = AccessibilityUtil.tagIsDefaultFocusable( element.tagName );

      // only add a tabindex when we are overriding the default focusable bahvior of the browser for the tag name
      if ( defaultFocusable !== focusable ) {
        element.tabIndex = focusable ? 0 : -1;
      }
      else {
        element.removeAttribute( 'tabindex' );
      }

      element.setAttribute( DATA_FOCUSABLE, focusable );
    },

    TAGS: {
      INPUT: INPUT_TAG,
      LABEL: LABEL_TAG,
      BUTTON: BUTTON_TAG,
      TEXTAREA: TEXTAREA_TAG,
      SELECT: SELECT_TAG,
      OPTGROUP: OPTGROUP_TAG,
      DATALIST: DATALIST_TAG,
      OUTPUT: OUTPUT_TAG,
      DIV: DIV_TAG,
      A: A_TAG,
      P: P_TAG,
      B: BOLD_TAG,
      STRONG: STRONG_TAG,
      I: I_TAG,
      EM: EM_TAG,
      MARK: MARK_TAG,
      SMALL: SMALL_TAG,
      DEL: DEL_TAG,
      INS: INS_TAG,
      SUB: SUB_TAG,
      SUP: SUP_TAG
    },

    // these elements are typically associated with forms, and support certain attributes
    FORM_ELEMENTS: [ INPUT_TAG, BUTTON_TAG, TEXTAREA_TAG, SELECT_TAG, OPTGROUP_TAG, DATALIST_TAG, OUTPUT_TAG, A_TAG ],

    // default tags for html elements of the Node.
    DEFAULT_CONTAINER_TAG_NAME: DIV_TAG,
    DEFAULT_DESCRIPTION_TAG_NAME: P_TAG,
    DEFAULT_LABEL_TAG_NAME: P_TAG,

    ASSOCIATION_ATTRIBUTES: ASSOCIATION_ATTRIBUTES,

    // valid input types that support the "checked" property/attribute for input elements
    INPUT_TYPES_THAT_SUPPORT_CHECKED: [ 'RADIO', 'CHECKBOX' ],

    DOM_EVENTS: DOM_EVENTS,

    DATA_TRAIL_ID: DATA_TRAIL_ID
  };

  scenery.register( 'AccessibilityUtil', AccessibilityUtil );

  return AccessibilityUtil;
} );
