// Copyright 2017-2022, University of Colorado Boulder

/**
 * Utility functions for scenery that are specifically useful for ParallelDOM.
 * These generally pertain to DOM traversal and manipulation.
 *
 * For the most part this file's methods are public in a scenery-internal context. Some exceptions apply. Please
 * consult @jessegreenberg and/or @zepumph before using this outside of scenery.
 *
 * @author Jesse Greenberg
 */

import validate from '../../../../axon/js/validate.js';
import Validation from '../../../../axon/js/Validation.js';
import merge from '../../../../phet-core/js/merge.js';
import stripEmbeddingMarks from '../../../../phet-core/js/stripEmbeddingMarks.js';
import { PDOMSiblingStyle, scenery } from '../../imports.js';

// constants
const NEXT = 'NEXT';
const PREVIOUS = 'PREVIOUS';

// HTML tag names
const INPUT_TAG = 'INPUT';
const LABEL_TAG = 'LABEL';
const BUTTON_TAG = 'BUTTON';
const TEXTAREA_TAG = 'TEXTAREA';
const SELECT_TAG = 'SELECT';
const OPTGROUP_TAG = 'OPTGROUP';
const DATALIST_TAG = 'DATALIST';
const OUTPUT_TAG = 'OUTPUT';
const DIV_TAG = 'DIV';
const A_TAG = 'A';
const AREA_TAG = 'AREA';
const P_TAG = 'P';
const IFRAME_TAG = 'IFRAME';

// tag names with special behavior
const BOLD_TAG = 'B';
const STRONG_TAG = 'STRONG';
const I_TAG = 'I';
const EM_TAG = 'EM';
const MARK_TAG = 'MARK';
const SMALL_TAG = 'SMALL';
const DEL_TAG = 'DEL';
const INS_TAG = 'INS';
const SUB_TAG = 'SUB';
const SUP_TAG = 'SUP';
const BR_TAG = 'BR';

// These browser tags are a definition of default focusable elements, converted from Javascript types,
// see https://stackoverflow.com/questions/1599660/which-html-elements-can-receive-focus
const DEFAULT_FOCUSABLE_TAGS = [ A_TAG, AREA_TAG, INPUT_TAG, SELECT_TAG, TEXTAREA_TAG, BUTTON_TAG, IFRAME_TAG ];

// collection of tags that are used for formatting text
const FORMATTING_TAGS = [ BOLD_TAG, STRONG_TAG, I_TAG, EM_TAG, MARK_TAG, SMALL_TAG, DEL_TAG, INS_TAG, SUB_TAG,
  SUP_TAG, BR_TAG ];

// these elements do not have a closing tag, so they won't support features like innerHTML. This is how PhET treats
// these elements, not necessary what is legal html.
const ELEMENTS_WITHOUT_CLOSING_TAG = [ INPUT_TAG ];

// valid DOM events that the display adds listeners to. For a list of scenery events that support pdom features
// see Input.PDOM_EVENT_TYPES
// NOTE: Update BrowserEvents if this is added to
const DOM_EVENTS = [ 'focusin', 'focusout', 'input', 'change', 'click', 'keydown', 'keyup' ];

// DOM events that must have been triggered from user input of some kind, and will trigger the
// Display.userGestureEmitter. focus and blur events will trigger from scripting so they must be excluded.
const USER_GESTURE_EVENTS = [ 'input', 'change', 'click', 'keydown', 'keyup' ];

// A collection of DOM events which should be blocked from reaching the scenery Display div
// if they are targeted at an ancestor of the PDOM. Some screen readers try to send fake
// mouse/touch/pointer events to elements but for the purposes of Accessibility we only
// want to respond to DOM_EVENTS.
const BLOCKED_DOM_EVENTS = [

  // touch
  'touchstart',
  'touchend',
  'touchmove',
  'touchcancel',

  // mouse
  'mousedown',
  'mouseup',
  'mousemove',
  'mouseover',
  'mouseout',

  // pointer
  'pointerdown',
  'pointerup',
  'pointermove',
  'pointerover',
  'pointerout',
  'pointercancel',
  'gotpointercapture',
  'lostpointercapture'
];

const ARIA_LABELLEDBY = 'aria-labelledby';
const ARIA_DESCRIBEDBY = 'aria-describedby';
const ARIA_ACTIVE_DESCENDANT = 'aria-activedescendant';

// data attribute to flag whether an element is focusable - cannot check tabindex because IE11 and Edge assign
// tabIndex=0 internally for all HTML elements, including those that should not receive focus
const DATA_FOCUSABLE = 'data-focusable';

// data attribute which contains the unique ID of a Trail that allows us to find the PDOMPeer associated
// with a particular DOM element. This is used in several places in scenery accessibility, mostly PDOMPeer and Input.
const DATA_PDOM_UNIQUE_ID = 'data-unique-id';

// {Array.<String>} attributes that put an ID of another attribute as the value, see https://github.com/phetsims/scenery/issues/819
const ASSOCIATION_ATTRIBUTES = [ ARIA_LABELLEDBY, ARIA_DESCRIBEDBY, ARIA_ACTIVE_DESCENDANT ];

/**
 * Get all 'element' nodes off the parent element, placing them in an array for easy traversal.  Note that this
 * includes all elements, even those that are 'hidden' or purely for structure.
 *
 * @param  {HTMLElement} domElement - parent whose children will be linearized
 * @returns {HTMLElement[]}
 */
function getLinearDOMElements( domElement ) {

  // gets ALL descendant children for the element
  const children = domElement.getElementsByTagName( '*' );

  const linearDOM = [];
  for ( let i = 0; i < children.length; i++ ) {

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
 * Get the next or previous focusable element in the parallel DOM under the parent element and relative to the currently
 * focused element. Useful if you need to set focus dynamically or need to prevent default behavior
 * when focus changes. If no next or previous focusable is found, it returns the currently focused element.
 * This function should not be used directly, use getNextFocusable() or getPreviousFocusable() instead.
 *
 * @param {string} direction - direction of traversal, one of 'NEXT' | 'PREVIOUS'
 * @param {HTMLElement} [parentElement] - optional, search will be limited to children of this element
 * @returns {HTMLElement}
 */
function getNextPreviousFocusable( direction, parentElement ) {

  // linearize the document [or the desired parent] for traversal
  const parent = parentElement || document.body;
  const linearDOM = getLinearDOMElements( parent );

  const activeElement = document.activeElement;
  const activeIndex = linearDOM.indexOf( activeElement );
  const delta = direction === NEXT ? +1 : -1;

  // find the next focusable element in the DOM
  let nextIndex = activeIndex + delta;
  while ( nextIndex < linearDOM.length && nextIndex >= 0 ) {
    const nextElement = linearDOM[ nextIndex ];
    nextIndex += delta;

    if ( PDOMUtils.isElementFocusable( nextElement ) ) {
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

const PDOMUtils = {

  /**
   * Given a Property or string, return the Propergy value if it is a property. Otherwise just return the string.
   * Useful for forwarding the string to DOM content, but allowing the API to take a StringProperty. Eventually
   * PDOM may support dynamic strings.
   * @param valueOrProperty
   * @returns {string|Property}
   */
  unwrapStringProperty( valueOrProperty ) {
    const result = valueOrProperty === null ? null : ( typeof valueOrProperty === 'string' ? valueOrProperty : valueOrProperty.value );

    assert && assert( result === null || typeof result === 'string' );

    return result;
  },

  /**
   * Get the next focusable element relative to the currently focused element and under the parentElement.
   * Can be useful if you want to emulate the 'Tab' key behavior or just transition focus to the next element
   * in the document. If no next focusable can be found, it will return the currently focused element.
   * @public
   *
   * @param {HTMLElement} [parentElement] - optional, search will be limited to elements under this element
   * @returns {HTMLElement}
   */
  getNextFocusable( parentElement ) {
    return getNextPreviousFocusable( NEXT, parentElement );
  },

  /**
   * Get the previous focusable element relative to the currently focused element under the parentElement. Can be
   * useful if you want to emulate 'Shift+Tab' behavior. If no next focusable can be found, it will return the
   * currently focused element.
   * @public
   *
   * @param {HTMLElement} [parentElement] - optional, search will be limited to elements under this parent
   * @returns {HTMLElement}
   */
  getPreviousFocusable( parentElement ) {
    return getNextPreviousFocusable( PREVIOUS, parentElement );
  },

  /**
   * Get the first focusable element under the parentElement. If no element is available, the document.body is
   * returned.
   *
   * @param {HTMLElement} [parentElement] - optionally restrict the search to elements under this parent
   * @returns {HTMLElement}
   */
  getFirstFocusable( parentElement ) {
    const parent = parentElement || document.body;
    const linearDOM = getLinearDOMElements( parent );

    // return the document.body if no element is found
    let firstFocusable = document.body;

    let nextIndex = 0;
    while ( nextIndex < linearDOM.length ) {
      const nextElement = linearDOM[ nextIndex ];
      nextIndex++;

      if ( PDOMUtils.isElementFocusable( nextElement ) ) {
        firstFocusable = nextElement;
        break;
      }
    }

    return firstFocusable;
  },

  /**
   * Return a random focusable element in the document. Particularly useful for fuzz testing.
   * @public
   *
   * @parma {Random} random
   * @returns {HTMLElement}
   */
  getRandomFocusable( random ) {
    assert && assert( random, 'Random expected' );

    const linearDOM = getLinearDOMElements( document.body );
    const focusableElements = [];
    for ( let i = 0; i < linearDOM.length; i++ ) {
      PDOMUtils.isElementFocusable( linearDOM[ i ] ) && focusableElements.push( linearDOM[ i ] );
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
  containsFormattingTags( textContent ) {

    // no-op for null case
    if ( textContent === null ) {
      return false;
    }
    assert && assert( typeof textContent === 'string', 'unsupported type for textContent.' );

    let i = 0;
    const openIndices = [];
    const closeIndices = [];

    // find open/close tag pairs in the text content
    while ( i < textContent.length ) {
      const openIndex = textContent.indexOf( '<', i );
      const closeIndex = textContent.indexOf( '>', i );

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
    let onlyFormatting = true;
    const upperCaseContent = textContent.toUpperCase();
    for ( let j = 0; j < openIndices.length; j++ ) {

      // get the name and remove the closing slash
      let subString = upperCaseContent.substring( openIndices[ j ] + 1, closeIndices[ j ] );
      subString = subString.replace( '/', '' );

      // if the left of the substring contains space, it is not a valid tag so allow
      const trimmed = trimLeft( subString );
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
   * @param {string|number|null} textContent - domElement is cleared of content if null, could have acceptable HTML
   *                                    "formatting" tags in it
   */
  setTextContent( domElement, textContent ) {
    assert && assert( domElement instanceof Element ); // parent to HTMLElement, to support other namespaces
    assert && assert( textContent === null || typeof textContent === 'string' );

    if ( textContent === null ) {
      domElement.innerHTML = '';
    }
    else {

      // XHTML requires <br/> instead of <br>, but <br/> is still valid in HTML. See
      // https://github.com/phetsims/scenery/issues/1309
      const textWithoutBreaks = textContent.replaceAll( '<br>', '<br/>' );

      // TODO: this line must be removed to support i18n Interactive Description, see https://github.com/phetsims/chipper/issues/798
      const textWithoutEmbeddingMarks = stripEmbeddingMarks( textWithoutBreaks );

      // Disallow any unfilled template variables to be set in the PDOM.
      validate( textWithoutEmbeddingMarks, Validation.STRING_WITHOUT_TEMPLATE_VARS_VALIDATOR );

      if ( tagNameSupportsContent( domElement.tagName ) ) {

        // only returns true if content contains listed formatting tags
        if ( PDOMUtils.containsFormattingTags( textWithoutEmbeddingMarks ) ) {
          domElement.innerHTML = textWithoutEmbeddingMarks;
        }
        else {
          domElement.textContent = textWithoutEmbeddingMarks;
        }
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
  tagIsDefaultFocusable( tagName ) {
    return _.includes( DEFAULT_FOCUSABLE_TAGS, tagName.toUpperCase() );
  },

  /**
   * Returns true if the element is focusable. Assumes that all focusable  elements have tabIndex >= 0, which
   * is only true for elements of the Parallel DOM.
   *
   * @param {HTMLElement} domElement
   * @returns {boolean}
   */
  isElementFocusable( domElement ) {

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
  tagNameSupportsContent( tagName ) {
    return tagNameSupportsContent( tagName );
  },

  /**
   * Helper function to remove multiple HTMLElements from another HTMLElement
   * @public
   *
   * @param {HTMLElement} element
   * @param {Array.<HTMLElement>} childrenToRemove
   */
  removeElements( element, childrenToRemove ) {

    for ( let i = 0; i < childrenToRemove.length; i++ ) {
      const childToRemove = childrenToRemove[ i ];

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
  insertElements( element, childrenToAdd, beforeThisElement ) {
    assert && assert( element instanceof window.Element );
    assert && assert( Array.isArray( childrenToAdd ) );
    for ( let i = 0; i < childrenToAdd.length; i++ ) {
      const childToAdd = childrenToAdd[ i ];
      element.insertBefore( childToAdd, beforeThisElement || null );
    }
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
  createElement( tagName, focusable, options ) {
    options = merge( {
      // {string|null} - If non-null, the element will be created with the specific namespace
      namespace: null,

      // {string|null} - A string id that uniquely represents this element in the DOM, must be completely
      // unique in the DOM.
      id: null
    }, options );

    const domElement = options.namespace
                       ? document.createElementNS( options.namespace, tagName )
                       : document.createElement( tagName );

    if ( options.id ) {
      domElement.id = options.id;
    }

    // set tab index if we are overriding default browser behavior
    PDOMUtils.overrideFocusWithTabIndex( domElement, focusable );

    // gives this element styling from SceneryStyle
    domElement.classList.add( PDOMSiblingStyle.SIBLING_CLASS_NAME );

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
  overrideFocusWithTabIndex( element, focusable ) {
    const defaultFocusable = PDOMUtils.tagIsDefaultFocusable( element.tagName );

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
  USER_GESTURE_EVENTS: USER_GESTURE_EVENTS,
  BLOCKED_DOM_EVENTS: BLOCKED_DOM_EVENTS,

  DATA_PDOM_UNIQUE_ID: DATA_PDOM_UNIQUE_ID,
  PDOM_UNIQUE_ID_SEPARATOR: '-',

  // attribute used for elements which Scenery should not dispatch SceneryEvents when DOM event input is received on
  // them, see ParallelDOM.setExcludeLabelSiblingFromInput for more information
  DATA_EXCLUDE_FROM_INPUT: 'data-exclude-from-input'
};

scenery.register( 'PDOMUtils', PDOMUtils );

export default PDOMUtils;