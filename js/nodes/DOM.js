// Copyright 2013-2021, University of Colorado Boulder

/**
 * Displays a DOM element directly in a node, so that it can be positioned/transformed properly, and bounds are handled properly in Scenery.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import extendDefined from '../../../phet-core/js/extendDefined.js';
import DOMDrawable from '../display/drawables/DOMDrawable.js';
import Renderer from '../display/Renderer.js';
import scenery from '../scenery.js';
import Node from './Node.js';

const DOM_OPTION_KEYS = [
  'element', // {HTMLElement} - Sets the element, see setElement() for more documentation
  'preventTransform' // {boolean} - Sets whether Scenery is allowed to transform the element. see setPreventTransform() for docs
];

class DOM extends Node {
  /**
   * @public
   *
   * @param {Element|Object} element - The HTML element, or a jQuery selector result.
   * @param {Object} [options] - DOM-specific options are documented in DOM_OPTION_KEYS above, and can be provided
   *                             along-side options for Node
   */
  constructor( element, options ) {
    assert && assert( options === undefined || Object.getPrototypeOf( options ) === Object.prototype,
      'Extra prototype on Node options object is a code smell' );
    assert && assert( element instanceof window.Element || element.jquery,
      'DOM nodes need to be passed an HTML/DOM element or a jQuery selection like $( ... )' );

    // unwrap from jQuery if that is passed in, for consistency
    if ( element && element.jquery ) {
      element = element[ 0 ];
      assert && assert( element instanceof window.Element );
    }

    super();

    // @public (scenery-internal) {HTMLDivElement} - Container div that will have our main element as a child (so we can position and mutate it).
    this._container = document.createElement( 'div' );

    // @private {Object} - jQuery selection so that we can properly determine size information
    this._$container = $( this._container );
    this._$container.css( 'position', 'absolute' );
    this._$container.css( 'left', 0 );
    this._$container.css( 'top', 0 );

    // @private {boolean} - Flag that indicates whether we are updating/invalidating ourself due to changes to the DOM element. The flag is needed so
    //                      that updates to our element that we make in the update/invalidate section doesn't trigger an infinite loop with another
    //                      update.
    this.invalidateDOMLock = false;

    // @private {boolean} - Flag that when true won't let Scenery apply a transform directly (the client will take care of that).
    this._preventTransform = false;

    // Have mutate() call setElement() in the proper order
    options = extendDefined( {
      element: element
    }, options );

    // will set the element after initializing
    this.mutate( options );

    // Only renderer supported, no need to dynamically compute
    this.setRendererBitmask( Renderer.bitmaskDOM );
  }


  /**
   * Computes the bounds of our current DOM element (using jQuery, as replacing this with other things seems a bit
   * bug-prone and has caused issues in the past).
   * @protected
   *
   * The dom element needs to be attached to the DOM tree in order for this to work.
   *
   * Alternative getBoundingClientRect explored, but did not seem sufficient (possibly due to CSS transforms)?
   *
   * @returns {Bounds2}
   */
  calculateDOMBounds() {
    const $element = $( this._element );
    return new Bounds2( 0, 0, $element.width(), $element.height() );
  }

  /**
   * Triggers recomputation of our DOM element's bounds.
   * @public
   *
   * This should be called after the DOM element's bounds may have changed, to properly update the bounding box
   * in Scenery.
   */
  invalidateDOM() {
    // prevent this from being executed as a side-effect from inside one of its own calls
    if ( this.invalidateDOMLock ) {
      return;
    }
    this.invalidateDOMLock = true;

    // we will place ourselves in a temporary container to get our real desired bounds
    const temporaryContainer = document.createElement( 'div' );
    $( temporaryContainer ).css( {
      display: 'hidden',
      padding: '0 !important',
      margin: '0 !important',
      position: 'absolute',
      left: 0,
      top: 0,
      width: 65535,
      height: 65535
    } );

    // move to the temporary container
    this._container.removeChild( this._element );
    temporaryContainer.appendChild( this._element );
    document.body.appendChild( temporaryContainer );

    // bounds computation and resize our container to fit precisely
    const selfBounds = this.calculateDOMBounds();
    this.invalidateSelf( selfBounds );
    this._$container.width( selfBounds.getWidth() );
    this._$container.height( selfBounds.getHeight() );

    // move back to the main container
    document.body.removeChild( temporaryContainer );
    temporaryContainer.removeChild( this._element );
    this._container.appendChild( this._element );

    // unlock
    this.invalidateDOMLock = false;
  }

  /**
   * Creates a DOM drawable for this DOM node.
   * @public (scenery-internal)
   * @override
   *
   * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param {Instance} instance - Instance object that will be associated with the drawable
   * @returns {DOMSelfDrawable}
   */
  createDOMDrawable( renderer, instance ) {
    return DOMDrawable.createFromPool( renderer, instance );
  }

  /**
   * Whether this Node itself is painted (displays something itself).
   * @public
   * @override
   *
   * @returns {boolean}
   */
  isPainted() {
    // Always true for DOM nodes
    return true;
  }

  /**
   * Changes the DOM element of this DOM node to another element.
   * @public
   *
   * @param {HTMLElement} element
   * @returns {DOM} - For chaining
   */
  setElement( element ) {
    assert && assert( !this._element, 'We should only ever attach one DOMElement to a DOM node' );

    if ( this._element !== element ) {
      if ( this._element ) {
        this._container.removeChild( this._element );
      }

      this._element = element;

      this._container.appendChild( this._element );

      this.invalidateDOM();
    }

    return this; // allow chaining
  }

  set element( value ) { this.setElement( value ); }

  /**
   * Returns the DOM element being displayed by this DOM node.
   * @public
   *
   * @returns {HTMLElement}
   */
  getElement() {
    return this._element;
  }

  get element() { return this.getElement(); }

  /**
   * Sets the value of the preventTransform flag.
   * @public
   *
   * When the preventTransform flag is set to true, Scenery will not reposition (CSS transform) the DOM element, but
   * instead it will be at the upper-left (0,0) of the Scenery Display. The client will be responsible for sizing or
   * positioning this element instead.
   *
   * @param {boolean} preventTransform
   */
  setPreventTransform( preventTransform ) {
    assert && assert( typeof preventTransform === 'boolean' );

    if ( this._preventTransform !== preventTransform ) {
      this._preventTransform = preventTransform;
    }
  }

  set preventTransform( value ) { this.setPreventTransform( value ); }

  /**
   * Returns the value of the preventTransform flag.
   * @public
   *
   * See the setPreventTransform documentation for more information on the flag.
   *
   * @returns {boolean}
   */
  isTransformPrevented() {
    return this._preventTransform;
  }

  get preventTransform() { return this.isTransformPrevented(); }
}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 * @protected
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
DOM.prototype._mutatorKeys = DOM_OPTION_KEYS.concat( Node.prototype._mutatorKeys );

scenery.register( 'DOM', DOM );

export default DOM;