// Copyright 2013-2015, University of Colorado Boulder

/**
 * Displays a DOM element directly in a node, so that it can be positioned/transformed properly, and bounds are handled properly in Scenery.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var escapeHTML = require( 'PHET_CORE/escapeHTML' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Node = require( 'SCENERY/nodes/Node' ); // DOM inherits from Node
  var Renderer = require( 'SCENERY/display/Renderer' );
  var DOMDrawable = require( 'SCENERY/display/drawables/DOMDrawable' );

  /**
   * @constructor
   *
   * @param {Element|Object} element - The HTML element, or a jQuery selector result.
   * @param {Object} [options] - Node and DOM options elements, see Node for details.
   */
  function DOM( element, options ) {
    assert && assert( element instanceof window.Element || element.jquery,
      'DOM nodes need to be passed an HTML/DOM element or a jQuery selection like $( ... )' );

    options = options || {};

    // unwrap from jQuery if that is passed in, for consistency
    if ( element && element.jquery ) {
      element = element[ 0 ];
      assert && assert( element instanceof window.Element );
    }

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

    // We'll have mutate() call setElement() in the proper order
    options.element = element;

    // will set the element after initializing
    Node.call( this, options );

    // Only renderer supported, no need to dynamically compute
    this.setRendererBitmask( Renderer.bitmaskDOM );
  }

  scenery.register( 'DOM', DOM );

  inherit( Node, DOM, {
    /**
     * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
     * order they will be evaluated in.
     * @protected
     *
     * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
     *       cases that may apply.
     */
    _mutatorKeys: [ 'element', 'preventTransform' ].concat( Node.prototype._mutatorKeys ),

    // needs to be attached to the DOM tree for this to work
    calculateDOMBounds: function() {
      // var boundingRect = this._element.getBoundingClientRect();
      // return new Bounds2( 0, 0, boundingRect.width, boundingRect.height );
      var $element = $( this._element );
      return new Bounds2( 0, 0, $element.width(), $element.height() );
    },

    createTemporaryContainer: function() {
      var temporaryContainer = document.createElement( 'div' );
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
      return temporaryContainer;
    },

    invalidateDOM: function() {
      // prevent this from being executed as a side-effect from inside one of its own calls
      if ( this.invalidateDOMLock ) {
        return;
      }
      this.invalidateDOMLock = true;

      // we will place ourselves in a temporary container to get our real desired bounds
      var temporaryContainer = this.createTemporaryContainer();

      // move to the temporary container
      this._container.removeChild( this._element );
      temporaryContainer.appendChild( this._element );
      document.body.appendChild( temporaryContainer );

      // bounds computation and resize our container to fit precisely
      var selfBounds = this.calculateDOMBounds();
      this.invalidateSelf( selfBounds );
      this._$container.width( selfBounds.getWidth() );
      this._$container.height( selfBounds.getHeight() );

      // move back to the main container
      document.body.removeChild( temporaryContainer );
      temporaryContainer.removeChild( this._element );
      this._container.appendChild( this._element );

      this.invalidateDOMLock = false;
    },

    getDOMElement: function() {
      return this._container;
    },

    /**
     * Creates a DOM drawable for this DOM node.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @param {Instance} instance - Instance object that will be associated with the drawable
     * @returns {DOMSelfDrawable}
     */
    createDOMDrawable: function( renderer, instance ) {
      return DOMDrawable.createFromPool( renderer, instance );
    },

    /**
     * Whether this Node itself is painted (displays something itself).
     * @public
     * @override
     *
     * @returns {boolean}
     */
    isPainted: function() {
      // Always true for DOM nodes
      return true;
    },

    setElement: function( element ) {
      assert && assert( !this._element, 'We should only ever attach one DOMElement to a DOM node' );

      if ( this._element !== element ) {
        if ( this._element ) {
          this._container.removeChild( this._element );
        }

        this._element = element;

        this._container.appendChild( this._element );

        // TODO: bounds issue, since this will probably set to empty bounds and thus a repaint may not draw over it
        this.invalidateDOM();
      }

      return this; // allow chaining
    },
    set element( value ) { this.setElement( value ); },

    getElement: function() {
      return this._element;
    },
    get element() { return this.getElement(); },

    setPreventTransform: function( preventTransform ) {
      assert && assert( typeof preventTransform === 'boolean' );

      if ( this._preventTransform !== preventTransform ) {
        this._preventTransform = preventTransform;

        // TODO: anything needed here?
      }
    },
    set preventTransform( value ) { this.setPreventTransform( value ); },

    isTransformPrevented: function() {
      return this._preventTransform;
    },
    get preventTransform() { return this.isTransformPrevented(); },

    /**
     * Returns a string containing constructor information for Node.string().
     * @protected
     * @override
     *
     * @param {string} propLines - A string representing the options properties that need to be set.
     * @returns {string}
     */
    getBasicConstructor: function( propLines ) {
      return 'new scenery.DOM( $( \'' + escapeHTML( this._container.innerHTML.replace( /'/g, '\\\'' ) ) + '\' ), {' + propLines + '} )';
    },

    /**
     * Returns the property object string for use with toString().
     * @protected (scenery-internal)
     * @override
     *
     * @param {string} spaces - Whitespace to add
     * @param {boolean} [includeChildren]
     */
    getPropString: function( spaces, includeChildren ) {
      return Node.prototype.getPropString.call( this, spaces, includeChildren );
    }
  } );

  return DOM;
} );
