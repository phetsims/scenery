// Copyright 2002-2013, University of Colorado

/**
 * DOM nodes. Currently lightweight handling
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var escapeHTML = require( 'PHET_CORE/escapeHTML' );
  var Bounds2 = require( 'DOT/Bounds2' );

  var scenery = require( 'SCENERY/scenery' );

  var Node = require( 'SCENERY/nodes/Node' ); // DOM inherits from Node
  require( 'SCENERY/layers/Renderer' );
  require( 'SCENERY/util/Util' );

  scenery.DOM = function DOM( element, options ) {
    options = options || {};

    this._interactive = false;

    // unwrap from jQuery if that is passed in, for consistency
    if ( element && element.jquery ) {
      element = element[0];
    }

    this._container = document.createElement( 'div' );
    this._$container = $( this._container );
    this._$container.css( 'position', 'absolute' );
    this._$container.css( 'left', 0 );
    this._$container.css( 'top', 0 );

    this.invalidateDOMLock = false;

    // so that the mutator will call setElement()
    options.element = element;

    // will set the element after initializing
    Node.call( this, options );
    this.setRendererBitmask( scenery.bitmaskSupportsDOM );
  };
  var DOM = scenery.DOM;

  inherit( Node, DOM, {
    // we use a single DOM instance, so this flag should indicate that we don't support duplicating it
    allowsMultipleDOMInstances: false,

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

    updateDOMElement: function( container ) {
      // nothing needed, since we are just displaying a single DOM element
    },

    updateCSSTransform: function( transform, element ) {
      // faster to use our jQuery reference instead of wrapping element
      scenery.Util.applyCSSTransform( transform.getMatrix(), this._container );
    },

    isPainted: function() {
      return true;
    },

    setElement: function( element ) {
      if ( this._element !== element ) {
        if ( this._element ) {
          this._container.removeChild( this._element );
        }

        this._element = element;
        this._$element = $( element );

        this._container.appendChild( this._element );

        // TODO: bounds issue, since this will probably set to empty bounds and thus a repaint may not draw over it
        this.invalidateDOM();
      }

      return this; // allow chaining
    },

    getElement: function() {
      return this._element;
    },

    setInteractive: function( interactive ) {
      if ( this._interactive !== interactive ) {
        this._interactive = interactive;

        // TODO: anything needed here?
      }
    },

    isInteractive: function() {
      return this._interactive;
    },

    set element( value ) { this.setElement( value ); },
    get element() { return this.getElement(); },

    set interactive( value ) { this.setInteractive( value ); },
    get interactive() { return this.isInteractive(); },

    getBasicConstructor: function( propLines ) {
      return 'new scenery.DOM( $( \'' + escapeHTML( this._container.innerHTML.replace( /'/g, '\\\'' ) ) + '\' ), {' + propLines + '} )';
    },

    getPropString: function( spaces, includeChildren ) {
      var result = Node.prototype.getPropString.call( this, spaces, includeChildren );
      if ( this.interactive ) {
        if ( result ) {
          result += ',\n';
        }
        result += spaces + 'interactive: true';
      }
      return result;
    }
  } );

  DOM.prototype._mutatorKeys = [ 'element', 'interactive' ].concat( Node.prototype._mutatorKeys );

  return DOM;
} );


